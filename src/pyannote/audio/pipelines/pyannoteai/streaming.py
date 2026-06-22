# The MIT License (MIT)
#
# Copyright (c) 2025- pyannoteAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Callable, Iterable

import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from pyannote.audio import Pipeline
from pyannote.audio.core.io import Audio, AudioFile
from pyannote.core import Annotation, Segment

from ..speaker_diarization import DiarizeOutput

# pyannoteAI streaming API expects 16kHz mono float32 PCM, sent in 100ms chunks
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)  # 1600 samples


class FakeStreamingDiarization(Pipeline):
    """Benchmark wrapper around the pyannoteAI streaming diarization API

    The pyannoteAI streaming (a.k.a. "live") API is designed to consume a
    real-time audio stream (e.g. from a microphone). This wrapper "fakes" such
    a stream by reading a pre-recorded file from disk and feeding it to the API
    chunk by chunk, then reconstructs a regular `pyannote.core.Annotation` from
    the `diarization_speaker_start` / `diarization_speaker_end` events emitted
    by the server. This makes it possible to benchmark the streaming model with
    the exact same tooling used for the offline pipelines.

    See https://docs.pyannote.ai/tutorials/streaming-diarized-transcription

    Parameters
    ----------
    token : str, optional
        pyannoteAI API key created from https://dashboard.pyannote.ai.
        Defaults to using `PYANNOTEAI_API_KEY` environment variable.
    base_url : str, optional
        Base URL of the pyannoteAI API. Defaults to "https://api.pyannote.ai/v1".
    concurrency : int, optional
        Maximum number of files to stream in parallel in `apply_many`.
        Defaults to 4.

    Note
    ----
    The streaming API only accepts audio at (or below) real-time pace: it
    enforces a maximum 5-second buffer and closes the connection if fed faster
    than real-time. Processing a file therefore takes roughly as long as the
    file's duration.

    Usage
    -----
    >>> import os
    >>> from pyannote.audio.pipelines.pyannoteai import FakeStreamingDiarization
    >>> pipeline = FakeStreamingDiarization(os.environ["PYANNOTEAI_API_KEY"])
    >>> output = pipeline("/path/to/your/audio.wav")
    >>> output.speaker_diarization  # pyannote.core.Annotation

    Multiple files can be streamed in parallel (up to `concurrency` at a time).
    `apply_many` is a context manager yielding a `submit` callable: each call
    schedules a file and immediately returns a `concurrent.futures.Future`, so
    results (and failures) can be tracked per file.

    >>> results = {}
    >>> with pipeline.apply_many() as submit:
    ...     for file in ["a.wav", "b.wav", "c.wav"]:
    ...         results[file] = submit(file)  # returns a Future
    >>> # on exit, every submitted file has finished streaming
    >>> for file, future in results.items():
    ...     try:
    ...         output = future.result()
    ...     except Exception as error:
    ...         print(f"{file} failed: {error}")

    A convenience eager form is also available; it blocks until every file is
    done and returns outputs in order (raising on the first failure):

    >>> outputs = pipeline.apply_many(["a.wav", "b.wav", "c.wav"])
    """

    def __init__(
        self,
        token: str | None = None,
        base_url: str = "https://api.pyannote.ai/v1",
        concurrency: int = 4,
        **kwargs,
    ):
        super().__init__()

        self.token = token or os.environ.get("PYANNOTEAI_API_KEY", None)
        if not self.token:
            raise ValueError(
                "A pyannoteAI API key must be provided either through the `token` "
                "argument or the `PYANNOTEAI_API_KEY` environment variable."
            )

        self.base_url = base_url.rstrip("/")
        self.concurrency = concurrency

        # 16kHz mono reader, as expected by the streaming API
        self._audio = Audio(sample_rate=SAMPLE_RATE, mono="downmix")

    def _open_stream(self) -> str:
        """Create a streaming session and return its (single-use) WebSocket URL"""
        try:
            import requests
        except ImportError as e:
            raise ImportError(
                "`requests` is required to use `FakeStreamingDiarization`. "
                "Install it with `pip install requests`."
            ) from e

        response = requests.post(
            f"{self.base_url}/live",
            headers={"Authorization": f"Bearer {self.token}"},
            json={},
        )
        response.raise_for_status()
        return response.json()["url"]

    def _stream(
        self,
        waveform: np.ndarray,
        url: str,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        """Stream `waveform` to `url` and collect all server events (in order)

        If provided, `on_progress` is called after each chunk is sent with the
        number of chunks sent so far and the total number of chunks.
        """
        try:
            import websocket
        except ImportError as e:
            raise ImportError(
                "`websocket-client` is required to use `FakeStreamingDiarization`. "
                "Install it with `pip install websocket-client`."
            ) from e

        events: list[dict] = []
        receiver_error: list[Exception] = []

        ws = websocket.create_connection(url)

        def receive():
            # collect server events until the connection is closed (code 1000
            # once the server has finalized all remaining diarization events)
            try:
                while True:
                    raw = ws.recv()
                    if not raw:
                        break
                    events.append(json.loads(raw))
            except websocket.WebSocketConnectionClosedException:
                pass
            except Exception as e:  # noqa: BLE001
                receiver_error.append(e)

        receiver = threading.Thread(target=receive, daemon=True)
        receiver.start()

        # send audio as 100ms chunks of little-endian float32 PCM, zero-padding
        # the very last chunk so that every frame is exactly CHUNK_SIZE samples.
        #
        # the server enforces a max 5s buffer and closes the connection when fed
        # faster than real-time, so chunk `i` is sent no earlier than `i * 100ms`
        # after the first one. absolute (rather than cumulative) scheduling keeps
        # the stream real-time without accumulating drift.
        clock = time.monotonic()
        total = (len(waveform) + CHUNK_SIZE - 1) // CHUNK_SIZE
        for index, start in enumerate(range(0, len(waveform), CHUNK_SIZE)):
            chunk = waveform[start : start + CHUNK_SIZE]
            if len(chunk) < CHUNK_SIZE:
                chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))

            delay = (clock + index * CHUNK_DURATION) - time.monotonic()
            if delay > 0:
                time.sleep(delay)

            ws.send(chunk.astype("<f4").tobytes(), websocket.ABNF.OPCODE_BINARY)

            if on_progress is not None:
                on_progress(index + 1, total)

        # tell the server we are done; it will flush remaining events and close
        ws.send(json.dumps({"type": "end_of_stream"}))

        receiver.join()
        try:
            ws.close()
        except Exception:  # noqa: BLE001
            pass

        if receiver_error:
            raise receiver_error[0]

        return events

    def _deserialize(self, events: list[dict], duration: float) -> Annotation:
        """Reconstruct an `Annotation` from streaming speaker start/end events"""
        annotation = Annotation()

        # timestamp of the last unmatched "start" event, per speaker
        pending: dict[str, float] = {}
        track = 0

        for event in events:
            event_type = event.get("type")
            data = event.get("data", {})

            if event_type == "diarization_speaker_start":
                pending[data["speaker"]] = data["timestamp"]

            elif event_type == "diarization_speaker_end":
                speaker = data["speaker"]
                start = pending.pop(speaker, None)
                if start is None:
                    # "end" without a matching "start": ignore
                    continue
                annotation[Segment(start, data["timestamp"]), track] = speaker
                track += 1

            elif event_type == "error":
                raise RuntimeError(
                    f"pyannoteAI streaming API error: {event.get('message')}"
                )

        # close speaker turns that were still active when the stream ended
        for speaker, start in pending.items():
            annotation[Segment(start, duration), track] = speaker
            track += 1

        return annotation.rename_tracks("string")

    def apply(
        self,
        file: AudioFile,
        on_progress: Callable[[int, int], None] | None = None,
        **kwargs,
    ) -> DiarizeOutput:
        """Speaker diarization by streaming `file` to the pyannoteAI live API

        Parameters
        ----------
        file : AudioFile
            Processed file.
        on_progress : callable, optional
            Called after each 100ms chunk is streamed with the number of chunks
            sent so far and the total number of chunks (both `int`).

        Returns
        -------
        output : DiarizeOutput
            Diarization output, whose `speaker_diarization` attribute is the
            reconstructed `pyannote.core.Annotation`.
        """

        # load the whole file as 16kHz mono float32 PCM
        waveform, _ = self._audio(file)
        waveform = waveform.numpy(force=True).reshape(-1).astype(np.float32)
        duration = len(waveform) / SAMPLE_RATE

        url = self._open_stream()
        events = self._stream(waveform, url, on_progress=on_progress)

        speaker_diarization = self._deserialize(events, duration)
        try:
            speaker_diarization.uri = file["uri"]
        except (KeyError, TypeError):
            pass

        # the streaming API does not provide a separate overlap-free diarization;
        # for benchmarking purposes we expose the same annotation under both keys
        return DiarizeOutput(
            speaker_diarization=speaker_diarization
        )

    @staticmethod
    def _uri(file: AudioFile) -> str:
        """Return the `uri` of `file`, enforcing that it exposes one

        `apply_many` identifies files by their `uri`, so every file must be a
        mapping (e.g. a `dict`) exposing a `uri` key.
        """
        try:
            uri = file["uri"]
        except (KeyError, TypeError):
            raise ValueError(
                "apply_many requires every file to expose a `uri` key, e.g. "
                '`{"uri": "my-recording", "audio": "/path/to/audio.wav"}`. '
                f"Got: {file!r}"
            ) from None
        return str(uri)

    @staticmethod
    def _label(file: AudioFile) -> str:
        """Best-effort short, human-readable name for `file` (used in progress bars)"""
        if isinstance(file, (str, Path)):
            return Path(file).name
        if isinstance(file, Mapping):
            if "uri" in file:
                return str(file["uri"])
            if "audio" in file and isinstance(file["audio"], (str, Path)):
                return Path(file["audio"]).name
        return str(file)

    def apply_many(
        self,
        files: Iterable[AudioFile] | None = None,
        concurrency: int | None = None,
        show_progress: bool = True,
    ) -> "_StreamingBatch | dict[str, DiarizeOutput]":
        """Stream several files to the pyannoteAI live API in parallel

        Because the streaming API runs at (at most) real-time pace, processing a
        file takes roughly as long as its duration. Streaming several files at
        once is therefore the simplest way to speed up benchmarking.

        Files are identified by their `uri`, so every file must be a mapping
        (e.g. a `dict`) exposing a `uri` key, such as
        `{"uri": "my-recording", "audio": "/path/to/audio.wav"}`.

        This method has two forms.

        As a context manager (recommended), it yields a `submit` callable. Each
        `submit(file)` call schedules `file` for streaming and immediately
        returns a `concurrent.futures.Future`. Leaving the `with` block blocks
        until every submitted file has finished. This lets you keep track of
        each file individually, including failures:

        >>> results = {}
        >>> with pipeline.apply_many() as submit:
        ...     for file in files:
        ...         results[file["uri"]] = submit(file)  # returns a Future
        >>> for uri, future in results.items():
        ...     try:
        ...         output = future.result()
        ...     except Exception as error:
        ...         print(f"{uri} failed: {error}")

        As an eager helper, passing `files` directly blocks until all files are
        done and returns a `{uri: output}` dictionary (raising on the first
        failure):

        >>> outputs = pipeline.apply_many(files)  # {uri: DiarizeOutput}

        Parameters
        ----------
        files : iterable of AudioFile, optional
            Files to process. Each file must expose a `uri` key. If omitted, a
            `_StreamingBatch` context manager is returned instead of a dict of
            outputs (see above).
        concurrency : int, optional
            Maximum number of files to stream in parallel. Defaults to
            `self.concurrency`.
        show_progress : bool, optional
            Show one progress bar per concurrent stream. Defaults to True.

        Returns
        -------
        batch : _StreamingBatch
            When `files` is omitted: a context manager yielding a `submit`
            callable (`submit(file) -> Future[DiarizeOutput]`).
        outputs : dict of {str: DiarizeOutput}
            When `files` is provided: diarization outputs keyed by `uri`.

        Raises
        ------
        ValueError
            If any file does not expose a `uri` key, or if two files share the
            same `uri`.
        """
        concurrency = self.concurrency if concurrency is None else concurrency
        batch = _StreamingBatch(self, concurrency, show_progress)

        # context-manager form: hand the batch back for the caller to drive
        if files is None:
            return batch

        # eager form: submit everything, then block until all are done and
        # return a {uri: output} dict (raising on the first failure)
        files = list(files)
        if not files:
            return {}

        # resolve (and validate) uris up-front, rejecting duplicates that would
        # otherwise silently collide in the returned dict
        uris = [self._uri(file) for file in files]
        seen = set()
        duplicates = sorted({uri for uri in uris if uri in seen or seen.add(uri)})
        if duplicates:
            raise ValueError(
                f"apply_many requires every file to expose a unique `uri`; "
                f"got duplicates: {duplicates}."
            )

        with batch as submit:
            futures = {uri: submit(file) for uri, file in zip(uris, files)}
        return {uri: future.result() for uri, future in futures.items()}


class _StreamingBatch:
    """Context manager driving a batch of parallel streams for `apply_many`

    Entering the context starts a `ThreadPoolExecutor` (and, optionally, a pool
    of reusable progress bars) and yields a `submit` callable. Each
    `submit(file)` call schedules `file` and returns a
    `concurrent.futures.Future` resolving to its `DiarizeOutput` (or carrying
    the exception raised while processing it). Leaving the context shuts the
    executor down, blocking until every submitted file has finished.

    This class is not meant to be instantiated directly; use
    `FakeStreamingDiarization.apply_many()`.
    """

    def __init__(
        self,
        pipeline: "FakeStreamingDiarization",
        concurrency: int,
        show_progress: bool,
    ):
        self._pipeline = pipeline
        self._concurrency = max(1, concurrency)
        self._show_progress = show_progress

        self._executor: ThreadPoolExecutor | None = None
        self._progress: Progress | None = None
        # a pool of `concurrency` reusable progress bars (one per concurrent
        # stream); each bar is borrowed by a worker for the duration of a file,
        # then returned for the next pending file to reuse.
        self._slots: "Queue[int]" = Queue()

    def __enter__(self) -> Callable[[AudioFile], "Future[DiarizeOutput]"]:
        self._executor = ThreadPoolExecutor(max_workers=self._concurrency)

        if self._show_progress:
            self._progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(elapsed_when_finished=True),
            )
            for _ in range(self._concurrency):
                self._slots.put(
                    self._progress.add_task("", total=None, visible=False)
                )
            self._progress.start()

        return self.submit

    def submit(self, file: AudioFile) -> "Future[DiarizeOutput]":
        """Schedule `file` for streaming and return a `Future` for its output"""
        if self._executor is None:
            raise RuntimeError(
                "submit() can only be called inside the `apply_many()` "
                "`with` block."
            )
        # enforce a `uri` key eagerly, so the error surfaces at the submit call
        # rather than being buried inside the returned Future
        self._pipeline._uri(file)
        return self._executor.submit(self._process, file)

    def _process(self, file: AudioFile) -> DiarizeOutput:
        if self._progress is None:
            return self._pipeline.apply(file)

        # borrow a progress bar for the duration of this file
        task_id = self._slots.get()
        self._progress.reset(
            task_id,
            description=self._pipeline._label(file),
            total=None,
            visible=True,
        )

        def on_progress(completed: int, total: int) -> None:
            self._progress.update(task_id, completed=completed, total=total)

        try:
            return self._pipeline.apply(file, on_progress=on_progress)
        finally:
            self._progress.update(task_id, visible=False)
            self._slots.put(task_id)

    def __exit__(self, *exc_info) -> bool:
        try:
            # block until every submitted file has finished
            self._executor.shutdown(wait=True)
        finally:
            self._executor = None
            if self._progress is not None:
                self._progress.stop()
                self._progress = None
        return False
