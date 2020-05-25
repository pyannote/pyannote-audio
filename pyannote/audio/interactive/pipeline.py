#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 CNRS

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

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr


from typing import Text, Union, Tuple, List, Iterator, Dict
from pathlib import Path

import numpy as np
from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Uniform

from pyannote.audio.features.wrapper import Wrapper
from pyannote.audio.features import RawAudio
from pyannote.audio.features.utils import get_audio_duration

from pyannote.database.protocol.protocol import ProtocolFile
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Segment
from pyannote.core import Timeline
from pyannote.core import Annotation
from pyannote.core import SlidingWindow
from pyannote.core import SlidingWindowFeature
from pyannote.audio.utils.signal import Binarize
from pyannote.core.utils.hierarchy import pool
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


import prodigy
from prodigy.components.loaders import Audio
from prodigy.components.db import connect
import scipy.io.wavfile
import base64
import io
import copy

PRODIGY_SAMPLE_RATE = 16000


class InteractiveDiarization(Pipeline):
    """Interactive diarization pipeline

    Parameters
    ----------
    sad : str or Path, optional
        Pretrained speech activity detection model. Defaults to "sad".
    emb : str or Path, optional
        Pretrained speaker embedding model. Defaults to "emb".
    batch_size : int, optional
        Batch size.

    Hyper-parameters
    ----------------
    sad_threshold_on, sad_threshold_off : float
        Onset/offset speech activity detection thresholds.
    sad_min_duration_on, sad_min_duration_off : float
        Minimum duration of speech/non-speech regions.
    emb_duration, emb_step_ratio : float
        Sliding window used for embedding extraction.
    emb_threshold : float
        Distance threshold used as stopping criterion for hierarchical
        agglomerative clustering.
    """

    def __init__(
        self,
        sad: Union[Text, Path] = {"sad": {"duration": 2.0, "step": 0.1}},
        emb: Union[Text, Path] = "emb",
        batch_size: int = None,
    ):

        super().__init__()

        self.sad = Wrapper(sad)
        if batch_size is not None:
            self.sad.batch_size = batch_size
        self.sad_speech_index_ = self.sad.classes.index("speech")

        self.sad_threshold_on = Uniform(0.0, 1.0)
        self.sad_threshold_off = Uniform(0.0, 1.0)
        self.sad_min_duration_on = Uniform(0.0, 0.5)
        self.sad_min_duration_off = Uniform(0.0, 0.5)

        self.emb = Wrapper(emb)
        if batch_size is not None:
            self.emb.batch_size = batch_size

        max_duration = self.emb.duration
        min_duration = getattr(self.emb, "min_duration", 0.25 * max_duration)
        self.emb_duration = Uniform(min_duration, max_duration)
        self.emb_step_ratio = Uniform(0.1, 1.0)
        self.emb_threshold = Uniform(0.0, 2.0)

    def initialize(self):
        """Initialize pipeline internals with current hyper-parameter values"""

        self.sad_binarize_ = Binarize(
            onset=self.sad_threshold_on,
            offset=self.sad_threshold_off,
            min_duration_on=self.sad_min_duration_on,
            min_duration_off=self.sad_min_duration_off,
        )

        # embeddings will be extracted with a sliding window
        # of "emb_duration" duration and "emb_step_ratio x emb_duration" step.
        self.emb.duration = self.emb_duration
        self.emb.step = self.emb_step_ratio

    def compute_speech(self, current_file: ProtocolFile) -> Timeline:
        """Apply speech activity detection

        Parameters
        ----------
        current_file : ProtocolFile
            Protocol file.

        Returns
        -------
        speech : Timeline
            Speech activity detection result.
        """

        # speech activity detection
        if "sad_scores" in current_file:
            sad_scores: SlidingWindowFeature = current_file["sad_scores"]
        else:
            sad_scores: SlidingWindowFeature = self.sad(current_file)
            if np.nanmean(sad_scores) < 0:
                sad_scores = np.exp(sad_scores)
            current_file["sad_scores"] = sad_scores

        speech: Timeline = self.sad_binarize_.apply(
            sad_scores, dimension=self.sad_speech_index_
        )

        return speech

    def compute_embedding(self, current_file: ProtocolFile) -> SlidingWindowFeature:
        """Extract speaker embedding

        Parameters
        ----------
        current_file : ProtocolFile
            Protocol file

        Returns
        -------
        embedding : SlidingWindowFeature
            Speaker embedding.
        """

        return self.emb(current_file)

    def get_segment_assignment(
        self, embedding: SlidingWindowFeature, speech: Timeline
    ) -> np.ndarray:
        """Get segment assignment

        Parameters
        ----------
        embedding : SlidingWindowFeature
            Embeddings.
        speech : Timeline
            Speech regions.

        Returns
        -------
        assignment : (num_embedding, ) np.ndarray
            * assignment[i] = s with s > 0 means that ith embedding is strictly
            contained in (1-based) sth segment.
            * assignment[i] = s with s < 0 means that more than half of ith
            embedding is part of (1-based) sth segment.
            * assignment[i] = 0 means that none of the above is true.
        """

        assignment: np.ndarray = np.zeros((len(embedding),), dtype=np.int32)

        for s, segment in enumerate(speech):
            indices = embedding.sliding_window.crop(segment, mode="strict")
            if len(indices) > 0:
                strict = 1
            else:
                strict = -1
                indices = embedding.sliding_window.crop(segment, mode="center")
            for i in indices:
                if i < 0 or i >= len(embedding):
                    continue
                assignment[i] = strict * (s + 1)

        return assignment

    def __call__(self, current_file: ProtocolFile) -> Annotation:
        """Apply speaker diarization

        Parameters
        ----------
        current_file : ProtocolFile
            Protocol file.

        Returns
        -------
        diarization : Annotation
            Speaker diarization result.
        """

        if "duration" not in current_file:
            current_file["duration"] = get_audio_duration(current_file)

        # in "interactive annotation" mode, pipeline hyper-parameters are fixed.
        # therefore, there is no need to recompute embeddings every time a file
        # is processed: they can be passed with the file directly.
        if "embedding" in current_file:
            embedding: SlidingWindowFeature = current_file["embedding"]

        # in "pipeline optimization" mode, pipeline hyper-parameters are different
        # every time a file is processed: embeddings must be recomputed
        else:
            embedding: SlidingWindowFeature = self.compute_embedding(current_file)

        # in "interactive annotation" mode, there is no need to recompute speech
        # regions every time a file is processed: they can be passed with the
        # file directly
        if "speech" in current_file:
            speech: Timeline = current_file["speech"]

        # in "pipeline optimization" mode, pipeline hyper-parameters are different
        # every time a file is processed: speech regions must be recomputed
        else:
            speech: Timeline = self.compute_speech(current_file)

        # segment_assignment[i] = s with s > 0 means that ith embedding is
        #       strictly contained in (1-based) sth segment.
        # segment_assignment[i] = s with s < 0 means that more than half of ith
        #       embedding is part of (1-based) sth segment.
        # segment_assignment[i] = 0 means that none of the above is true.
        segment_assignment: np.ndarray = self.get_segment_assignment(embedding, speech)

        # cluster_assignment[i] = k (k > 0) means that the ith embedding belongs
        #                           to kth cluster
        # cluster_assignment[i] = 0 when segment_assignment[i] = 0
        cluster_assignment: np.ndarray = np.zeros((len(embedding),), dtype=np.int32)

        strict_indices = np.where(segment_assignment > 0)[0]
        if len(strict_indices) < 2:
            cluster_assignment[strict_indices] = 1

        else:
            dendrogram = pool(embedding[strict_indices], metric="cosine")
            clusters = fcluster(dendrogram, self.emb_threshold, criterion="distance")
            for i, k in zip(strict_indices, clusters):
                cluster_assignment[i] = k

        loose_indices = np.where(segment_assignment < 0)[0]
        if len(strict_indices) == 0:
            if len(loose_indices) < 2:
                clusters = [1] * len(loose_indices)
            else:
                dendrogram = pool(embedding[loose_indices], metric="cosine")
                clusters = fcluster(
                    dendrogram, self.emb_threshold, criterion="distance"
                )
            for i, k in zip(loose_indices, clusters):
                cluster_assignment[i] = k

        else:
            # TODO. try distance to average instead
            distance = cdist(
                embedding[strict_indices], embedding[loose_indices], metric="cosine"
            )
            nearest_neighbor = np.argmin(distance, axis=0)
            for loose_index, nn in zip(loose_indices, nearest_neighbor):
                strict_index = strict_indices[nn]
                cluster_assignment[loose_index] = cluster_assignment[strict_index]

        # convert cluster assignment to pyannote.core.Annotation
        # (make sure to keep speech regions unchanged)
        hypothesis = Annotation(uri=current_file.get("uri", None))
        for s, segment in enumerate(speech):

            indices = np.where(segment_assignment == s + 1)[0]
            if len(indices) == 0:
                indices = np.where(segment_assignment == -(s + 1))[0]
                if len(indices) == 0:
                    continue

            clusters = cluster_assignment[indices]

            start, k = segment.start, clusters[0]
            change_point = np.diff(clusters) != 0
            for i, new_k in zip(indices[1:][change_point], clusters[1:][change_point]):
                end = embedding.sliding_window[i].middle
                hypothesis[Segment(start, end)] = k
                start = end
                k = new_k
            hypothesis[Segment(start, segment.end)] = k

        return hypothesis.support()

    def get_metric(self) -> DiarizationErrorRate:
        return DiarizationErrorRate(collar=0.0, skip_overlap=False)

    @staticmethod
    def iter_chunks(duration: float, chunk: float = 30) -> Iterator[Segment]:
        """Partition [0, duration] time range into smaller chunks

        Parameters
        ----------
        duration : float
            Total duration, in seconds.
        chunk : float, optional
            Chunk duration, in seconds. Defaults to 30.

        Yields
        ------
        focus : Segment
        """

        sliding_window = SlidingWindow(start=0.0, step=chunk, duration=chunk)
        whole = Segment(0, duration)
        for window in sliding_window(whole):
            yield window
        if window.end < duration:
            yield Segment(window.end, duration)

    @staticmethod
    def normalize_audio(waveform: np.ndarray) -> np.ndarray:
        """Normalize waveform for better display in Prodigy UI"""
        return waveform / (np.max(np.abs(waveform)) + 1e-8)

    @staticmethod
    def prodigy_base64_audio(waveform: np.ndarray) -> Text:
        with io.BytesIO() as content:
            scipy.io.wavfile.write(content, PRODIGY_SAMPLE_RATE, waveform)
            content.seek(0)
            b64 = base64.b64encode(content.read()).decode()
            b64 = f"data:audio/x-wav;base64,{b64}"
        return b64

    @staticmethod
    def prodigy_audio_spans(annotation: Annotation, focus: Segment = None) -> Dict:
        """Convert pyannote.core.Annotation to Prodigy's audio_spans

        Parameters
        ----------
        annotation : Annotation
            Annotation with t=0s time origin.
        focus : Segment, optional
            When provided, use its start time as audio_spans time origin.

        Returns
        -------
        audio_spans : list of dict
        """
        shift = 0.0 if focus is None else focus.start
        return [
            {"start": segment.start - shift, "end": segment.end - shift, "label": label}
            for segment, _, label in annotation.itertracks(yield_label=True)
        ]

    def prodigy_sad_manual_stream(
        self, source: Path, chunk: float = 30
    ) -> Iterator[Dict]:
        """Task stream for pyannote.sad.manual Prodigy recipe

        Parameters
        ----------
        source : Path
            Directory containing audio files to annotate.
        chunk : float, optional
            Split long audio files into shorter chunks of that many seconds each.
            Default to 30s.

        Yields
        ------
        task : dict
            Prodigy task with the following keys:
            "path" : path to audio file
            "text" : name of audio file
            "chunk" : chunk start and end times
            "audio" : base64 encoding of audio chunk
            "audio_spans" : speech spans detected by pretrained SAD model
            "audio_spans_original" : copy of "audio_spans"
            "meta" : additional meta-data displayed in Prodigy UI
        """

        raw_audio = RawAudio(sample_rate=PRODIGY_SAMPLE_RATE, mono=True)

        for audio_source in Audio(source):

            path = audio_source["path"]
            text = audio_source["text"]
            file = {"uri": text, "database": source, "audio": path}

            duration = get_audio_duration(file)
            file["duration"] = duration

            prodigy.log(f"RECIPE: detecting speech regions in '{path}'")

            speech: Annotation = self.compute_speech(file).to_annotation(
                generator=iter(lambda: "SPEECH", None)
            )

            if duration <= chunk:
                waveform = raw_audio.crop(file, Segment(0, duration))
                task_audio = self.prodigy_base64_audio(self.normalize_audio(waveform))
                task_audio_spans = self.prodigy_audio_spans(speech)

                yield {
                    "path": path,
                    "text": text,
                    "audio": task_audio,
                    "audio_spans": task_audio_spans,
                    "audio_spans_original": copy.deepcopy(task_audio_spans),
                    "chunk": {"start": 0, "end": duration},
                    "meta": {"file": text},
                    "recipe": "pyannote.sad.manual",
                }

            else:
                for focus in self.iter_chunks(duration, chunk=chunk):
                    task_text = f"{text} [{focus.start:.1f}, {focus.end:.1f}]"
                    waveform = raw_audio.crop(file, focus)
                    task_audio = self.prodigy_base64_audio(
                        self.normalize_audio(waveform)
                    )
                    task_audio_spans = self.prodigy_audio_spans(
                        speech.crop(focus, mode="intersection"), focus=focus
                    )

                    yield {
                        "path": path,
                        "text": task_text,
                        "audio": task_audio,
                        "audio_spans": task_audio_spans,
                        "audio_spans_original": copy.deepcopy(task_audio_spans),
                        "chunk": {"start": focus.start, "end": focus.end},
                        "meta": {
                            "file": text,
                            "start": f"{focus.start:.1f}",
                            "end": f"{focus.end:.1f}",
                        },
                        "recipe": "pyannote.sad.manual",
                    }

    @staticmethod
    def prodigy_sad_manual_before_db(examples: List[Dict]) -> List[Dict]:

        for eg in examples:

            # remove (heavy) base64 audio
            if "audio" in eg:
                del eg["audio"]

            # shift audio spans back to the whole file referential
            chunk = eg.get("chunk", None)
            if chunk is not None:
                start = chunk["start"]
                for span in eg["audio_spans"]:
                    span["start"] += start
                    span["end"] += start
                for span in eg["audio_spans_original"]:
                    span["start"] += start
                    span["end"] += start

        return examples

    @staticmethod
    def prodigy_sad_manual_load(
        dataset: Text, source: Path, text: Text, path: Text
    ) -> ProtocolFile:

        db = connect()

        examples = [
            eg
            for eg in db.get_dataset(dataset)
            if eg["recipe"] == "pyannote.sad.manual"
            and eg["path"] == path
            and eg["answer"] == "accept"
        ]

        speech = Timeline(
            segments=[
                Segment(span["start"], span["end"])
                for eg in examples
                for span in eg["audio_spans"]
            ]
        ).support()

        prodigy.log(f"RECIPE: {path}: loaded speech regions")

        return {
            "uri": text,
            "database": source,
            "audio": path,
            "speech": speech,
        }

    @staticmethod
    def prodigy_dia_binary_before_db(examples: List[Dict]) -> List[Dict]:

        for eg in examples:

            # remove (heavy) base64 audio
            if "audio" in eg:
                del eg["audio"]

        return examples

    @staticmethod
    def prodigy_dia_binary_load(
        dataset: Text, path: Text
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Load existing 'cannot link' and 'must link' constraints

        Parameters
        ----------
        dataset : str
            Dataset.
        path : str
            Path to audio file.

        Returns
        -------
        cannot_link : list
            List of "cannot link" constraints. For instance, [(1., 3.5)] means
            that t=1s and t=3.5s cannot end up in the same cluster.
        must_link : list
            List of "must link" constraints. For instance, [(1., 3.5)] means
            that t=1s and t=3.5s must end up in the same cluster.
        dont_know : list
            List of "don't know" annotations.
        """

        db = connect()

        examples = [
            eg
            for eg in db.get_dataset(dataset)
            if eg["recipe"] == "pyannote.dia.binary" and eg["path"] == path
        ]

        cannot_link = [
            (eg["t1"], eg["t2"]) for eg in examples if eg["answer"] == "reject"
        ]
        must_link = [
            (eg["t1"], eg["t2"]) for eg in examples if eg["answer"] == "accept"
        ]
        dont_know = [
            (eg["t1"], eg["t2"])
            for eg in examples
            if eg["answer"] not in ["accept", "reject"]
        ]

        if len(cannot_link) > 0:
            prodigy.log(
                f"RECIPE: {path}: init: {len(cannot_link)} cannot link constraints"
            )
        if len(must_link) > 0:
            prodigy.log(f"RECIPE: {path}: init: {len(must_link)} must link constraints")

        return cannot_link, must_link, dont_know

    @staticmethod
    def time2clean(
        constraints_time: List[Tuple[float, float]],
        window: SlidingWindow,
        assignment: np.ndarray,
        all2clean: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """Convert time-based constraints to index in clean embedding

        Parameters
        ----------
        constraints_time : list of time pairs
            Time-based constraints
        window : SlidingWindow
            Window used for embedding extraction
        assignment : np.ndarray

        Returns
        -------
        constraints : list of index pairs
            Index-based constraints in clean embeddings array.

        """

        constraints = []
        for t1, t2 in constraints_time:
            i1 = window.closest_frame(t1)
            if assignment[i1] <= 0:
                continue
            i2 = window.closest_frame(t2)
            if assignment[i2] <= 0:
                continue
            constraints.append((all2clean[i1], all2clean[i2]))
        return constraints

    def prodigy_dia_binary_update(self, examples):

        for eg in examples:

            t1, t2 = eg["t1"], eg["t2"]

            if eg["answer"] == "accept":
                self.must_link_time.append((t1, t2))
                prodigy.log(f"RECIPE: {self.path}: new constraint: must link")

            elif eg["answer"] == "reject":
                self.cannot_link_time.append((t1, t2))
                prodigy.log(f"RECIPE: {self.path}: new constraint: cannot link")

            else:
                self.dont_know_time.append((t1, t2))
                prodigy.log(f"RECIPE: {self.path}: new constraint: skip")

    def prodigy_dia_binary_stream(self, dataset: Text, source: Path) -> Iterator[Dict]:

        raw_audio = RawAudio(sample_rate=PRODIGY_SAMPLE_RATE, mono=True)

        for audio_source in Audio(source):

            path = audio_source["path"]
            self.path = path
            text = audio_source["text"]

            # load human-validated speech activity detection
            file = self.prodigy_sad_manual_load(dataset, source, text, path)

            if not file["speech"]:
                prodigy.log(f"RECIPE: {path}: skip: no annotated speech")
                continue
            speech = file["speech"]

            # load human-validated (time-based) clustering constraints
            (
                self.cannot_link_time,
                self.must_link_time,
                self.dont_know_time,
            ) = self.prodigy_dia_binary_load(dataset, path)

            # compute embeddings for current file
            prodigy.log(f"RECIPE: {path}: extracting speaker embeddings")
            embedding = self.compute_embedding(file)
            window = embedding.sliding_window

            # number of consecutive steps with overlap
            n_steps = int(np.ceil(window.duration / window.step))

            # extract embeddings fully included in speech regions
            assignment = self.get_segment_assignment(embedding, speech)
            clean_embedding = embedding[assignment > 0]

            if len(clean_embedding) < 2:
                prodigy.log(f"RECIPE: {path}: skip: not enough speech")
                continue

            # convert
            all2clean = np.cumsum(assignment > 0) - 1
            clean2all = np.arange(len(embedding))[assignment > 0]

            done_with_current_file = False
            while not done_with_current_file:

                # IMPROVE do not recompute if no new constraint since last time

                # filter and convert time-based constraints in whole file referential
                # to index-based constraints in clean-only embeddings referential
                cannot_link = self.time2clean(
                    self.cannot_link_time, window, assignment, all2clean
                )
                must_link = self.time2clean(
                    self.must_link_time, window, assignment, all2clean
                )

                prodigy.log(f"RECIPE: {path}: applying constrained clustering")

                dendrogram = pool(
                    clean_embedding,
                    metric="cosine",
                    cannot_link=cannot_link if cannot_link else None,
                    must_link=must_link if must_link else None,
                )

                # iterate from dendrogram top to bottom
                iterations = iter(range(len(dendrogram) - 1, 0, -1))

                # IDEA instead of iterating from top to bottom,
                # we could start by the iteration whose merging distance
                # is the most similar to an "optimal" distance and then
                # progressively wander away from it
                # FIXME make sure iteration is not < 1
                # iterations = iter(np.argsort(
                #     np.abs(pipeline.emb_threshold - dendrogram[:, 2])
                # ))

                # IDEA we could stop annotation early once the
                # current distance is very very small (and we can be sure
                # that all iterations up to this point are correct)
                # TODO

                while True:

                    try:
                        i = next(iterations)
                    except StopIteration as e:
                        done_with_current_file = True
                        break

                    distance = dendrogram[i, 2]

                    # if distance is infinite, this is a fake clustering step
                    # prevented by a "cannot link" constraint.
                    # see pyannote.core.hierarchy.pool for details
                    if distance == np.infty:
                        prodigy.log(f"RECIPE: {path}: depth {i}: skip: cannot link")
                        continue

                    # find clusters k1 and k2 that were merged at iteration i
                    current = fcluster(
                        dendrogram, dendrogram[i, 2], criterion="distance"
                    )
                    previous = fcluster(
                        dendrogram, dendrogram[i - 1, 2], criterion="distance",
                    )
                    n_current, n_previous = max(current), max(previous)

                    # TODO handle these corner cases better
                    if n_current >= n_previous or n_previous - n_current > 1:
                        prodigy.log(f"RECIPE: {path}: depth {i}: skip: corner case")
                        continue
                    C = np.zeros((n_current, n_previous))
                    for k_current, k_previous in zip(current, previous):
                        C[k_current - 1, k_previous - 1] += 1
                    k1, k2 = (
                        np.where(C[int(np.where(np.sum(C > 0, axis=1) == 2)[0])] > 0)[0]
                        + 1
                    )

                    # find indices of embeddings fully included in clusters k1 and k2
                    neighbors1 = np.convolve(previous == k1, [1] * n_steps, mode="same")
                    indices1 = np.where(neighbors1 == n_steps)[0]
                    # if indices1.size == 0:
                    #     indices1 = np.where(neighbors1 == np.max(neighbors1))[0]

                    neighbors2 = np.convolve(previous == k2, [1] * n_steps, mode="same")
                    indices2 = np.where(neighbors2 == n_steps)[0]
                    # if indices2.size == 0:
                    #     indices2 = np.where(neighbors2 == np.max(neighbors2))[0]

                    if indices1.size == 0 or indices2.size == 0:
                        prodigy.log(
                            f"RECIPE: {path}: depth {i}: skip: too short segments"
                        )
                        continue

                    # find centroids of clusters k1 and k2
                    i1 = indices1[
                        np.argmin(
                            np.mean(
                                squareform(
                                    pdist(clean_embedding[indices1], metric="cosine")
                                ),
                                axis=1,
                            )
                        )
                    ]

                    i2 = indices2[
                        np.argmin(
                            np.mean(
                                squareform(
                                    pdist(clean_embedding[indices2], metric="cosine")
                                ),
                                axis=1,
                            )
                        )
                    ]

                    i1, i2 = sorted([i1, i2])
                    distance = cdist(
                        clean_embedding[np.newaxis, i1],
                        clean_embedding[np.newaxis, i2],
                        metric="cosine",
                    )[0, 0]

                    segment1 = window[clean2all[i1]]
                    t1 = segment1.middle
                    segment2 = window[clean2all[i2]]
                    t2 = segment2.middle

                    # did the human in the loop already provide feedback on this pair of segments?
                    pair = (t1, t2)

                    if (
                        pair in self.cannot_link_time
                        or pair in self.must_link_time
                        or pair in self.dont_know_time
                    ):
                        # do not annotate the same pair twice
                        prodigy.log(f"RECIPE: {path}: depth {i}: skip: exists")
                        continue

                    prodigy.log(f"RECIPE: {path}: depth {i}: annotate")

                    task_text = f"{text} t={t1:.1f}s vs. t={t2:.1f}s"

                    waveform1 = self.normalize_audio(raw_audio.crop(file, segment1))
                    waveform2 = self.normalize_audio(raw_audio.crop(file, segment2))
                    task_audio = self.prodigy_base64_audio(
                        np.vstack([waveform1, waveform2])
                    )
                    # task_audio_spans = [
                    #     {
                    #         "start": segment1.middle
                    #         - 0.5 * window.step
                    #         - segment1.start,
                    #         "end": segment1.middle + 0.5 * window.step - segment1.start,
                    #         "label": "SPEAKER",
                    #     },
                    #     {
                    #         "start": segment1.duration
                    #         + segment2.middle
                    #         - 0.5 * window.step
                    #         - segment2.start,
                    #         "end": segment1.duration
                    #         + segment2.middle
                    #         + 0.5 * window.step
                    #         - segment2.start,
                    #         "label": "SAME_SPEAKER",
                    #     },
                    # ]

                    task_audio_spans = [
                        {"start": 0.0, "end": segment1.duration, "label": "SPEAKER",},
                        {
                            "start": segment1.duration,
                            "end": segment1.duration + segment2.duration,
                            "label": "SAME_SPEAKER",
                        },
                    ]

                    yield {
                        "path": path,
                        "text": task_text,
                        "audio": task_audio,
                        "audio_spans": task_audio_spans,
                        "t1": t1,
                        "t2": t2,
                        "meta": {
                            "t1": f"{t1:.1f}s",
                            "t2": f"{t2:.1f}s",
                            "file": text,
                            "distance": f"{distance:.2f}",
                        },
                        "recipe": "pyannote.dia.binary",
                    }

                    # at that point, "prodigy_dia_binary_update" is called. hence,
                    # we exit the loop because the dendrogram needs to be updated
                    break
