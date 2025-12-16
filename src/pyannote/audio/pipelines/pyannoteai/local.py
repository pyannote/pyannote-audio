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

import os

from pyannote.audio import Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.core import Annotation, Segment

from ..speaker_diarization import DiarizeOutput


class Local(Pipeline):
    """Wrapper around official pyannoteAI on-premise package

    Parameters
    ----------
    token : str, optional
        pyannoteAI API key created from https://dashboard.pyannote.ai.
        Defaults to using `PYANNOTEAI_API_KEY` environment variable.

    Usage
    -----
    >>> from pyannote.audio.pipelines.pyannoteai.local import Local
    >>> pipeline = Local(token="{PYANNOTEAI_API_KEY}")
    >>> speaker_diarization = pipeline("/path/to/your/audio.wav")
    """

    def __init__(self, token: str | None = None, **kwargs):
        super().__init__()

        from pyannoteai.local import Pipeline as _LocalPipeline

        self.token = token or os.environ.get("PYANNOTEAI_API_KEY", None)
        self._pipeline = _LocalPipeline(self.token)

    def _deserialize(self, diarization: list[dict]) -> Annotation:
        # deserialize the output into a good-old Annotation instance
        annotation = Annotation()
        for t, turn in enumerate(diarization):
            segment = Segment(start=turn["start"], end=turn["end"])
            speaker = turn["speaker"]
            annotation[segment, t] = speaker
        return annotation.rename_tracks("string")

    def apply(
        self,
        file: AudioFile,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        **kwargs,
    ) -> DiarizeOutput:
        """Speaker diarization using on-premise pyannoteAI package.

        Parameters
        ----------
        file : AudioFile
            Processed file.
        num_speakers : int, optional
            Force number of speakers to diarize. If not provided, the
            number of speakers will be determined automatically.
        min_speakers : int, optional
            Not supported yet. Minimum number of speakers. Has no effect when `num_speakers` is provided.
        max_speakers : int, optional
            Not supported yet. Maximum number of speakers. Has no effect when `num_speakers` is provided.

        Returns
        -------
        output : DiarizeOutput
            DiarizeOutput object containing both regular and exclusive speaker diarization results.
        """

        # if file provides "audio" path
        if "audio" in file:
            predictions = self._pipeline.diarize(
                file["audio"],
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                **kwargs,
            )

        # if file provides "waveform", make sure it is numpy (and not torch) array
        elif "waveform" in file:
            waveform = file["waveform"]
            if hasattr(waveform, "numpy"):
                waveform = waveform.numpy(force=True)

            predictions = self._pipeline.diarize(
                {"waveform": waveform, "sample_rate": file["sample_rate"]},
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                **kwargs,
            )
        else:
            raise ValueError("AudioFile must provide either 'audio' or 'waveform' key")

        speaker_diarization: Annotation = self._deserialize(predictions["diarization"])
        exclusive_speaker_diarization: Annotation = self._deserialize(
            predictions["exclusive_diarization"]
        )

        return DiarizeOutput(
            speaker_diarization=speaker_diarization,
            exclusive_speaker_diarization=exclusive_speaker_diarization,
        )
