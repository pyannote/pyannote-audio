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
from pathlib import Path

from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment


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

    def _to_annotation(self, completed_job: dict) -> Annotation:
        """Deserialize job output into pyannote.core Annotation"""

        output = completed_job["output"]["diarization"]
        job_id = completed_job["jobId"]

        annotation = Annotation(uri=job_id)
        for t, turn in enumerate(output):
            segment = Segment(start=turn["start"], end=turn["end"])
            speaker = turn["speaker"]
            annotation[segment, t] = speaker

        return annotation.rename_tracks("string")

    def apply(
        self,
        file: Path,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        exclusive: bool = False,
    ) -> Annotation:
        """Speaker diarization using pyannoteAI on-premise package

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
        exclusive : bool, optional
            Enable exclusive diarization.

        Returns
        -------
        speaker_diarization : Annotation
            Speaker diarization result (when successful)

        Raises
        ------
        PyannoteAIFailedJob
            If the job failed
        PyannoteAICanceledJob
            If the job was canceled
        HTTPError
            If something else went wrong
        """

        predictions = self._pipeline.diarize(
            file["audio"],
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        # use exclusive diarization whenever requested
        if exclusive:
            diarization = predictions["exclusive_diarization"]
        else:
            diarization = predictions["diarization"]

        # deserialize the output into a good-old Annotation instance
        annotation = Annotation()
        for t, turn in enumerate(diarization):
            segment = Segment(start=turn["start"], end=turn["end"])
            speaker = turn["speaker"]
            annotation[segment, t] = speaker
        return annotation.rename_tracks("string")
