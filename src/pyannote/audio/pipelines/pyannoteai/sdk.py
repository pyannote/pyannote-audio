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
from typing import Optional

from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment

from pyannoteai.sdk import Client


class SDK(Pipeline):
    """Wrapper around official pyannoteAI API client

    Parameters
    ----------
    token : str, optional
        pyannoteAI API key created from https://dashboard.pyannote.ai.
        Defaults to using `PYANNOTEAI_API_TOKEN` environment variable.

    Usage
    -----
    >>> from pyannote.audio.pipelines.pyannoteai.sdk import SDK
    >>> pipeline = SDK(token="{PYANNOTEAI_API_KEY}")
    >>> speaker_diarization = pipeline("/path/to/your/audio.wav")
    """

    def __init__(self, token: str | None = None, **kwargs):
        super().__init__()

        self.token = token or os.environ.get("PYANNOTEAI_API_KEY", None)
        self._client = Client(self.token)

    def apply(
        self,
        file: Path,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        model: str = "precision-2",
        exclusive: bool = False,
    ) -> Annotation:
        """Speaker diarization using pyannoteAI web API

        This method will upload `file`, initiate a diarization job,
        retrieve its output, and deserialize the latter into a good
        old pyannote.core.Annotation instance.

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
        model : str, optional
            pyannoteAI diarization model to use. Defaults to "precision-2".
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

        # upload file to pyannoteAI cloud API
        media_url: str = self._client.upload(file)

        # initiate diarization job
        job_id = self._client.diarize(
            media_url,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            confidence=False,
            model=model,
            exclusive=exclusive,
        )

        # retrieve job output (once completed)
        job_output = self._client.retrieve(job_id)

        # use exclusive diarization whenever requested
        if exclusive:
            diarization: list[dict] = job_output["output"]["exclusiveDiarization"]
        else:
            diarization: list[dict] = job_output["output"]["diarization"]

        # deserialize the output into a good-old Annotation instance
        annotation = Annotation()
        for t, turn in enumerate(diarization):
            segment = Segment(start=turn["start"], end=turn["end"])
            speaker = turn["speaker"]
            annotation[segment, t] = speaker
        return annotation.rename_tracks("string")

