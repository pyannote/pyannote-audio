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

from pyannoteai.sdk import Client

from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput

class SDK(Pipeline):
    """Wrapper around official pyannoteAI API client

    Parameters
    ----------
    model : str, optional
        pyannoteAI speaker diarization model.
        Defaults to "precision-2".
    token : str, optional
        pyannoteAI API key created from https://dashboard.pyannote.ai.
        Defaults to using `PYANNOTEAI_API_KEY` environment variable.

    Usage
    -----
    >>> from pyannote.audio.pipelines.pyannoteai.sdk import SDK
    >>> pipeline = SDK(token="{PYANNOTEAI_API_KEY}")
    >>> speaker_diarization = pipeline("/path/to/your/audio.wav")
    """

    def __init__(self, model: str = "precision-2", token: str | None = None, **kwargs):
        super().__init__()

        self.model = model
        self.token = token or os.environ.get("PYANNOTEAI_API_KEY", None)
        self._client = Client(self.token)

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
    ) -> DiarizeOutput:
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
            model=self.model,
            exclusive=True,
        )

        # retrieve job output (once completed)
        job_output = self._client.retrieve(job_id)

        speaker_diarization: Annotation = self._deserialize(job_output["output"]["diarization"])
        exclusive_speaker_diarization: Annotation = self._deserialize(job_output["output"]["exclusiveDiarization"])

        return DiarizeOutput(
            speaker_diarization=speaker_diarization,
            exclusive_speaker_diarization=exclusive_speaker_diarization,
        )

