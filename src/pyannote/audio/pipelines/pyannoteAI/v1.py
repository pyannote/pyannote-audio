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

import hashlib
import io
import os
import time
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import requests
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature
from requests import Response
from requests.exceptions import ConnectionError, HTTPError

from pyannote.audio import Audio, Pipeline, __version__
from pyannote.audio.core.io import AudioFile


class PyannoteAIFailedJob(Exception):
    """Raised when a job failed on the pyannoteAI web API"""

    pass


class PyannoteAICanceledJob(Exception):
    """Raised when a job was canceled on the pyannoteAI web API"""

    pass


class UploadingCallbackBytesIO(io.BytesIO):
    """BytesIO subclass that calls a callback during the upload process

    Parameters
    ----------
    callback : Callable
        Callback called during upload as `callback(total_in_bytes, completed_in_bytes)`
    total_size : int
        Total size to upload (in bytes)
    initial_bytes : bytes
        Initial bytes to upload
    """

    def __init__(
        self,
        callback: Callable,
        total_size: int,
        initial_bytes: bytes,
    ):
        self.total_size = total_size
        self._completed_size = 0
        self._callback = callback
        super().__init__(initial_bytes)

    def read(self, size=-1) -> bytes:
        data = super().read(size)
        self._completed_size += len(data)
        if self._callback:
            self._callback(
                total=self.total_size,
                completed=self._completed_size,
            )
        return data


class V1(Pipeline):
    """Official client for pyannoteAI web API

    Parameters
    ----------
    token : str, optional
        pyannoteAI API key created from https://dashboard.pyannote.ai.
        Defaults to using `PYANNOTEAI_API_TOKEN` environment variable.

    Usage
    -----
    # instantiate client for pyannoteAI web API
    >>> from pyannote.audio.pipelines.pyannoteAI.v1 import V1
    >>> pipeline = V1(token="{PYANNOTEAI_API_KEY}")

    # upload your audio file to the pyannoteAI web API
    # and store it for a few hours for later re-use.
    >>> media_url = pipeline.upload("/path/to/your/audio.wav")

    # initiate a diarization job on the pyannoteAI web API
    >>> job_id = pipeline.diarize(media_url)

    # retrieve prediction from pyannoteAI web API
    >>> prediction = pipeline.retrieve(job_id)

    # convert to pyannote.core.Annotation instance
    >>> speaker_diarization = pipeline.deserialize(prediction)['diarization']

    # or do all of this at once with
    >>> speaker_diarization = pipeline("/path/to/your/audio.wav")
    """

    API_URL = "https://api.pyannote.ai/v1"

    def __init__(self, token: Optional[str] = None, **kwargs):
        super().__init__()
        self.token = token
        self.api_key = token or os.environ.get("PYANNOTEAI_API_KEY", "")

    def _raise_for_status(self, response: Response):
        """Raise an exception if the response status code is not 2xx"""

        if response.status_code == 401:
            raise HTTPError(
                """
Failed to authenticate to pyannoteAI API. Please create an API key on https://dashboard.pyannote.ai/ and
provide it either via `PYANNOTEAI_API_TOKEN` environment variable or with `token` parameter."""
            )

        # TODO: add support for other status code when
        # they are documented on docs.pyannote.ai

        response.raise_for_status()

    def _authenticated_get(self, route: str) -> Response:
        """Send GET authenticated request to pyannoteAI API

        Parameters
        ----------
        route : str
            API route to send the GET request to.

        Returns
        -------
        response : Response

        Raises
        ------
        ConnectionError
        HTTPError
        """

        try:
            response = requests.get(f"{self.API_URL}{route}", headers=self._headers)
        except ConnectionError:
            raise ConnectionError(
                """
Failed to connect to pyannoteAI web API. Please check your internet connection
or visit https://pyannote.openstatus.dev/ to check the status of the pyannoteAI web API."""
            )

        self._raise_for_status(response)

        return response

    def _authenticated_post(self, route: str, json: Optional[dict] = None) -> Response:
        """Send POST authenticated request to pyannoteAI web API

        Parameters
        ----------
        route : str
            API route to send the GET request to.
        json : dict, optional
            Request body to send with the POST request.

        Returns
        -------
        response : Response

        Raises
        ------
        ConnectionError
        HTTPError
        """

        try:
            response = requests.post(
                f"{self.API_URL}{route}", json=json, headers=self._headers
            )
        except ConnectionError:
            raise ConnectionError(
                """
Failed to connect to pyannoteAI web API. Please check your internet connection
or visit https://pyannote.openstatus.dev/ to check the status of the pyannoteAI web API."""
            )

        self._raise_for_status(response)

        return response

    @property
    def api_key(self):
        return self._api_key

    @api_key.setter
    def api_key(self, api_key: str) -> None:
        if not api_key:
            raise ValueError(
                """
        Failed to authenticate to pyannoteAI web API. Please create an API key on https://dashboard.pyannote.ai/ and
        provide it either via `PYANNOTEAI_API_TOKEN` environment variable or with `token` parameter."""
            )

        # store the API key and prepare the headers
        self._api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": f"pyannote-audio/{__version__}",
            "Content-Type": "application/json",
        }
        # test authentication
        self._authenticated_get("/test")

    def _create_presigned_url(self, media_url: str) -> str:
        """Create a presigned URL to upload audio file to pyannoteAI platform

        Parameters
        ----------
        media_url : str
            Unique identifier used to retrieve the uploaded audio file on the pyannoteAI platform.
            Any combination of letters (a-z, A-Z), digits (0-9), and the characters -./  prefixed
            with media:// is allowed. One would usually use a string akin to a path on filesystem
            (e.g. "media://path/to/audio.wav").

        Returns
        -------
        url : str
            Presigned URL to upload audio file to pyannoteAI platform
        """

        response = self._authenticated_post("/media/input", json={"url": media_url})
        return response.json()["url"]

    def _hash_md5(self, file: Union[str, Path]) -> str:
        """Compute MD5 hash of a file (used for media_url when not provided)"""
        # source: https://stackoverflow.com/questions/3431825/how-to-generate-an-md5-checksum-of-a-file
        hash_md5 = hashlib.md5()
        with open(file, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def upload(
        self,
        audio: AudioFile,
        media_url: Optional[str] = None,
        callback: Optional[Callable] = None,
    ) -> str:
        """Upload audio file to pyannoteAI platform

        Parameters
        ----------
        audio : AudioFile
            Audio file to be uploaded, can be:
            - a "str" or "Path" instance: "audio.wav" or Path("audio.wav")
            - a "IOBase" instance with "read" and "seek" support: open("audio.wav", "rb")
            - a "Mapping" with any of the above as "audio" key: {"audio": ...}
            - a "Mapping" with both "waveform" and "sample_rate" key:
                {"waveform": (channel, time) torch.Tensor, "sample_rate": 44100}
            For last two options, an additional "channel" key can be provided as a zero-indexed
            integer to load a specific channel: {"audio": "stereo.wav", "channel": 0}
        media_url : str, optional
            Unique identifier used to retrieve the uploaded audio file on the pyannoteAI platform.
            Any combination of letters {a-z, A-Z}, digits {0-9}, and {-./} characters prefixed
            with 'media://' is allowed. One would usually use a string akin to a path on filesystem
            (e.g. "media://path/to/audio.wav"). Defaults to media://{md5-hash-of-audio-file}.
        callback : Callable, optional
            When provided, `callback` is called during the uploading process with the following signature:
                callback(total=...,     # number of bytes to upload
                         completed=...) # number of bytes uploaded)

        Returns
        -------
        media_url : str
            Same as the input `media_url` parameter when provided,
            or "media://{md5-hash-of-audio-file}" otherwise.
        """

        # validate the audio file
        validated_file = Audio.validate_file(audio)

        if "waveform" in validated_file:
            # TODO: add support for uploading in-memory audio
            raise ValueError(
                "Uploading in-memory waveform is not supported yet. Please use a file path directly."
            )

            # could most likely be done through a combination
            # of scipy.io.wavfile.write and BytesIO

        elif isinstance(validated_file["audio"], io.IOBase):
            # TODO: add support for uploading file-like object
            raise ValueError(
                "Uploading file-like object is not supported yet. Please use a file path directly."
            )

            # would most likely remove the possibility of
            # estimating the total size of the file to upload
            # unless something like below is acceptable:

            #     # get the current position
            #     current_position = f.tell()
            #     # seek to the end to get the total size
            #     total_size = f.seek(0, os.SEEK_END) - current_position
            #     # seek back to the original position
            #     _ = f.seek(current_position)

        elif isinstance(validated_file["audio"], (str, Path)):
            # get the total size of the file to upload
            # to provide progress information to the hook
            total_size = os.path.getsize(validated_file["audio"])

            if media_url is None:
                media_url = f"media://{self._hash_md5(validated_file['audio'])}"

        # for now, only str and Path audio instances are supported
        with open(validated_file["audio"], "rb") as f:
            # wrap the file object in a UploadingCallbackBytesIO instance
            # to allow calling the hook during the upload process
            data = UploadingCallbackBytesIO(callback, total_size, f.read())

        if not (isinstance(media_url, str) and media_url.startswith("media://")):
            raise ValueError(
                f"""
Invalid media URI: {media_url}. Any combination of letters {{a-z, A-Z}}, digits {{0-9}},
and {{-./}} characters prefixed with 'media://' is allowed."""
            )

        # created the presigned URL to upload the audio file
        presigned_url = self._create_presigned_url(media_url)

        # upload the audio file to the presigned URL
        try:
            response = requests.put(presigned_url, data=data)
        except ConnectionError:
            raise ConnectionError(
                f"""
Failed to upload audio to presigned URL {presigned_url}.
Please check your internet connection or visit https://pyannote.openstatus.dev/ to check the status of the pyannoteAI API."""
            )

        # TODO: handle HTTPError returned by the API
        response.raise_for_status()

        return media_url

    def diarize(
        self,
        media_url: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        confidence: bool = False,
    ) -> str:
        """Initiate a diarization job on the pyannoteAI web API

        Parameters
        ----------
        media_url : str
            media://{...} URL created with the `upload` method or
            any other public URL pointing to an audio file.
        num_speakers : int, optional
            Force number of speakers to diarize. If not provided, the
            number of speakers will be determined automatically.
        min_speakers : int, optional
            Not supported yet. Minimum number of speakers. Has no effect when `num_speakers` is provided.
        max_speakers : int, optional
            Not supported yet. Maximum number of speakers. Has no effect when `num_speakers` is provided.
        confidence : bool, optional
            Defaults to False

        Returns
        -------
        job_id: str

        Raises
        ------
        HTTPError
            If something else went wrong
        """

        assert min_speakers is None, "`min_speakers` is not supported yet"
        assert max_speakers is None, "`max_speakers` is not supported yet"

        json = {"url": media_url, "numSpeakers": num_speakers, "confidence": confidence}

        response = self._authenticated_post("/diarize", json=json)
        data = response.json()
        return data["jobId"]

    def retrieve(self, job_id: str, every_seconds: int = 10) -> dict:
        """Retrieve output of a job (once completed) from pyannoteAI web API

        Parameters
        ----------
        job_id : str
            Job ID.

        Returns
        -------
        job_output : dict
            Job output

        Raises
        ------
        PyannoteAIFailedJob
            If the job failed
        PyannoteAICanceledJob
            If the job was canceled
        HTTPError
            If something else went wrong
        """

        job_status = None

        while True:
            job = self._authenticated_get(f"/jobs/{job_id}").json()
            job_status = job["status"]

            if job_status not in ["succeeded", "canceled", "failed"]:
                time.sleep(every_seconds)
                continue

            break

        if job_status == "failed":
            raise PyannoteAIFailedJob("Job failed", job_id)

        if job_status == "canceled":
            raise PyannoteAICanceledJob("Job canceled", job_id)

        return job

    def deserialize(self, completed_job: dict) -> dict:
        """Deserialize job output into pyannote.core objects"""

        output = completed_job["output"]
        job_id = completed_job["jobId"]

        serialized_output = dict()
        for key, value in output.items():
            if key in ["diarization", "identification"]:
                serialized_output[key] = self._serialize_segmentation(value)

            if key == "confidence":
                serialized_output[key] = self._serialize_confidence(value, job_id)

        return serialized_output

    def _serialize_segmentation(self, output: list[dict]) -> Annotation:
        """Deserialize segmentation output into pyannote.core.Annotation

        Parameters
        ----------
        output : dict
            Segmentation output of a diarization/identification job

        Returns
        -------
        segmentation : Annotation
            Deserialized diarization/identification
        """
        annotation = Annotation()
        for t, turn in enumerate(output):
            segment = Segment(start=turn["start"], end=turn["end"])
            speaker = turn["speaker"]
            annotation[segment, t] = speaker

        return annotation

    def _serialize_confidence(
        self, output: dict, media_url: str
    ) -> SlidingWindowFeature:
        """Deserialize confidence output into pyannote.core.SlidingWindowFeature

        Parameters
        ----------
        output : dict
            Confidence output of a diarization/identification job

        Returns
        -------
        confidence : SlidingWindowFeature
            Deserialized confidence scores
        """
        resolution: float = output["resolution"]
        frames: SlidingWindow = SlidingWindow(
            start=0.0, duration=resolution, step=resolution
        )
        score = np.array(output["score"]).reshape(-1, 1)

        return SlidingWindowFeature(score, frames)

    def __call__(
        self,
        file: AudioFile,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        hook: Optional[Callable] = None,
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
        hook : Callable, optional
            Not supported yet.

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

        assert hook is None, "`hook` is not supported yet"

        # upload file to pyannoteAI cloud API
        media_url: str = self.upload(file)

        # initiate diarization job
        job_id = self.diarize(
            media_url,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            confidence=False,
        )

        # retrieve job output (once completed)
        job_output = self.retrieve(job_id)

        # deserialize the output into a good-old Annotation instance
        return self.deserialize(job_output)["diarization"]
