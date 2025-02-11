import hashlib
import io
import os
import time
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import requests

# from httpx import HTTPError
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature
from requests import Response
from requests.exceptions import ConnectionError

from pyannote.audio import Audio, Pipeline, __version__
from pyannote.audio.core.io import AudioFile


class FailedJob(Exception):
    pass


class CanceledJob(Exception):
    pass


class UploadingHookBytesIO(io.BytesIO):
    """BytesIO subclass that calls a hook during the read process

    Parameters
    ----------
    hook : Callable
    total_size : int
        Total size to upload (in bytes)
    initial_bytes : bytes
        Initial bytes to upload
    """

    def __init__(
        self,
        hook: Callable,
        total_size: int,
        initial_bytes: bytes,
    ):
        self.total_size = total_size
        self._completed_size = 0
        self._hook = hook
        super().__init__(initial_bytes)

    def read(self, size=-1) -> bytes:
        data = super().read(size)
        self._completed_size += len(data)
        self._hook(
            "Uploading to pyannoteAI",
            None,
            total=self.total_size,
            completed=self._completed_size,
        )
        return data


class V1(Pipeline):
    """Speaker diarization using pyannote.ai API

    Parameters
    ----------
    token : str
        pyannoteAI API key created from https://dashboard.pyannote.ai

    Usage
    -----
    # instantiate pipeline from HF hub...
    >>> from pyannote.audio import Pipeline
    >>> pipeline = Pipeline.from_pretrained("pyannoteAI/precision-api", token="{PYANNOTEAI_API_KEY}")

    # ... or from pyannote
    >>> from pyannote.audio.pipelines.api import V1
    >>> pipeline = V1(token="{PYANNOTEAI_API_KEY}")

    # upload your audio file to the pyannoteAI platform (and store it for a few hours for later re-use if needed)
    >>> media_url = pipeline.upload("/path/to/your/audio.wav", "/unique/identifier")

    # `diarize` method exposes more options
    >>> diarization, confidence = pipeline.diarize(media_url, return_confidence=True)
    >>> assert instance(confidence, pyannote.core.SlidigWindowFeature)

    >>> pipeline.diarize(media_url, skip_serialization=True)
    {'jobId': '...',
     'status': 'succeeded',
     'createdAt': '2024-12-12T10:46:26.236Z',
     'updatedAt': '2024-12-12T10:46:42.273Z',
     'output': {
      'diarization': [
        {'speaker': 'SPEAKER_00', 'start': 6.665,'end': 7.165},
        ...,
        {'speaker': 'SPEAKER_01', 'start': 21.745, 'end': 28.545}
       ],
      'confidence': {'score': [96, 97,  97, ...], 'resolution': 0.02},
    }

    # apply diarization on your audio file (upload and process in one go)
    >>> diarization = pipeline("/path/to/your/audio.wav")
    >>> assert isinstance(diarization, pyannote.core.Annotation)
    """

    API_URL = "https://api.pyannote.ai/v1"

    def __init__(self, token: Optional[str] = None):
        super().__init__()
        self.token = token
        self.api_key = token or os.getenv("PYANNOTEAI_API_KEY", "")

    def _authenticated_get(self, route: str) -> Response:
        """Send GET request to pyannoteAI API"""

        try:
            response = requests.get(f"{self.API_URL}{route}", headers=self._headers)
        except ConnectionError:
            raise ConnectionError(
                """
        Failed to connect to pyannoteAI API. Please check your internet connection
        or visit https://pyannote.openstatus.dev/ to check the status of the pyannoteAI API."""
            )

        # TODO: handle HTTPError returned by the API
        response.raise_for_status()

        return response

    def _authenticated_post(self, route: str, json: Optional[dict] = None) -> Response:
        """Send POST request to pyannoteAI API"""

        try:
            response = requests.post(
                f"{self.API_URL}{route}", json=json, headers=self._headers
            )
        except ConnectionError:
            raise ConnectionError(
                """
        Failed to connect to pyannoteAI API. Please check your internet connection
        or visit https://pyannote.openstatus.dev/ to check the status of the pyannoteAI API."""
            )

        # TODO: handle HTTPError returned by the API
        response.raise_for_status()

        return response

    @property
    def api_key(self):
        return self._api_key

    @api_key.setter
    def api_key(self, api_key: str) -> None:
        if not api_key:
            raise ValueError(
                """
        Failed to authenticate to pyannoteAI API. Please create an API key on https://dashboard.pyannote.ai/ and
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
        hook: Optional[Callable] = None,
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
        hook : Callable, optional
            When provided, `hook` is called during the uploading process with the following signature:
                hook("Uploading to pyannoteAI",
                     None,          # always None
                     total=...,     # number of bytes to upload
                     completed=..., # number of bytes uploaded)
                     file=..., )    # can be ignored

        Returns
        -------
        media_url : str
            Same as the input `media_url` parameter when provided,
            or "media://{md5-hash-of-audio-file}" otherwise.
        """

        if not (isinstance(media_url, str) and media_url.startswith("media://")):
            raise ValueError(
                f"""
Invalid media URI: {media_url}. Any combination of letters {{a-z, A-Z}}, digits {{0-9}},
and {{-./}} characters prefixed with 'media://' is allowed."""
            )

        # validate the audio file
        validated_file = Audio.validate_file(audio)

        # let the hook know about the validated file
        # (probably not needed)
        hook = self.setup_hook(validated_file, hook)

        # created the presigned URL to upload the audio file
        presigned_url = self._create_presigned_url(media_url)

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
            # wrap the file object in a UploadingHookBytesIO instance
            # to allow calling the hook during the upload process
            data = UploadingHookBytesIO(hook, total_size, f.read())

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

    def _diarize(
        self,
        media_url: str,
        num_speakers: Optional[int] = None,
        confidence: bool = False,
    ) -> str:
        json = {"url": media_url, "numSpeakers": num_speakers, "confidence": confidence}

        response = self._authenticated_post("/diarize", json=json)
        data = response.json()
        return data["jobId"]

    def _job(self, job_id: str) -> dict:
        response = self._authenticated_get(f"/jobs/{job_id}")
        return response.json()

    def _poll(
        self,
        job_id: str,
        hook: Optional[Callable] = None,
        every_seconds: float = 10.0,
    ) -> dict:
        if hook is not None:
            hook("Processing", None, total=3, completed=0)

        job_status = None

        while job_status not in ["succeeded", "canceled", "failed"]:
            job = self._job(job_id)
            job_status = job["status"]

            if job_status == "created":
                if hook is not None:
                    hook("Processing", None, total=3, completed=1)

            if job_status == "running":
                if hook is not None:
                    hook("Processing", None, total=3, completed=2)

            time.sleep(every_seconds)

        if hook is not None:
            hook("Processing", None, total=3, completed=3)

        if job_status == "failed":
            raise FailedJob("Job failed", job_id)

        if job_status == "canceled":
            raise CanceledJob("Job canceled", job_id)

        return job

    def serialize_output(self, completed_job: dict, media_url: str) -> dict:
        """Serialize output of completed job"""

        output = completed_job["output"]

        serialized_output = dict()
        for key, value in output.items():
            if key in ["diarization", "identification"]:
                serialized_output[key] = self._serialize_diarization(value, media_url)

            if key == "confidence":
                serialized_output[key] = self._serialize_confidence(value, media_url)

        return serialized_output

    def _serialize_diarization(self, output: list[dict], media_url: str) -> Annotation:
        """Serialize speaker diarization/identification result

        Parameters
        ----------
        output : dict
            Output of the diarization/identification job
        media_url : str
            media://{...} URL created with the `upload` method or
            any other public URL pointing to an audio file

        Returns
        -------
        prediction : Annotation
            Speaker diarization/identification
        """
        annotation = Annotation(uri=media_url)
        for t, turn in enumerate(output):
            segment = Segment(start=turn["start"], end=turn["end"])
            speaker = turn["speaker"]
            annotation[segment, t] = speaker

        return annotation

    def _serialize_confidence(
        self, output: dict, media_url: str
    ) -> SlidingWindowFeature:
        """Serialize confidence scores

        Parameters
        ----------
        output : dict
            Output of confidence scores

        Returns
        -------
        confidence : SlidingWindowFeature
            Confidence scores
        """
        resolution: float = output["resolution"]
        frames: SlidingWindow = SlidingWindow(
            start=0.0, duration=resolution, step=resolution
        )
        score = np.array(output["score"]).reshape(-1, 1)

        confidence = SlidingWindowFeature(score, frames)
        confidence.uri = media_url

        return confidence

    def diarize(
        self,
        media_url: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> Union[dict, Annotation, tuple[Annotation, SlidingWindowFeature]]:
        """Perform speaker diarization

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

        Returns
        -------
        output : dict

        Raises
        ------
        FailedJob
            If the job failed
        CanceledJob
            If the job was canceled
        HTTPError
            If something else went wrong
        """

        # create diarization job (might raise HTTPError)
        job_id: str = self._diarize(media_url, num_speakers=num_speakers)

        # poll diarization job (might raise FailedJob, CanceledJob, or HTTPError)
        try:
            completed_job: dict = self._poll(job_id)
        except (FailedJob, CanceledJob):
            raise

        return completed_job

    # make diarize method available as __call__ (which is the default method OSS users are used to)
    def __call__(
        self,
        file: AudioFile,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        hook: Optional[Callable] = None,
    ) -> Annotation:
        """Apply speaker diarization on pyannoteAI cloud API

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
        diarization : Annotation
            Speaker diarization result (when successful)

        Raises
        ------
        FailedJob
            If the job failed
        CanceledJob
            If the job was canceled
        HTTPError
            If something else went wrong
        """

        assert min_speakers is None, "`min_speakers` is not supported yet"
        assert max_speakers is None, "`max_speakers` is not supported yet"
        assert hook is None, "`hook` is not supported yet"

        # upload file to pyannoteAI cloud API
        media_url: str = self.upload(file)

        # request diarization of the uploaded file
        completed_job = self.diarize(media_url, num_speakers=num_speakers)

        serialized_output = self.serialize_output(completed_job, media_url)
        return serialized_output["diarization"]
