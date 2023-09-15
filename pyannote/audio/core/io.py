# MIT License
#
# Copyright (c) 2020- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
# Audio IO

pyannote.audio relies on torchaudio for reading and resampling.

"""

import math
import random
import warnings
from io import IOBase
from pathlib import Path
from typing import Mapping, Optional, Text, Tuple, Union

import numpy as np
import torch.nn.functional as F
import torchaudio
from pyannote.core import Segment
from torch import Tensor

torchaudio.set_audio_backend("soundfile")

AudioFile = Union[Text, Path, IOBase, Mapping]

AudioFileDocString = """
Audio files can be provided to the Audio class using different types:
    - a "str" or "Path" instance: "audio.wav" or Path("audio.wav")
    - a "IOBase" instance with "read" and "seek" support: open("audio.wav", "rb")
    - a "Mapping" with any of the above as "audio" key: {"audio": ...}
    - a "Mapping" with both "waveform" and "sample_rate" key:
        {"waveform": (channel, time) numpy.ndarray or torch.Tensor, "sample_rate": 44100}

For last two options, an additional "channel" key can be provided as a zero-indexed
integer to load a specific channel: {"audio": "stereo.wav", "channel": 0}
"""


def get_torchaudio_info(file: AudioFile):
    """Protocol preprocessor used to cache output of torchaudio.info

    This is useful to speed future random access to this file, e.g.
    in dataloaders using Audio.crop a lot....
    """

    info = torchaudio.info(file["audio"])

    # rewind if needed
    if isinstance(file["audio"], IOBase):
        file["audio"].seek(0)

    return info


class Audio:
    """Audio IO

    Parameters
    ----------
    sample_rate: int, optional
        Target sampling rate. Defaults to using native sampling rate.
    mono : {'random', 'downmix'}, optional
        In case of multi-channel audio, convert to single-channel audio
        using one of the following strategies: select one channel at
        'random' or 'downmix' by averaging all channels.

    Usage
    -----
    >>> audio = Audio(sample_rate=16000, mono='downmix')
    >>> waveform, sample_rate = audio({"audio": "/path/to/audio.wav"})
    >>> assert sample_rate == 16000
    >>> sample_rate = 44100
    >>> two_seconds_stereo = torch.rand(2, 2 * sample_rate)
    >>> waveform, sample_rate = audio({"waveform": two_seconds_stereo, "sample_rate": sample_rate})
    >>> assert sample_rate == 16000
    >>> assert waveform.shape[0] == 1
    """

    PRECISION = 0.001

    @staticmethod
    def power_normalize(waveform: Tensor) -> Tensor:
        """Power-normalize waveform

        Parameters
        ----------
        waveform : (..., time) Tensor
            Waveform(s)

        Returns
        -------
        waveform: (..., time) Tensor
            Power-normalized waveform(s)
        """
        rms = waveform.square().mean(dim=-1, keepdim=True).sqrt()
        return waveform / (rms + 1e-8)

    @staticmethod
    def validate_file(file: AudioFile) -> Mapping:
        """Validate file for use with the other Audio methods

        Parameter
        ---------
        file: AudioFile

        Returns
        -------
        validated_file : Mapping
            {"audio": str, "uri": str, ...}
            {"waveform": array or tensor, "sample_rate": int, "uri": str, ...}
            {"audio": file, "uri": "stream"} if `file` is an IOBase instance

        Raises
        ------
        ValueError if file format is not valid or file does not exist.

        """

        if isinstance(file, Mapping):
            pass

        elif isinstance(file, (str, Path)):
            file = {"audio": str(file), "uri": Path(file).stem}

        elif isinstance(file, IOBase):
            return {"audio": file, "uri": "stream"}

        else:
            raise ValueError(AudioFileDocString)

        if "waveform" in file:
            waveform: Union[np.ndarray, Tensor] = file["waveform"]
            if len(waveform.shape) != 2 or waveform.shape[0] > waveform.shape[1]:
                raise ValueError(
                    "'waveform' must be provided as a (channel, time) torch Tensor."
                )

            sample_rate: int = file.get("sample_rate", None)
            if sample_rate is None:
                raise ValueError(
                    "'waveform' must be provided with their 'sample_rate'."
                )

            file.setdefault("uri", "waveform")

        elif "audio" in file:
            if isinstance(file["audio"], IOBase):
                return file

            path = Path(file["audio"])
            if not path.is_file():
                raise ValueError(f"File {path} does not exist")

            file.setdefault("uri", path.stem)

        else:
            raise ValueError(
                "Neither 'waveform' nor 'audio' is available for this file."
            )

        return file

    def __init__(self, sample_rate=None, mono=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.mono = mono

    def downmix_and_resample(self, waveform: Tensor, sample_rate: int) -> Tensor:
        """Downmix and resample

        Parameters
        ----------
        waveform : (channel, time) Tensor
            Waveform.
        sample_rate : int
            Sample rate.

        Returns
        -------
        waveform : (channel, time) Tensor
            Remixed and resampled waveform
        sample_rate : int
            New sample rate
        """

        # downmix to mono
        num_channels = waveform.shape[0]
        if num_channels > 1:
            if self.mono == "random":
                channel = random.randint(0, num_channels - 1)
                waveform = waveform[channel : channel + 1]
            elif self.mono == "downmix":
                waveform = waveform.mean(dim=0, keepdim=True)

        # resample
        if (self.sample_rate is not None) and (self.sample_rate != sample_rate):
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.sample_rate
            )
            sample_rate = self.sample_rate

        return waveform, sample_rate

    def get_duration(self, file: AudioFile) -> float:
        """Get audio file duration in seconds

        Parameters
        ----------
        file : AudioFile
            Audio file.

        Returns
        -------
        duration : float
            Duration in seconds.
        """

        file = self.validate_file(file)

        if "waveform" in file:
            frames = len(file["waveform"].T)
            sample_rate = file["sample_rate"]

        else:
            if "torchaudio.info" in file:
                info = file["torchaudio.info"]
            else:
                info = get_torchaudio_info(file)

            frames = info.num_frames
            sample_rate = info.sample_rate

        return frames / sample_rate

    def get_num_samples(self, duration: float, sample_rate: int = None) -> int:
        """Deterministic number of samples from duration and sample rate"""

        sample_rate = sample_rate or self.sample_rate

        if sample_rate is None:
            raise ValueError(
                "`sample_rate` must be provided to compute number of samples."
            )

        return math.floor(duration * sample_rate)

    def __call__(self, file: AudioFile) -> Tuple[Tensor, int]:
        """Obtain waveform

        Parameters
        ----------
        file : AudioFile

        Returns
        -------
        waveform : (channel, time) torch.Tensor
            Waveform
        sample_rate : int
            Sample rate

        See also
        --------
        AudioFile
        """

        file = self.validate_file(file)

        if "waveform" in file:
            waveform = file["waveform"]
            sample_rate = file["sample_rate"]

        elif "audio" in file:
            waveform, sample_rate = torchaudio.load(file["audio"])

            # rewind if needed
            if isinstance(file["audio"], IOBase):
                file["audio"].seek(0)

        channel = file.get("channel", None)

        if channel is not None:
            waveform = waveform[channel : channel + 1]

        return self.downmix_and_resample(waveform, sample_rate)

    def crop(
        self,
        file: AudioFile,
        segment: Segment,
        duration: Optional[float] = None,
        mode="raise",
    ) -> Tuple[Tensor, int]:
        """Fast version of self(file).crop(segment, **kwargs)

        Parameters
        ----------
        file : AudioFile
            Audio file.
        segment : `pyannote.core.Segment`
            Temporal segment to load.
        duration : float, optional
            Overrides `Segment` 'focus' duration and ensures that the number of
            returned frames is fixed (which might otherwise not be the case
            because of rounding errors).
        mode : {'raise', 'pad'}, optional
            Specifies how out-of-bounds segments will behave.
            * 'raise' -- raise an error (default)
            * 'pad' -- zero pad

        Returns
        -------
        waveform : (channel, time) torch.Tensor
            Waveform
        sample_rate : int
            Sample rate

        """
        file = self.validate_file(file)

        if "waveform" in file:
            waveform = file["waveform"]
            frames = waveform.shape[1]
            sample_rate = file["sample_rate"]

        elif "torchaudio.info" in file:
            info = file["torchaudio.info"]
            frames = info.num_frames
            sample_rate = info.sample_rate

        else:
            info = get_torchaudio_info(file)
            frames = info.num_frames
            sample_rate = info.sample_rate

        channel = file.get("channel", None)

        # infer which samples to load from sample rate and requested chunk
        start_frame = math.floor(segment.start * sample_rate)

        if duration:
            num_frames = math.floor(duration * sample_rate)
            end_frame = start_frame + num_frames

        else:
            end_frame = math.floor(segment.end * sample_rate)
            num_frames = end_frame - start_frame

        if mode == "raise":
            if num_frames > frames:
                raise ValueError(
                    f"requested fixed duration ({duration:6f}s, or {num_frames:d} frames) is longer "
                    f"than file duration ({frames / sample_rate:.6f}s, or {frames:d} frames)."
                )

            if end_frame > frames + math.ceil(self.PRECISION * sample_rate):
                raise ValueError(
                    f"requested chunk [{segment.start:.6f}s, {segment.end:.6f}s] (frames #{start_frame:d} to #{end_frame:d}) "
                    f"lies outside of {file.get('uri', 'in-memory')} file bounds [0., {frames / sample_rate:.6f}s] ({frames:d} frames)."
                )
            else:
                end_frame = min(end_frame, frames)
                start_frame = end_frame - num_frames

            if start_frame < 0:
                raise ValueError(
                    f"requested chunk [{segment.start:.6f}s, {segment.end:.6f}s] (frames #{start_frame:d} to #{end_frame:d}) "
                    f"lies outside of {file.get('uri', 'in-memory')} file bounds [0, {frames / sample_rate:.6f}s] ({frames:d} frames)."
                )

        elif mode == "pad":
            pad_start = -min(0, start_frame)
            pad_end = max(end_frame, frames) - frames
            start_frame = max(0, start_frame)
            end_frame = min(end_frame, frames)
            num_frames = end_frame - start_frame

        if "waveform" in file:
            data = file["waveform"][:, start_frame:end_frame]

        else:
            try:
                data, _ = torchaudio.load(
                    file["audio"], frame_offset=start_frame, num_frames=num_frames
                )
                # rewind if needed
                if isinstance(file["audio"], IOBase):
                    file["audio"].seek(0)
            except RuntimeError:
                if isinstance(file["audio"], IOBase):
                    msg = "torchaudio failed to seek-and-read in file-like object."
                    raise RuntimeError(msg)

                msg = (
                    f"torchaudio failed to seek-and-read in {file['audio']}: "
                    f"loading the whole file instead."
                )

                warnings.warn(msg)
                waveform, sample_rate = self.__call__(file)
                data = waveform[:, start_frame:end_frame]

                # storing waveform and sample_rate for next time
                # as it is very likely that seek-and-read will
                # fail again for this particular file
                file["waveform"] = waveform
                file["sample_rate"] = sample_rate

        if channel is not None:
            data = data[channel : channel + 1, :]

        # pad with zeros
        if mode == "pad":
            data = F.pad(data, (pad_start, pad_end))

        return self.downmix_and_resample(data, sample_rate)
