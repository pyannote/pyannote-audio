# MIT License
#
# Copyright (c) 2020 CNRS
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

pyannote.audio relies on SoundFile for reading and librosa for resampling.

The reasons behind this technical choice are summarized in the following benchmarking notebooks:
- Loading: https://gist.github.com/fcbec863c79e8b63242779f77f87a995
- Resampling: https://gist.github.com/mogwai/a5df03e89ab33bc0a5648965280d5445

Those benchmarks (and implementation choices) are meant to be updated when
better options become available: we welcome PRs!
"""

import math
import warnings
from pathlib import Path
from typing import Optional, Text, Tuple, Union

import librosa
import torch
import torchaudio
from torch import Tensor

from pyannote.core import Segment
from pyannote.core.utils.types import Alignment
from pyannote.database import ProtocolFile

AudioFile = Union[Path, Text, ProtocolFile, dict]

"""
Audio files can be provided to the Audio class using different types:
    - a "str" instance: "/path/to/audio.wav"
    - a "Path" instance: Path("/path/to/audio.wav")
    - a ProtocolFile (or regular dict) with an "audio" key:
        {"audio": Path("/path/to/audio.wav")}
    - a ProtocolFile (or regular dict) with both "waveform" and "sample_rate" key:
        {"waveform": (time, channel) numpy array, "sample_rate": 44100}

For last two options, an additional "channel" key can be provided as a zero-indexed
integer to load a specific channel:
        {"audio": Path("/path/to/stereo.wav"), "channel": 0}
"""

# TODO: Remove this when it is the default
torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
torchaudio.set_audio_backend("soundfile")


class Audio:
    """Audio IO

    Parameters
    ----------
    sample_rate: int, optional
        Target sampling rate. Defaults to using native sampling rate.
    mono : int, optional
        Convert multi-channel to mono. Defaults to True.

    Usage
    -----
    >>> audio = Audio(sample_rate=16000, mono=True)
    >>> waveform, sample_rate = audio({"audio": "/path/to/audio.wav"})
    >>> assert sample_rate == 16000
    >>> sample_rate = 44100
    >>> two_seconds_stereo = torch.rand(2, 2 * sample_rate)
    >>> waveform, sample_rate = audio({"waveform": two_seconds_stereo, "sample_rate": sample_rate})
    >>> assert sample_rate == 16000
    >>> assert waveform.shape[0] == 1
    """

    @staticmethod
    def power_normalize(waveform: Tensor) -> Tensor:
        """Power-normalize waveform

        Parameters
        ----------
        waveform : (channel, time) Tensor
            Single or multichannel waveform


        Returns
        -------
        waveform: (channel, time) Tensor
            Power-normalized waveform
        """
        rms = waveform.square().mean(dim=1).sqrt()
        return (waveform.t() / (rms + 1e-8)).t()

    @staticmethod
    def get_duration(file: AudioFile) -> float:
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

        if isinstance(file, (ProtocolFile, dict)):
            audio = file["audio"]
        else:
            audio = file

        if isinstance(audio, Path):
            audio = str(audio)

        info = torchaudio.info(audio)
        return info.num_frames / info.sample_rate

    @staticmethod
    def is_valid(file: AudioFile) -> bool:

        if isinstance(file, (ProtocolFile, dict)):

            if "waveform" in file:

                waveform = file["waveform"]
                if len(waveform.shape) != 2 or waveform.shape[0] > waveform.shape[1]:
                    raise ValueError(
                        "'waveform' must be provided as a (channel, time) torch Tensor."
                    )

                sample_rate = file.get("sample_rate", None)
                if sample_rate is None:
                    raise ValueError(
                        "'waveform' must be provided with their 'sample_rate'."
                    )
                return True

            elif "audio" in file:
                return True

            else:
                # TODO improve error message
                raise ValueError("either 'audio' or 'waveform' key must be provided.")

        return True

    def __init__(self, sample_rate=None, mono=True):
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
        if self.mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample
        if (self.sample_rate is not None) and (self.sample_rate != sample_rate):
            waveform = waveform.numpy()
            if self.mono:
                # librosa expects mono audio to be of shape (n,), but we have (1, n).
                waveform = librosa.core.resample(
                    waveform[0], sample_rate, self.sample_rate
                )[None]
            else:
                waveform = librosa.core.resample(
                    waveform.T, sample_rate, self.sample_rate
                ).T
            sample_rate = self.sample_rate
            waveform = torch.tensor(waveform)
        return waveform, sample_rate

    def __call__(self, file: AudioFile) -> Tuple[Tensor, int]:
        """Obtain waveform

        Parameters
        ----------
        file : AudioFile

        Returns
        -------
        waveform : (time, channel) numpy array
            Waveform
        sample_rate : int
            Sample rate

        See also
        --------
        AudioFile
        """

        self.is_valid(file)
        audio = file
        waveform = None
        sample_rate = None
        channel = None

        if isinstance(file, (ProtocolFile, dict)):

            if "waveform" in file:
                audio = None
                waveform = file["waveform"]
                sample_rate = file.get("sample_rate", None)

            elif "audio" in file:
                audio = file["audio"]

            else:
                pass

            channel = file.get("channel", None)

        if isinstance(audio, Path):
            audio = str(audio)

        if waveform is None:
            waveform, sample_rate = torchaudio.load(audio)

        if channel is not None:
            waveform = waveform[channel - 1 : channel]

        return self.downmix_and_resample(waveform, sample_rate)

    def crop(
        self,
        file: AudioFile,
        segment: Segment,
        mode: Alignment = "center",
        fixed: Optional[float] = None,
    ) -> Tuple[Tensor, int]:
        """Fast version of self(file).crop(segment, **kwargs)

        Parameters
        ----------
        file : AudioFile
            Audio file.
        segment : `pyannote.core.Segment`
            Temporal segment to load.
        mode : {'loose', 'strict', 'center'}, optional
            In 'strict' mode, only samples fully included in 'segment' are
            returned. In 'loose' mode, any intersecting frames are returned. In
            'center' mode, first and last frames are chosen to be the ones
            whose centers are the closest to 'focus' start and end times.
            Defaults to 'center'.
        fixed : float, optional
            Overrides `Segment` 'focus' duration and ensures that the number of
            returned frames is fixed (which might otherwise not be the case
            because of rounding errors). Has no effect in 'strict' or 'loose'
            modes.

        Returns
        -------
        waveform : (time, channel) numpy array
            Waveform
        sample_rate : int
            Sample rate

        TODO: remove support for "mode" option. It is always "center" anyway.
        """

        self.is_valid(file)
        audio = file
        waveform = None
        channel = None

        if isinstance(file, (ProtocolFile, dict)):
            if "waveform" in file:
                audio = None
                waveform = file["waveform"]
                sample_rate = file.get("sample_rate", None)
                frames = waveform.shape[1]

            elif "audio" in file:
                audio = file["audio"]

            channel = file.get("channel", None)

        if isinstance(audio, Path):
            audio = str(audio)

        # read sample rate and number of frames
        if waveform is None:
            info = torchaudio.info(audio)
            sample_rate = info.sample_rate
            frames = info.num_frames

        # infer which samples to load from sample rate and requested chunk
        start_frame = int(segment.start * sample_rate)

        if fixed:
            num_frames = math.floor(fixed * sample_rate)
        else:
            num_frames = math.floor(segment.end * sample_rate - start_frame)

        end_frame = start_frame + num_frames

        if start_frame < 0 or end_frame > frames:
            raise ValueError(
                f"requested chunk [{segment.start:.6f}, {segment.end:.6f}] "
                f"lies outside of file bounds [0., {frames / sample_rate:.6f}]."
            )

        if waveform is not None:
            data = waveform[:, start_frame:end_frame]
        else:
            try:
                data, _ = torchaudio.load(
                    audio, frame_offset=start_frame, num_frames=num_frames
                )
            except RuntimeError:
                msg = (
                    f"torchaudio failed to seek-and-read in "
                    f"{audio}: loading the whole file..."
                )
                warnings.warn(msg)
                return self(audio).crop(segment, mode=mode, fixed=fixed)

        if channel is not None:
            data = data[channel - 1 : channel, :]

        return self.downmix_and_resample(data, sample_rate)
