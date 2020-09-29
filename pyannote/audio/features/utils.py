#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2019 CNRS

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

import warnings
import numpy as np

import librosa
from librosa.util import valid_audio
from librosa.util.exceptions import ParameterError

from pyannote.core import SlidingWindow, SlidingWindowFeature

from soundfile import SoundFile
import soundfile as sf


def get_audio_duration(current_file):
    """Return audio file duration

    Parameters
    ----------
    current_file : dict
        Dictionary given by pyannote.database.

    Returns
    -------
    duration : float
        Audio file duration.
    """

    with SoundFile(current_file["audio"], "r") as f:
        duration = float(f.frames) / f.samplerate

    return duration


def get_audio_sample_rate(current_file):
    """Return audio file sampling rate

    Parameters
    ----------
    current_file : dict
        Dictionary given by pyannote.database.

    Returns
    -------
    sample_rate : int
        Sampling rate
    """
    with SoundFile(current_file["audio"], "r") as f:
        sample_rate = f.samplerate

    return sample_rate


def read_audio(current_file, sample_rate=None, mono=True):
    """Read audio file

    Parameters
    ----------
    current_file : dict
        Dictionary given by pyannote.database.
    sample_rate: int, optional
        Target sampling rate. Defaults to using native sampling rate.
    mono : int, optional
        Convert multi-channel to mono. Defaults to True.

    Returns
    -------
    y : (n_samples, n_channels) np.array
        Audio samples.
    sample_rate : int
        Sampling rate.

    Notes
    -----
    In case `current_file` contains a `channel` key, data of this (1-indexed)
    channel will be returned.

    """

    y, file_sample_rate = sf.read(
        current_file["audio"], dtype="float32", always_2d=True
    )

    # extract specific channel if requested
    channel = current_file.get("channel", None)
    if channel is not None:
        y = y[:, channel - 1 : channel]

    # convert to mono
    if mono and y.shape[1] > 1:
        y = np.mean(y, axis=1, keepdims=True)

    # resample if sample rates mismatch
    if (sample_rate is not None) and (file_sample_rate != sample_rate):
        if y.shape[1] == 1:
            # librosa expects mono audio to be of shape (n,), but we have (n, 1).
            y = librosa.core.resample(y[:, 0], file_sample_rate, sample_rate)[:, None]
        else:
            y = librosa.core.resample(y.T, file_sample_rate, sample_rate).T
    else:
        sample_rate = file_sample_rate

    return y, sample_rate


class RawAudio:
    """Raw audio with on-the-fly data augmentation

    Parameters
    ----------
    sample_rate: int, optional
        Target sampling rate. Defaults to using native sampling rate.
    mono : int, optional
        Convert multi-channel to mono. Defaults to True.
    augmentation : `pyannote.audio.augmentation.Augmentation`, optional
        Data augmentation.
    """

    def __init__(self, sample_rate=None, mono=True, augmentation=None):

        super().__init__()
        self.sample_rate = sample_rate
        self.mono = mono

        self.augmentation = augmentation

        if sample_rate is not None:
            self.sliding_window_ = SlidingWindow(
                start=-0.5 / sample_rate,
                duration=1.0 / sample_rate,
                step=1.0 / sample_rate,
            )

    @property
    def dimension(self):
        return 1

    @property
    def sliding_window(self):
        return self.sliding_window_

    def get_features(self, y, sample_rate):

        # convert to mono
        if self.mono:
            y = np.mean(y, axis=1, keepdims=True)

        # resample if sample rates mismatch
        if (self.sample_rate is not None) and (self.sample_rate != sample_rate):
            if y.shape[1] == 1:
                # librosa expects mono audio to be of shape (n,), but we have (n, 1).
                y = librosa.core.resample(y[:, 0], sample_rate, self.sample_rate)[:, None]
            else:
                y = librosa.core.resample(y.T, sample_rate, self.sample_rate).T
            sample_rate = self.sample_rate

        # augment data
        if self.augmentation is not None:
            y = self.augmentation(y, sample_rate)

        # TODO: how time consuming is this thing (needs profiling...)
        try:
            valid = valid_audio(y[:, 0], mono=True)
        except ParameterError as e:
            msg = f"Something went wrong when augmenting waveform."
            raise ValueError(msg)

        return y

    def __call__(self, current_file, return_sr=False):
        """Obtain waveform

        Parameters
        ----------
        current_file : dict
            `pyannote.database` files.
        return_sr : `bool`, optional
            Return sample rate. Defaults to False

        Returns
        -------
        waveform : `pyannote.core.SlidingWindowFeature`
            Waveform
        sample_rate : `int`
            Only when `return_sr` is set to True
        """

        if "waveform" in current_file:

            if self.sample_rate is None:
                msg = (
                    "`RawAudio` needs to be instantiated with an actual "
                    "`sample_rate` if one wants to use precomputed "
                    "waveform."
                )
                raise ValueError(msg)
            sample_rate = self.sample_rate

            y = current_file["waveform"]

            if len(y.shape) != 2:
                msg = (
                    f"Precomputed waveform should be provided as a "
                    f"(n_samples, n_channels) `np.ndarray`."
                )
                raise ValueError(msg)

        else:
            y, sample_rate = sf.read(
                current_file["audio"], dtype="float32", always_2d=True
            )

        # extract specific channel if requested
        channel = current_file.get("channel", None)
        if channel is not None:
            y = y[:, channel - 1 : channel]

        y = self.get_features(y, sample_rate)

        sliding_window = SlidingWindow(
            start=-0.5 / sample_rate, duration=1.0 / sample_rate, step=1.0 / sample_rate
        )

        if return_sr:
            return (
                SlidingWindowFeature(y, sliding_window),
                sample_rate if self.sample_rate is None else self.sample_rate,
            )

        return SlidingWindowFeature(y, sliding_window)

    def get_context_duration(self):
        return 0.0

    def crop(self, current_file, segment, mode="center", fixed=None):
        """Fast version of self(current_file).crop(segment, **kwargs)

        Parameters
        ----------
        current_file : dict
            `pyannote.database` file.
        segment : `pyannote.core.Segment`
            Segment from which to extract features.
        mode : {'loose', 'strict', 'center'}, optional
            In 'strict' mode, only frames fully included in 'segment' are
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
        waveform : (n_samples, n_channels) numpy array
            Waveform

        See also
        --------
        `pyannote.core.SlidingWindowFeature.crop`
        """

        if self.sample_rate is None:
            msg = (
                "`RawAudio` needs to be instantiated with an actual "
                "`sample_rate` if one wants to use `crop`."
            )
            raise ValueError(msg)

        # find the start and end positions of the required segment
        ((start, end),) = self.sliding_window_.crop(
            segment, mode=mode, fixed=fixed, return_ranges=True
        )

        # this is expected number of samples.
        # this will be useful later in case of on-the-fly resampling
        n_samples = end - start

        if "waveform" in current_file:

            y = current_file["waveform"]

            if len(y.shape) != 2:
                msg = (
                    f"Precomputed waveform should be provided as a "
                    f"(n_samples, n_channels) `np.ndarray`."
                )
                raise ValueError(msg)

            sample_rate = self.sample_rate
            data = y[start:end]

        else:
            # read file with SoundFile, which supports various fomats
            # including NIST sphere
            with SoundFile(current_file["audio"], "r") as audio_file:

                sample_rate = audio_file.samplerate

                # if the sample rates are mismatched,
                # recompute the start and end
                if sample_rate != self.sample_rate:

                    sliding_window = SlidingWindow(
                        start=-0.5 / sample_rate,
                        duration=1.0 / sample_rate,
                        step=1.0 / sample_rate,
                    )
                    ((start, end),) = sliding_window.crop(
                        segment, mode=mode, fixed=fixed, return_ranges=True
                    )

                try:
                    audio_file.seek(start)
                    data = audio_file.read(end - start, dtype="float32", always_2d=True)
                except RuntimeError as e:
                    msg = (
                        f"SoundFile failed to seek-and-read in "
                        f"{current_file['audio']}: loading the whole file..."
                    )
                    warnings.warn(msg)
                    return self(current_file).crop(segment, mode=mode, fixed=fixed)

        # extract specific channel if requested
        channel = current_file.get("channel", None)
        if channel is not None:
            data = data[:, channel - 1 : channel]

        return self.get_features(data, sample_rate)


# # THIS SCRIPT CAN BE USED TO CRASH-TEST THE ON-THE-FLY RESAMPLING

# import numpy as np
# from pyannote.audio.features import RawAudio
# from pyannote.core import Segment
# from pyannote.audio.features.utils import get_audio_duration
# from tqdm import tqdm
#
# TEST_FILE = '/Users/bredin/Corpora/etape/BFMTV_BFMStory_2010-09-03_175900.wav'
# current_file = {'audio': TEST_FILE}
# duration = get_audio_duration(current_file)
#
# for sample_rate in [8000, 16000, 44100, 48000]:
#     raw_audio = RawAudio(sample_rate=sample_rate)
#     for i in tqdm(range(1000), desc=f'{sample_rate:d}Hz'):
#         start = np.random.rand() * (duration - 1.)
#         data = raw_audio.crop(current_file, Segment(start, start + 1), fixed=1.)
#         assert len(data) == sample_rate
