#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2018 CNRS

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

    # use precomputed duration when available
    if 'duration' in current_file:
        return current_file['duration']

    # otherwise use SoundFile
    with SoundFile(current_file['audio'], 'r') as f:
        duration = f.duration

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
    with SoundFile(current_file['audio'], 'r') as f:
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

    y, file_sample_rate = sf.read(current_file['audio'], dtype='float32')

    # resample if sample rates mismatch
    if file_sample_rate != sample_rate:
        y = librosa.core.resample(y, file_sample_rate, sample_rate)

    # reshape mono files to (1, n) [was (n, )]
    if y.ndim == 1:
        y = y.reshape(1, -1)

    # extract specific channel if requested
    channel = current_file.get('channel', None)
    if channel is not None:
        y = y[channel - 1, :]

    # convert to mono
    if mono:
        y = librosa.to_mono(y)

    return y.T, sample_rate


class RawAudio(object):
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

    def __init__(self, sample_rate=None, mono=True,
                 augmentation=None):

        super(RawAudio, self).__init__()
        self.sample_rate = sample_rate
        self.mono = mono

        self.augmentation = augmentation

        if sample_rate is not None:
            self.sliding_window_ = SlidingWindow(start=-.5/sample_rate,
                                                 duration=1./sample_rate,
                                                 step=1./sample_rate)

    @property
    def dimension(self):
        return 1

    @property
    def sliding_window(self):
        return self.sliding_window_

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

        if 'waveform' in current_file:
            if self.sample_rate is None:
                msg = ('`RawAudio` needs to be instantiated with an actual '
                       '`sample_rate` if one wants to use precomputed '
                       'waveform.')
                raise ValueError(msg)

            y = current_file['waveform']
            sample_rate = self.sample_rate

        else:
            y, sample_rate = read_audio(current_file,
                                        sample_rate=self.sample_rate,
                                        mono=self.mono)

        if len(y.shape) < 2:
            y = y.reshape(-1, 1)

        if self.augmentation is not None:
            y = self.augmentation(y, sample_rate)

            try:
                valid = valid_audio(y[:, 0], mono=True)
            except ParameterError as e:
                msg = (f"Something went wrong when augmenting waveform.")
                raise ValueError(msg)

        sliding_window = SlidingWindow(
            start=-.5/sample_rate,
            duration=1./sample_rate,
            step=1./sample_rate)

        if return_sr:
            return SlidingWindowFeature(y, sliding_window), sample_rate

        return SlidingWindowFeature(y, sliding_window)

    def get_context_duration(self):
        return 0.

    def crop(self, current_file, segment, mode='center', fixed=None):
        """Fast version of self(current_file).crop(segment, **kwargs)

        Parameters
        ----------
        current_file : dict
            `pyannote.database` file.
        segment : `pyannote.core.Segment`
            Segment from which to extract features.

        Returns
        -------
        waveform : (n_samples, 1) numpy array
            Waveform

        See also
        --------
        `pyannote.core.SlidingWindowFeature.crop`
        """
        if self.sample_rate is None:
            msg = ('`RawAudio` needs to be instantiated with an actual '
                   '`sample_rate` if one wants to use the `crop` method.')
            raise ValueError(msg)

        # find the start and end positions of the required segment
        (start, end), = self.sliding_window_.crop(
            segment, mode=mode, fixed=fixed, return_ranges=True)

        if 'waveform' in current_file:
            y = current_file['waveform']
            sample_rate = self.sample_rate
            data = y[start:end]

        else:  # read file with SoundFile, which supports various fomats, including NIST sphere
            try:
                with SoundFile(current_file['audio'], 'r') as audio_file:
                    sample_rate = audio_file.samplerate
                    # if the sample rates are mismatched, recompute the start and end
                    if sample_rate != self.sample_rate:
                        start = int(1. * sample_rate / self.sample_rate * start)
                        end = int(1. * sample_rate / self.sample_rate * end)
                    audio_file.seek(start)
                    data = audio_file.read(end - start, dtype='float32')

            except OSError as e:
                msg = ('ERROR: problems when reading file {0} with segment {1}. '.format(current_file, segment) + str(e) )
                raise ValueError(msg)

            # if sample rate of the file we just read does not match the expected one,
            # resample the piece of data on the fly
            if sample_rate != self.sample_rate:
                data = librosa.core.resample(data, sample_rate, self.sample_rate)
                sample_rate = self.sample_rate

        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
        msg = f'Audio file was loaded using (unsupported) {data.dtype} data-type.'

        if data.dtype == np.uint8:
            raise NotImplementedError(msg)

        elif data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0

        elif data.dtype == np.int32:
            raise NotImplementedError(msg)

        elif data.dtype == np.float32:
            pass

        else:
            raise NotImplementedError(msg)

        # add `n_channels` dimension
        if len(data.shape) < 2:
            data = data.reshape(-1, 1)

        # convert to mono if needed
        if self.mono and len(data.shape) > 1:
            data = np.mean(data, axis=1, keepdims=True)

        try:
            valid = valid_audio(data[:, 0], mono=True)
        except ParameterError as e:
            msg = (f"Something went wrong when trying to extract waveform of "
                   f"file {current_file['database']}/{current_file['uri']} "
                   f"between {segment.start:.3f}s and {segment.end:.3f}s.")
            raise ValueError(msg)

        if self.augmentation is not None:
            data = self.augmentation(data, sample_rate)

        return data
