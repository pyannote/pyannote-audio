#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2020 CNRS

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

"""
# Noise-based data augmentation
"""


import random
import numpy as np
from pyannote.core import Segment
from pyannote.audio.features.utils import RawAudio
from pyannote.audio.features.utils import get_audio_duration
from pyannote.core.utils.random import random_subsegment
from pyannote.core.utils.random import random_segment
from pyannote.database import get_protocol
from pyannote.database import get_annotated
from pyannote.database import FileFinder
from .base import Augmentation


normalize = lambda wav: wav / (np.sqrt(np.mean(wav ** 2)) + 1e-8)


class AddNoise(Augmentation):
    """Additive noise data augmentation

    Parameters
    ----------
    collection : `str` or `list` of `str`
        `pyannote.database` collection(s) used for adding noise. Defaults to
        'MUSAN.Collection.BackgroundNoise' available in `pyannote.db.musan`
        package.
    snr_min, snr_max : int, optional
        Defines Signal-to-Noise Ratio (SNR) range in dB. Defaults to [5, 20].
    """

    def __init__(self, collection=None, snr_min=5, snr_max=20):
        super().__init__()

        if collection is None:
            collection = 'MUSAN.Collection.BackgroundNoise'
        if not isinstance(collection, (list, tuple)):
            collection = [collection]
        self.collection = collection

        self.snr_min = snr_min
        self.snr_max = snr_max

        # load noise database
        self.files_ = []
        preprocessors = {'audio': FileFinder(),
                         'duration': get_audio_duration}
        for collection in self.collection:
            protocol = get_protocol(collection, preprocessors=preprocessors)
            self.files_.extend(protocol.files())

    def __call__(self, original, sample_rate):
        """Augment original waveform

        Parameters
        ----------
        original : `np.ndarray`
            (n_samples, n_channels) waveform.
        sample_rate : `int`
            Sample rate.

        Returns
        -------
        augmented : `np.ndarray`
            (n_samples, n_channels) noise-augmented waveform.
        """

        raw_audio = RawAudio(sample_rate=sample_rate, mono=True)

        original_duration = len(original) / sample_rate

        # accumulate enough noise to cover duration of original waveform
        noises = []
        left = original_duration
        while left > 0:

            # select noise file at random
            file = random.choice(self.files_)
            duration = file['duration']

            # if noise file is longer than what is needed, crop it
            if duration > left:
                segment = next(random_subsegment(Segment(0, duration), left))
                noise = raw_audio.crop(file, segment,
                                       mode='center', fixed=left)
                left = 0

            # otherwise, take the whole file
            else:
                noise = raw_audio(file).data
                left -= duration

            noise = normalize(noise)
            noises.append(noise)

        # concatenate
        # FIXME: use fade-in between concatenated noises
        noise = np.vstack(noises)

        # select SNR at random
        snr = (self.snr_max - self.snr_min) * np.random.random_sample() + self.snr_min
        alpha = np.exp(-np.log(10) * snr / 20)

        return normalize(original) + alpha * noise


class AddNoiseFromGaps(Augmentation):
    """Additive noise data augmentation.

    While AddNoise assumes that files contain only noise, this class uses
    non-speech regions (= gaps) as noise. This is expected to generate more
    realistic noises.

    Parameters
    ----------
    protocol : `str` or `pyannote.database.Protocol`
        Protocol name (e.g. AMI.SpeakerDiarization.MixHeadset)
    subset : {'train', 'development', 'test'}, optional
        Use this subset. Defaults to 'train'.
    snr_min, snr_max : int, optional
        Defines Signal-to-Noise Ratio (SNR) range in dB. Defaults to [5, 20].

    See also
    --------
    `AddNoise`
    """

    def __init__(self, protocol=None, subset='train',
                 snr_min=5, snr_max=20):
        super().__init__()

        self.protocol = protocol
        self.subset = subset
        self.snr_min = snr_min
        self.snr_max = snr_max

        # returns gaps in annotation as pyannote.core.Timeline instance
        get_gaps = lambda f: f['annotation'].get_timeline().gaps(
            support=get_annotated(f))

        if isinstance(protocol, str):
            preprocessors = {
                'audio': FileFinder(),
                'duration': get_audio_duration,
                'gaps': get_gaps}
            protocol = get_protocol(self.protocol,
                                    preprocessors=preprocessors)
        else:
            protocol.preprocessors['gaps'] = get_gaps

        self.files_ = list(getattr(protocol, self.subset)())

    def __call__(self, original, sample_rate):
        """Augment original waveform

        Parameters
        ----------
        original : `np.ndarray`
            (n_samples, n_channels) waveform.
        sample_rate : `int`
            Sample rate.

        Returns
        -------
        augmented : `np.ndarray`
            (n_samples, n_channels) noise-augmented waveform.
        """

        raw_audio = RawAudio(sample_rate=sample_rate, mono=True)

        # accumulate enough noise to cover duration of original waveform
        noises = []
        len_left = len(original)
        while len_left > 0:

            # select noise file at random
            file = random.choice(self.files_)

            # select noise segment at random
            segment = next(random_segment(file['gaps'], weighted=False))
            duration = segment.duration
            segment_len = duration * sample_rate

            # if noise segment is longer than what is needed, crop it at random
            if segment_len > len_left:
                duration = len_left / sample_rate
                segment = next(random_subsegment(segment, duration))

            noise = raw_audio.crop(file, segment,
                                   mode='center', fixed=duration)

            # decrease the `len_left` value by the size of the returned noise
            len_left -= len(noise)

            noise = normalize(noise)
            noises.append(noise)

        # concatenate
        # FIXME: use fade-in between concatenated noises
        noise = np.vstack(noises)

        # select SNR at random
        snr = (self.snr_max - self.snr_min) * np.random.random_sample() + self.snr_min
        alpha = np.exp(-np.log(10) * snr / 20)

        return normalize(original) + alpha * noise
