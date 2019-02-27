#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018 CNRS

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
from pyannote.core import Segment
from pyannote.audio.features.utils import RawAudio
from pyannote.audio.features.utils import get_audio_duration
from pyannote.generators.fragment import random_subsegment
from pyannote.database import get_protocol
from pyannote.database import FileFinder
from .base import Augmentation


class AddNoise(Augmentation):
    """Add noise

    Parameters
    ----------
    collection : str or list of str
        `pyannote.database` collection(s) used for adding noise. Defaults to
        'MUSAN.Collection.BackgroundNoise' available in `pyannote.db.musan`
        package.
    db_yml : str, optional
        Path to `pyannote.database` configuration file.
    snr_min, snr_max : int, optional
        Defines Signal-to-Noise Ratio (SNR) range in dB. Defaults to [5, 20].
    """

    def __init__(self, collection=None, db_yml=None, snr_min=5, snr_max=20):
        super().__init__()

        if collection is None:
            collection = 'MUSAN.Collection.BackgroundNoise'
        if not isinstance(collection, (list, tuple)):
            collection = [collection]
        self.collection = collection
        self.db_yml = db_yml

        self.snr_min = snr_min
        self.snr_max = snr_max

        # load noise database
        self.files_ = []
        preprocessors = {'audio': FileFinder(config_yml=db_yml),
                         'duration': get_audio_duration}
        for collection in self.collection:
            protocol = get_protocol(collection, preprocessors=preprocessors)
            self.files_.extend(protocol.files())

    def normalize(self, waveform):
        return waveform / (np.sqrt(np.mean(waveform ** 2)) + 1e-8)

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
            file = np.random.choice(self.files_)
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

            noise = self.normalize(noise)
            noises.append(noise)

        # concatenate
        # FIXME: use fade-in between concatenated noises
        noise = np.vstack(noises)

        # select SNR at random
        snr = (self.snr_max - self.snr_min) * np.random.random_sample() + self.snr_min
        alpha = np.exp(-np.log(10) * snr / 20)
        if len(noise) > len(original):
            noise = noise[:len(original)]
        return self.normalize(original) + alpha * noise
