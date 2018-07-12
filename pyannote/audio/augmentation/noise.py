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
from .base import Augmentation
from glob import glob


class AddNoise(Augmentation):
    """Add noise

    Parameters
    ----------
    noise_dir : str
        Path containing noise .wav files.
    snr_min, snr_max : int, optional
        Defines Signal-to-Noise Ratio (SNR) range in dB. Defaults to [5, 20].
    """

    def __init__(self, noise_dir, snr_min=5, snr_max=20):
        super().__init__()
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.noise_dir = noise_dir

        # load noise database
        self.filenames_, self.durations_ = [], []
        for filename in glob(f'{noise_dir}/**/*.wav', recursive=True):
            self.filenames_.append(filename)
            duration = get_audio_duration({'audio': filename})
            self.durations_.append(duration)
        self.durations_ = np.array(self.durations_)
        #self.probabilities_ = self.durations_ / np.sum(self.durations_)

    def normalize(self, waveform):
        return waveform / np.sqrt(np.mean(waveform ** 2))

    def __call__(self, original, sample_rate):

        raw_audio = RawAudio(sample_rate=sample_rate, mono=True)

        original_duration = len(original) / sample_rate

        # accumulate enough noise to cover duration of original waveform
        noises = []
        left = original_duration
        while left > 0:

            # select noise file at random
            # chosen = np.random.choice(len(self.filenames_),
            #                           p=self.probabilities_)
            chosen = np.random.choice(len(self.filenames_))
            filename = self.filenames_[chosen]
            duration = self.durations_[chosen]

            # if noise file is longer than what is needed, crop it
            if duration > left:
                segment = next(random_subsegment(Segment(0, duration), left))
                noise = raw_audio.crop({'audio': filename}, segment)
                left -= left

            # otherwise, take the whole file
            else:
                noise = raw_audio({'audio': filename}).data
                left -= duration

            noise = self.normalize(noise)
            noises.append(noise)

        # concatenate
        # FIXME: use fade-in between concatenated noises
        noise = np.vstack(noises)

        # select SNR at random
        snr = (self.snr_max - self.snr_min) * np.random.random_sample() + self.snr_min
        alpha = np.exp(-np.log(10) * snr / 20)

        # add noise
        return self.normalize(original) + alpha * noise
