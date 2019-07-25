#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019 CNRS

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
# Marvin Lavechin - marvinlavechin@gmail.com

from pyannote.audio.train.callback import Callback
import numpy as np
import random


class SpecAugmentCallback(Callback):
    """
    Callback for spectrogram augmentation. Two-step process :

    1) Apply frequency mask(s)
    2) Apply time mask(s)
    (3) Time warping) : Not implemented yet. Shown as leading to a small improvement
    in the reference.

    Parameters
    ----------
    frequency_masking_para : `int`, optional
        Maximal size of the frequency mask, in number of frames
        (random between 0 and frequency_masking_para). Defaults to 27
    time_masking_para : `int`, optional
        Maximal size of the time mask, in number of frames
        (random between 0 and time_masking_para). Defaults to 100.
    nb_frequency_masks : `int`, optional
        Number of frequency masks. Defaults to 1.
    nb_time_masks : `int`, optional
        Number of time masks. Defaults to 1.

    Reference
    ---------
    https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html

    Usage
    -----
    # config.yml
    callbacks:
      - name: pyannote.audio.augmentation.spec_augment_callback.SpecAugmentCallback
        params:
          time_masking_para: 100
          frequency_masking_para: 27
          nb_time_masks: 1
          nb_frequency_masks: 1
    """

    def __init__(self, frequency_masking_para=27, time_masking_para=100,
                 nb_frequency_masks=1, nb_time_masks=1):
        super().__init__()
        self.frequency_masking_para = frequency_masking_para
        self.time_masking_para = time_masking_para
        self.nb_frequency_masks = nb_frequency_masks
        self.nb_time_masks = nb_time_masks

    def on_train_start(self, trainer):
        nb_frames = float(trainer.batch_generator_.duration) / float(trainer.batch_generator_.frame_info.step)

        # We don't want too wide time masks (same as in the google ref.)
        self.time_masking_para = int(min(0.2*nb_frames, self.time_masking_para))

    def on_batch_start(self, trainer, batch):
        # TODO : provide a better implementation, one that would not rely on nested loops
        # (as this might be slow)
        for i, spec in enumerate(batch['X']):

            tau = spec.shape[0]
            v = spec.shape[1]

            augmented_spec = spec.copy()

            # 1) Frequency masking
            for j in range(self.nb_frequency_masks):
                f = np.random.uniform(low=0.0, high=self.frequency_masking_para)
                f = int(f)
                f0 = random.randint(0, v - f)
                augmented_spec[:, f0:f0 + f] = 0

            # 2) Time masking
            for j in range(self.nb_time_masks):
                t = np.random.uniform(low=0.0, high=self.time_masking_para)
                t = int(t)
                t0 = random.randint(0, tau - t)
                augmented_spec[t0:t0 + t, :] = 0

            batch['X'][i] = augmented_spec

        return batch
