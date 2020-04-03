#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019-2020 CNRS

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


from typing import Union
from typing import List
from typing import Optional

NoiseCollection = Union[str, List[str]]

import random
import numpy as np

from pyannote.core import Segment
from pyannote.core.utils.random import random_subsegment

from pyannote.audio.features import RawAudio
from pyannote.audio.features.utils import get_audio_duration

from pyannote.database import FileFinder
from pyannote.database import get_protocol


normalize = lambda wav: wav / (np.sqrt(np.mean(wav ** 2)) + 1e-8)


class Noise:
    """Noise generator

    Parameters
    ----------
    collection : str or list of str
        `pyannote.database` collection(s) used for adding noise.
        Defaults to "MUSAN.Collection.BackgroundNoise"
    """

    def __init__(self, collection: Optional[NoiseCollection] = None):

        if collection is None:
            collection = "MUSAN.Collection.BackgroundNoise"

        if not isinstance(collection, (list, tuple)):
            collection = [collection]

        self.collection = collection

        self.files_ = []
        preprocessors = {"audio": FileFinder(), "duration": get_audio_duration}
        for collection in self.collection:
            protocol = get_protocol(collection, preprocessors=preprocessors)
            self.files_.extend(protocol.files())

    def __call__(self, n_samples: int, sample_rate: int) -> np.ndarray:
        """Generate noise

        Parameters
        ----------
        n_samples : int
        sample_rate : int

        Returns
        -------
        noise : (n_samples, 1) np.ndarray
        """

        target_duration = n_samples / sample_rate

        raw_audio = RawAudio(sample_rate=sample_rate, mono=True)

        # accumulate enough noise to cover duration
        noises = []
        left = target_duration
        while left > 0:

            # select noise file at random
            file = random.choice(self.files_)
            duration = file["duration"]

            # if noise file is longer than what is needed, crop it
            if duration > left:
                segment = next(random_subsegment(Segment(0, duration), left))
                noise = raw_audio.crop(file, segment, mode="center", fixed=left)
                left = 0

            # otherwise, take the whole file
            else:
                noise = raw_audio(file).data
                left -= duration

            noise = normalize(noise)
            noises.append(noise)

        # concatenate
        noise = np.vstack(noises)
        # TODO: use fade-in between concatenated noises

        return noise
