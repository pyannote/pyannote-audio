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
# Hervé BREDIN - http://herve.niderb.fr

from typing import Optional
from pathlib import Path
from tqdm import tqdm

import torch
from .base import Application
from pyannote.database import get_protocol
from pyannote.database import get_annotated
from pyannote.audio.features import Precomputed
from pyannote.audio.features import RawAudio


class BaseLabeling(Application):

    @property
    def config_default_module(self):
        return 'pyannote.audio.labeling.tasks'

    def validate_init(self, protocol_name,
                            subset='development'):
        """Initialize validation data

        Parameters
        ----------
        protocol_name : `str`
        subset : {'train', 'development', 'test'}
            Defaults to 'development'.

        Returns
        -------
        validation_data : object
            Validation data.

        """

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)
        files = getattr(protocol, subset)()

        # convert lazy ProtocolFile to regular dict for multiprocessing
        files = [dict(file) for file in files]

        if isinstance(self.feature_extraction_, (Precomputed, RawAudio)):
            return files

        validation_data = []
        for current_file in tqdm(files, desc='Feature extraction'):
            current_file['features'] = self.feature_extraction_(current_file)
            validation_data.append(current_file)

        return validation_data
