#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2020 CNRS

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

import os
import errno
from pathlib import Path
from typing import Text
from typing import Union

from pyannote.database.protocol.protocol import ProtocolFile
from pyannote.core import Segment
from pyannote.core import SlidingWindowFeature
import numpy as np


def mkdir_p(path):
    """Create directory and all its parents if they do not exist

    This is the equivalent of Unix 'mkdir -p path'

    Parameter
    ---------
    path : str
        Path to new directory.

    Reference
    ---------
    http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    """

    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise exc


class Pre___ed:
    """Wrapper around Precomputed, Pretrained, or torch.hub model

    This class allows user-facing APIs to support (torch.hub or locally)
    pretrained models or precomputed output interchangeably.

    * Pre___ed('sad_ami') is equivalent to
      torch.hub.load('pyannote/pyannote-audio', 'sad_ami')

    * Pre___ed('/path/to/xp/train/.../validate/...') is equivalent to
      Pretrained('/path/to/xp/train/.../validate/...')

    * Pre___ed('/path/to/xp/train/.../validate/.../apply/...') is equivalent to
      Precomputed('/path/to/xp/train/.../validate/.../apply/...')

    Bonus: Pre___ed('@scores') is equivalent to lambda f: f['scores']

    Parameter
    ---------
    placeholder : Text or Path
    """

    def __init__(self, placeholder: Union[Text, Path]):
        super().__init__()

        # if the path to a directory is provided
        if Path(placeholder).is_dir():
            directory = Path(placeholder)

            # if this succeeds, it means that 'placeholder' was indeed a path
            # to the output of "pyannote-audio ... apply"
            try:
                from pyannote.audio.features import Precomputed
                scorer = Precomputed(root_dir=directory)
            except Exception as e:
                scorer = None

            if scorer is None:
                # if this succeeds, it means that 'placeholder' was indeed a
                # path to the output of "pyannote-audio ... validate"
                try:
                    from pyannote.audio.features import Pretrained
                    scorer = Pretrained(validate_dir=directory)
                except Exception as e:
                    scorer = None

            if scorer is None:
                msg = (
                    f'"{placeholder}" directory does not seem to be the path '
                    f'to precomputed features nor the path to a model '
                    f'validation step.'
                )

        # otherwise it should be a string
        elif isinstance(placeholder, Text):

            # @key means that one should read the "key" key of protocol files
            if placeholder.startswith('@'):
                key = placeholder[1:]
                scorer = lambda current_file: current_file[key]

            # if string does not start with "@", it means that 'placeholder'
            # is the name of a torch.hub model
            else:
                try:
                    import torch
                    scorer = torch.hub.load('pyannote/pyannote-audio:develop',
                                            placeholder)
                except Exception as e:
                    msg = (
                        f'Could not load {placeholder} model from torch.hub. '
                        f'The following exception was raised:\n{e}')
                    scorer = None

        # warn the user the something went wrong
        if scorer is None:
            raise ValueError(msg)

        self.scorer_ = scorer

    def crop(self, current_file: ProtocolFile,
                   segment: Segment,
                   mode: Text = 'center',
                   fixed: float = None) -> np.ndarray:
        """Extract frames from a specific region

        Parameters
        ----------
        current_file : ProtocolFile
            Protocol file
        segment : Segment
            Region of the file to process.
        mode : {'loose', 'strict', 'center'}, optional
            In 'strict' mode, only frames fully included in 'segment' support are
            returned. In 'loose' mode, any intersecting frames are returned. In
            'center' mode, first and last frames are chosen to be the ones
            whose centers are the closest to 'segment' start and end times.
            Defaults to 'center'.
        fixed : float, optional
            Overrides 'segment' duration and ensures that the number of
            returned frames is fixed (which might otherwise not be the case
            because of rounding errors).

        Returns
        -------
        frames : np.ndarray
            Frames.
        """

        from pyannote.audio.features import Precomputed
        from pyannote.audio.features import Pretrained

        if isinstance(self.scorer_, (Precomputed, Pretrained)):
            return self.scorer_.crop(current_file,
                                     segment,
                                     mode=mode,
                                     fixed=fixed)

        return self.scorer_(current_file).crop(segment,
                                               mode=mode,
                                               fixed=fixed,
                                               return_data=True)

    def __call__(self, current_file) -> SlidingWindowFeature:
        """Extract frames from the whole file

        Parameters
        ----------
        current_file : ProtocolFile
            Protocol file

        Returns
        -------
        frames : np.ndarray
            Frames.
        """
        return self.scorer_(current_file)

    # used to "inherit" most scorer_ attributes
    def __getattr__(self, name):
        return getattr(self.scorer_, name)
