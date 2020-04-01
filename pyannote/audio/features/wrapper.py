#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 CNRS

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


from pathlib import Path
from typing import Text
from typing import Union
from typing import Dict

from pyannote.database.protocol.protocol import ProtocolFile
from pyannote.core import Segment
from pyannote.core import SlidingWindowFeature
import numpy as np

Wrappable = Union['Precomputed', 'Pretrained', 'RawAudio', 'FeatureExtraction',
                  Dict, Text, Path]

class Wrapper:
    """FeatureExtraction-compliant wrapper

    Parameters
    ----------
    wrappable : Wrappable
        Wrappable object. See "Usage" section for a detailed description.
    **params : Dict
        Keyword parameters passed to the wrapped object when supported.

    Usage
    -----
    If `wrappable` already complies with the `FeatureExtraction` API , it is
    kept unchanged. This includes instances of any `FeatureExtraction` subclass,
    `RawAudio` instances, `Precomputed` instances, and `Pretrained instances.
    In this case, keyword parameters are not used.

    * If `wrappable` is a `Path` to a directory containing precomputed features
      or scores (e.g. the one created by `pyannote-audio [...] apply [...]`), it
      stands for:

      Precomputed(root_dir=wrappable, **params)

    * If `wrappable` is a `Path` to the validation directory created by calling
      `pyannote-audio [...] validate [...]`, it stands for:

      Pretrained(validate_dir=wrappable, **params)

    * If `wrappable` is a `Path` to a checkpoint created by calling
      `pyannote-audio [...] train [...]`, (i.e. with the following structure:
      '{root_dir}/train/{protocol}.train/weights/{epoch}.pt)', it stands for:

      Pretrained(validate_dir='{root_dir}/train/{protocol}.train/validate/fake',
                 epoch=int({epoch}), **params)

    * If `wrappable` is a `Text` containing the name of an existing `torch.hub`
      model, it stands for:

      torch.hub.load('pyannote/pyannote-audio', wrappable, **params)

    * If `wrappable` is a `Text` starting with '@' such as '@key', it stands for:

      lambda current_file: current_file['key']

    In any other situation, it will raise an error.

    Notes
    -----
    It is also possible to provide a `Dict` `wrappable`, in which case it is
    expected to contain a unique key which is the name of a `torch.hub` model
    (or any supported `Path` described above), whose corresponding value is a
    dictionary of custom parameters. For instance,

      Wrapper({'sad': {'step': 0.1}}) is the same as Wrapper('sad', step=0.1)

    This is especially useful in `pyannote-pipeline` configuration files:

        ~~~~~ content of 'config.yml' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          pipeline:
             name: pyannote.audio.pipeline.SpeechActivityDetection
             params:
                scores:
                   sad:
                      step: 0.1
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    def __init__(self, wrappable: Wrappable, **params):
        super().__init__()

        from pyannote.audio.features import Pretrained
        from pyannote.audio.features import Precomputed
        from pyannote.audio.features import FeatureExtraction
        from pyannote.audio.features import RawAudio

        scorer = None
        msg = ''

        # corner
        if isinstance(wrappable, dict):
            wrappable, custom_params = dict(wrappable).popitem()
            params.update(**custom_params)

        # If `wrappable` already complies with the `FeatureExtraction` API , it
        # is kept unchanged. This includes instances of any `FeatureExtraction`
        # subclass,`RawAudio` instances, `Precomputed` instances, and
        # `Pretrained` instances.
        if isinstance(wrappable, (FeatureExtraction, RawAudio, Pretrained, Precomputed)):
            scorer = wrappable

        elif Path(wrappable).is_dir():
            directory = Path(wrappable)

            # If `wrappable` is a `Path` to a directory containing precomputed
            # features or scores, wrap the corresponding `Precomputed` instance
            try:
                scorer = Precomputed(root_dir=directory)
            except Exception as e:
                scorer = None

            # If `wrappable` is a `Path` to a validation directory,
            # wrap the corresponding `Pretrained` instance
            if scorer is None:
                try:
                    scorer = Pretrained(validate_dir=directory, **params)
                except Exception as e:
                    scorer = None

            if scorer is None:
                msg = (
                    f'"{wrappable}" directory does not seem to be the path '
                    f'to precomputed features nor the path to a model '
                    f'validation step.'
                )

        # If `wrappable` is a `Path` to a pretrined model checkpoint,
        # wrap the corresponding `Pretrained` instance
        elif Path(wrappable).is_file():
            checkpoint = Path(wrappable)

            try:
                validate_dir = checkpoint.parents[1] / 'validate' / 'fake'
                epoch = int(checkpoint.stem)
                scorer = Pretrained(validate_dir=validate_dir, epoch=epoch,
                                    **params)
            except Exception as e:
                msg = (
                    f'"{wrappable}" directory does not seem to be the path '
                    f'to a pretrained model checkpoint.'
                )
                scorer = None

        elif isinstance(wrappable, Text):

            # If `wrappable` is a `Text` starting with '@' such as '@key',
            # it means that one should read the "key" key of protocol files
            if wrappable.startswith('@'):
                key = wrappable[1:]
                scorer = lambda current_file: current_file[key]

            # If `wrappable` is a `Text` containing the name of an existing
            # `torch.hub` model, wrap the corresponding `Pretrained`.
            else:
                try:
                    import torch
                    scorer = torch.hub.load('pyannote/pyannote-audio',
                                            wrappable, **params)
                    if not isinstance(scorer, Pretrained):
                        msg = (
                            f'"{wrappable}" exists on torch.hub but does not '
                            f'return a `Pretrained` model instance.'
                        )
                        scorer = None

                except Exception as e:
                    msg = (
                        f'Could not load {wrappable} model from torch.hub. '
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
        from pyannote.audio.features import RawAudio
        from pyannote.audio.features import FeatureExtraction


        if isinstance(self.scorer_, (FeatureExtraction, RawAudio, Pretrained, Precomputed)):
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
