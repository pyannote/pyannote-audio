# The MIT License (MIT)
#
# Copyright (c) 2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# AUTHOR
# Hervé Bredin - http://herve.niderb.fr

import warnings
from typing import Optional
from typing import Union
from typing import Text
from pathlib import Path

import torch
import pescador
import numpy as np

from pyannote.core import Segment
from pyannote.core import SlidingWindow
from pyannote.core import SlidingWindowFeature

from pyannote.audio.train.model import RESOLUTION_FRAME
from pyannote.audio.train.model import RESOLUTION_CHUNK

from pyannote.audio.augmentation import Augmentation
from pyannote.audio.features import FeatureExtraction

from pyannote.audio.applications.config import load_config
from pyannote.audio.applications.config import load_specs
from pyannote.audio.applications.config import load_params


class Pretrained(FeatureExtraction):
    """

    Parameters
    ----------
    validate_dir : Path
        Path to a validation directory.
    epoch : int, optional
        If provided, force loading this epoch.
        Defaults to reading epoch in validate_dir/params.yml.
    """

    # TODO: add progress bar (at least for demo purposes)

    def __init__(self, validate_dir: Union[Text, Path] = None,
                       epoch: int = None,
                       augmentation: Optional[Augmentation] = None,
                       duration: float = None,
                       step: float = 0.25,
                       batch_size: int = 32,
                       device: Optional[Union[Text, torch.device]] = None,
                       return_intermediate = None):

        self.validate_dir = validate_dir.expanduser().resolve(strict=True)

        train_dir = self.validate_dir.parents[1]
        root_dir = train_dir.parents[1]

        config_yml = root_dir / 'config.yml'
        config = load_config(config_yml, training=False)

        # use feature extraction from config.yml configuration file
        self.feature_extraction_ = config['feature_extraction']

        super().__init__(augmentation=augmentation,
                         sample_rate=self.feature_extraction_.sample_rate)

        self.feature_extraction_.augmentation = self.augmentation

        specs_yml = train_dir / 'specs.yml'
        specifications = load_specs(specs_yml)

        if epoch is None:
            params_yml = self.validate_dir / 'params.yml'
            params = load_params(params_yml)
            self.epoch_ = params['epoch']
            # keep track of pipeline parameters
            self.pipeline_params_ = params.get('params', {})
        else:
            self.epoch_ = epoch

        self.preprocessors_ = config['preprocessors']

        self.weights_pt_ = train_dir / 'weights' / f'{self.epoch_:04d}.pt'

        model = config['get_model_from_specs'](specifications)
        model.load_state_dict(
            torch.load(self.weights_pt_,
                       map_location=lambda storage, loc: storage))

        # defaults to using GPU when available
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # send model to device
        self.model_ = model.eval().to(self.device)

        # initialize chunks duration with that used during training
        self.duration = getattr(config['task'], 'duration', None)

        # override chunks duration by user-provided value
        if duration is not None:
            # warn that this might be sub-optimal
            if self.duration is not None and duration != self.duration:
                msg = (
                    f'Model was trained with {self.duration:g}s chunks and '
                    f'is applied on {duration:g}s chunks. This might lead '
                    f'to sub-optimal results.'
                )
                warnings.warn(msg)
            # do it anyway
            self.duration = duration

        self.step = step
        self.chunks_ = SlidingWindow(duration=self.duration,
                                     step=self.step * self.duration)

        self.batch_size = batch_size

        self.resolution_ = self.model_.resolution

        # model returns one vector per input frame
        if self.resolution_ == RESOLUTION_FRAME:
            self.resolution_ = self.feature_extraction_.sliding_window

        # model returns one vector per input window
        if self.resolution_ == RESOLUTION_CHUNK:
            self.resolution_ = self.chunks_

        try:
            self.dimension_ = self.model_.dimension
        except AttributeError:
            self.dimension_ = len(self.model_.classes)

        self.return_intermediate = return_intermediate

    @property
    def classes(self):
        return self.model_.classes

    def get_dimension(self) -> int:
        return self.dimension_

    def get_resolution(self) -> SlidingWindow:
        return self.resolution_

    def apply(self, X: np.ndarray) -> np.ndarray:
        tX = torch.tensor(X, dtype=torch.float32, device=self.device)
        # FIXME: fix support for return_intermediate
        fX = self.model_(tX, return_intermediate=self.return_intermediate)
        return fX.detach().to('cpu').numpy()

    def get_features(self, y, sample_rate) -> np.ndarray:

        features = SlidingWindowFeature(
            self.feature_extraction_.get_features(y, sample_rate),
            self.feature_extraction_.sliding_window)

        duration = len(y) / sample_rate
        support = Segment(0, duration)

        # corner case where file is shorter than duration used for training
        if duration < self.duration:
            chunks = [support]
            fixed = duration
        else:
            chunks = list(self.chunks_(support, align_last=True))
            fixed = self.duration


        batches = pescador.maps.buffer_stream(
            iter({'X': features.crop(chunk, mode='center',
                                     fixed=fixed)}
                 for chunk in chunks),
            self.batch_size, partial=True)

        fX = np.vstack([self.apply(batch['X']) for batch in batches])

        if (self.model_.resolution == RESOLUTION_CHUNK) or \
           (self.return_intermediate is not None):
            return fX

        # get total number of frames (based on last window end time)
        n_frames = self.resolution_.samples(chunks[-1].end, mode='center')

        # data[i] is the sum of all predictions for frame #i
        data = np.zeros((n_frames, self.dimension_), dtype=np.float32)

        # k[i] is the number of chunks that overlap with frame #i
        k = np.zeros((n_frames, 1), dtype=np.int8)

        for chunk, fX_ in zip(chunks, fX):

            # indices of frames overlapped by chunk
            indices = self.resolution_.crop(chunk,
                                            mode=self.model_.alignment,
                                            fixed=fixed)

            # accumulate the outputs
            data[indices] += fX_

            # keep track of the number of overlapping sequence
            # TODO - use smarter weights (e.g. Hamming window)
            k[indices] += 1

        # compute average embedding of each frame
        data = data / np.maximum(k, 1)

        return data

    def get_context_duration(self) -> float:
        # FIXME: add half window duration to context?
        return self.feature_extraction_.get_context_duration()
