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
import numpy as np

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
    augmentation : Augmentation, optional
    duration : float, optional
        Use audio chunks with that duration. Defaults to the fixed duration
        used during training, when available.
    step : float, optional
        Ratio of audio chunk duration used as step between two consecutive
        audio chunks. Defaults to 0.25.
    device : optional
    return_intermediate : optional
    """

    # TODO: add progress bar (at least for demo purposes)

    def __init__(
        self,
        validate_dir: Path = None,
        epoch: int = None,
        augmentation: Optional[Augmentation] = None,
        duration: float = None,
        step: float = None,
        batch_size: int = 32,
        device: Optional[Union[Text, torch.device]] = None,
        return_intermediate=None,
        progress_hook=None,
    ):

        try:
            validate_dir = Path(validate_dir)
        except TypeError as e:
            msg = (
                f'"validate_dir" must be str, bytes or os.PathLike object, '
                f"not {type(validate_dir).__name__}."
            )
            raise TypeError(msg)

        strict = epoch is None
        self.validate_dir = validate_dir.expanduser().resolve(strict=strict)

        train_dir = self.validate_dir.parents[1]
        root_dir = train_dir.parents[1]

        config_yml = root_dir / "config.yml"
        config = load_config(config_yml, training=False)

        # use feature extraction from config.yml configuration file
        self.feature_extraction_ = config["feature_extraction"]

        super().__init__(
            augmentation=augmentation, sample_rate=self.feature_extraction_.sample_rate
        )

        self.feature_extraction_.augmentation = self.augmentation

        specs_yml = train_dir / "specs.yml"
        specifications = load_specs(specs_yml)

        if epoch is None:
            params_yml = self.validate_dir / "params.yml"
            params = load_params(params_yml)
            self.epoch_ = params["epoch"]
            # keep track of pipeline parameters
            self.pipeline_params_ = params.get("params", {})
        else:
            self.epoch_ = epoch

        self.preprocessors_ = config["preprocessors"]

        self.weights_pt_ = train_dir / "weights" / f"{self.epoch_:04d}.pt"

        model = config["get_model_from_specs"](specifications)
        model.load_state_dict(
            torch.load(self.weights_pt_, map_location=lambda storage, loc: storage)
        )

        # defaults to using GPU when available
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # send model to device
        self.model_ = model.eval().to(self.device)

        # initialize chunks duration with that used during training
        self.duration = getattr(config["task"], "duration", None)

        self.min_duration = getattr(config["task"], "min_duration", None)

        # override chunks duration by user-provided value
        if duration is not None:
            # warn that this might be sub-optimal
            if self.duration is not None and duration != self.duration:
                # TODO: do not show this message if min_duration < new_duration < duration
                msg = (
                    f"Model was trained with {self.duration:g}s chunks and "
                    f"is applied on {duration:g}s chunks. This might lead "
                    f"to sub-optimal results."
                )
                warnings.warn(msg)
            # do it anyway
            self.duration = duration

        if step is None:
            step = 0.25
        self.step = step
        self.chunks_ = SlidingWindow(
            duration=self.duration, step=self.step * self.duration
        )

        self.batch_size = batch_size

        self.return_intermediate = return_intermediate
        self.progress_hook = progress_hook

    @property
    def duration(self):
        return self.duration_

    @duration.setter
    def duration(self, duration: float):
        self.duration_ = duration
        self.chunks_ = SlidingWindow(
            duration=self.duration, step=self.step * self.duration
        )

    @property
    def step(self):
        return getattr(self, "step_", 0.25)

    @step.setter
    def step(self, step: float):
        self.step_ = step
        self.chunks_ = SlidingWindow(
            duration=self.duration, step=self.step * self.duration
        )

    @property
    def classes(self):
        return self.model_.classes

    def get_dimension(self) -> int:
        try:
            dimension = self.model_.dimension
        except AttributeError:
            dimension = len(self.model_.classes)
        return dimension

    def get_resolution(self) -> SlidingWindow:

        resolution = self.model_.resolution

        # model returns one vector per input frame
        if resolution == RESOLUTION_FRAME:
            resolution = self.feature_extraction_.sliding_window

        # model returns one vector per input window
        if resolution == RESOLUTION_CHUNK:
            resolution = self.chunks_

        return resolution

    def get_features(self, y, sample_rate) -> np.ndarray:

        features = SlidingWindowFeature(
            self.feature_extraction_.get_features(y, sample_rate),
            self.feature_extraction_.sliding_window,
        )

        return self.model_.slide(
            features,
            self.chunks_,
            batch_size=self.batch_size,
            device=self.device,
            return_intermediate=self.return_intermediate,
            progress_hook=self.progress_hook,
        ).data

    def get_context_duration(self) -> float:
        # FIXME: add half window duration to context?
        return self.feature_extraction_.get_context_duration()
