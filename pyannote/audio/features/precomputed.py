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


import yaml
import io
from pathlib import Path
from glob import glob
import numpy as np
from numpy.lib.format import open_memmap

from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.database.util import get_unique_identifier
from pyannote.audio.utils.path import mkdir_p


class PyannoteFeatureExtractionError(Exception):
    pass


class Precomputed:
    """Precomputed features

    Parameters
    ----------
    root_dir : `str`
        Path to directory where precomputed features are stored.
    use_memmap : `bool`, optional
        Defaults to False.
    sliding_window : `SlidingWindow`, optional
        Sliding window used for feature extraction. This is not used when
        `root_dir` already exists and contains `metadata.yml`.
    dimension : `int`, optional
        Dimension of feature vectors. This is not used when `root_dir` already
        exists and contains `metadata.yml`.
    classes : iterable, optional
        Human-readable name for each dimension.

    Notes
    -----
    If `root_dir` directory does not exist, one must provide both
    `sliding_window` and `dimension` parameters in order to create and
    populate file `root_dir/metadata.yml` when instantiating.

    """

    def get_path(self, item):
        uri = get_unique_identifier(item)
        path = "{root_dir}/{uri}.npy".format(root_dir=self.root_dir, uri=uri)
        return path

    def __init__(
        self,
        root_dir=None,
        use_memmap=False,
        sliding_window=None,
        dimension=None,
        classes=None,
        augmentation=None,
    ):

        if augmentation is not None:
            msg = "Data augmentation is not supported by `Precomputed`."
            raise ValueError(msg)

        super(Precomputed, self).__init__()
        self.root_dir = Path(root_dir).expanduser().resolve(strict=False)
        self.use_memmap = use_memmap

        path = self.root_dir / "metadata.yml"
        if path.exists():

            with io.open(path, "r") as f:
                params = yaml.load(f, Loader=yaml.SafeLoader)

            self.dimension_ = params.pop("dimension")
            self.classes_ = params.pop("classes", None)
            self.sliding_window_ = SlidingWindow(**params)

            if dimension is not None and self.dimension_ != dimension:
                msg = 'inconsistent "dimension" (is: {0}, should be: {1})'
                raise ValueError(msg.format(dimension, self.dimensions_))

            if classes is not None and self.classes_ != classes:
                msg = 'inconsistent "classes" (is {0}, should be: {1})'
                raise ValueError(msg.format(classes, self.classes_))

            if (sliding_window is not None) and (
                (sliding_window.start != self.sliding_window_.start)
                or (sliding_window.duration != self.sliding_window_.duration)
                or (sliding_window.step != self.sliding_window_.step)
            ):
                msg = 'inconsistent "sliding_window"'
                raise ValueError(msg)

        else:

            if dimension is None:
                if classes is None:
                    msg = (
                        f"Please provide either `dimension` or `classes` "
                        f"parameters (or both) when instantiating "
                        f"`Precomputed`."
                    )
                dimension = len(classes)

            if sliding_window is None or dimension is None:
                msg = (
                    f"Either directory {self.root_dir} does not exist or it "
                    f"does not contain precomputed features. In case it exists "
                    f"and this was done on purpose, please provide both "
                    f"`sliding_window` and `dimension` parameters when "
                    f"instantianting `Precomputed`."
                )
                raise ValueError(msg)

            # create parent directory
            mkdir_p(path.parent)

            params = {
                "start": sliding_window.start,
                "duration": sliding_window.duration,
                "step": sliding_window.step,
                "dimension": dimension,
            }
            if classes is not None:
                params["classes"] = classes

            with io.open(path, "w") as f:
                yaml.dump(params, f, default_flow_style=False)

            self.sliding_window_ = sliding_window
            self.dimension_ = dimension
            self.classes_ = classes

    def augmentation():
        doc = "Data augmentation."

        def fget(self):
            return None

        def fset(self, augmentation):
            if augmentation is not None:
                msg = "Data augmentation is not supported by `Precomputed`."
                raise AttributeError(msg)

        return locals()

    augmentation = property(**augmentation())

    @property
    def sliding_window(self):
        """Sliding window used for feature extraction"""
        return self.sliding_window_

    @property
    def dimension(self):
        """Dimension of feature vectors"""
        return self.dimension_

    @property
    def classes(self):
        """Human-readable label of each dimension"""
        return self.classes_

    def __call__(self, current_file):
        """Obtain features for file

        Parameters
        ----------
        current_file : dict
            `pyannote.database` files.

        Returns
        -------
        features : `pyannote.core.SlidingWindowFeature`
            Features
        """

        path = Path(self.get_path(current_file))

        if not path.exists():
            uri = current_file["uri"]
            database = current_file["database"]
            msg = (
                f"Directory {self.root_dir} does not contain "
                f'precomputed features for file "{uri}" of '
                f'"{database}" database.'
            )
            raise PyannoteFeatureExtractionError(msg)

        if self.use_memmap:
            data = np.load(str(path), mmap_mode="r")
        else:
            data = np.load(str(path))

        return SlidingWindowFeature(data, self.sliding_window_)

    def crop(self, current_file, segment, mode="center", fixed=None):
        """Fast version of self(current_file).crop(segment, **kwargs)

        Parameters
        ----------
        current_file : dict
            `pyannote.database` file.
        segment : `pyannote.core.Segment`
            Segment from which to extract features.

        Returns
        -------
        features : (n_frames, dimension) numpy array
            Extracted features

        See also
        --------
        `pyannote.core.SlidingWindowFeature.crop`
        """

        # match default FeatureExtraction.crop behavior
        if mode == "center" and fixed is None:
            fixed = segment.duration

        memmap = open_memmap(self.get_path(current_file), mode="r")
        swf = SlidingWindowFeature(memmap, self.sliding_window_)
        result = swf.crop(segment, mode=mode, fixed=fixed)
        del memmap
        return result

    def shape(self, item):
        """Faster version of precomputed(item).data.shape"""
        memmap = open_memmap(self.get_path(item), mode="r")
        shape = memmap.shape
        del memmap
        return shape

    def dump(self, item, features):
        path = Path(self.get_path(item))
        mkdir_p(path.parent)
        np.save(path, features.data)
