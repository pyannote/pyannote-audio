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


import yaml
import io
from pathlib import Path
from glob import glob
import numpy as np
from numpy.lib.format import open_memmap
from struct import unpack

from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.database.util import get_unique_identifier
from pyannote.audio.util import mkdir_p


class PyannoteFeatureExtractionError(Exception):
    pass


class Precomputed(object):
    """Precomputed features

    Parameters
    ----------
    root_dir : `str`
        Path to directory where precomputed features are stored.
    use_memmap : `bool`, optional
        Defaults to True.
    sliding_window : `SlidingWindow`, optional
        Sliding window used for feature extraction. This is not used when
        `root_dir` already exists and contains `metadata.yml`.
    dimension : `int`, optional
        Dimension of feature vectors. This is not used when `root_dir` already
        exists and contains `metadata.yml`.
    labels : iterable, optional
        Human-readable name for each dimension.

    Notes
    -----
    If `root_dir` directory does not exist, one must provide both
    `sliding_window` and `dimension` parameters in order to create and
    populate file `root_dir/metadata.yml` when instantiating.

    """

    def get_path(self, item):
        uri = get_unique_identifier(item)
        path = '{root_dir}/{uri}.npy'.format(root_dir=self.root_dir, uri=uri)
        return path

    def __init__(self, root_dir=None, use_memmap=True,
                 sliding_window=None, dimension=None, labels=None,
                 augmentation=None):

        if augmentation is not None:
            msg = 'Data augmentation is not supported by `Precomputed`.'
            raise ValueError(msg)

        super(Precomputed, self).__init__()
        self.root_dir = Path(root_dir).expanduser().resolve(strict=False)
        self.use_memmap = use_memmap

        path = self.root_dir / 'metadata.yml'
        if path.exists():

            with io.open(path, 'r') as f:
                params = yaml.load(f)

            self.dimension_ = params.pop('dimension')
            self.labels_ = params.pop('labels', None)
            self.sliding_window_ = SlidingWindow(**params)

            if dimension is not None and self.dimension_ != dimension:
                msg = 'inconsistent "dimension" (is: {0}, should be: {1})'
                raise ValueError(msg.format(dimension, self.dimensions_))

            if labels is not None and self.labels_ != labels:
                msg = 'inconsistent "labels" (is {0}, should be: {1})'
                raise ValueError(msg.format(labels, self.labels_))

            if ((sliding_window is not None) and
                ((sliding_window.start != self.sliding_window_.start) or
                 (sliding_window.duration != self.sliding_window_.duration) or
                 (sliding_window.step != self.sliding_window_.step))):
                msg = 'inconsistent "sliding_window"'
                raise ValueError(msg)

        else:

            if dimension is None:
                if labels is None:
                    msg = (
                        f'Please provide either `dimension` or `labels` '
                        f'parameters (or both) when instantiating '
                        f'`Precomputed`.'
                    )
                dimension = len(labels)

            if sliding_window is None or dimension is None:
                msg = (
                    f'Either directory {self.root_dir} does not exist or it '
                    f'does not contain precomputed features. In case it exists '
                    f'and this was done on purpose, please provide both '
                    f'`sliding_window` and `dimension` parameters when '
                    f'instantianting `Precomputed`.')
                raise ValueError(msg)

            # create parent directory
            mkdir_p(path.parent)


            params = {'start': sliding_window.start,
                      'duration': sliding_window.duration,
                      'step': sliding_window.step,
                      'dimension': dimension}
            if labels is not None:
                params['labels'] = labels

            with io.open(path, 'w') as f:
                yaml.dump(params, f, default_flow_style=False)

            self.sliding_window_ = sliding_window
            self.dimension_ = dimension
            self.labels_ = labels

    def augmentation():
        doc = "Data augmentation."
        def fget(self):
            return None
        def fset(self, augmentation):
            if augmentation is not None:
                msg = 'Data augmentation is not supported by `Precomputed`.'
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
    def labels(self):
        """Human-readable label of each dimension"""
        return self.labels_

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
            uri = get_unique_identifier(current_file)
            msg = f'No precomputed features for "{uri}".'
            raise PyannoteFeatureExtractionError(msg)

        if self.use_memmap:
            data = np.load(str(path), mmap_mode='r')
        else:
            data = np.load(str(path))

        return SlidingWindowFeature(data, self.sliding_window_)

    def crop(self, current_file, segment, mode='center', fixed=None):
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
        if mode == 'center' and fixed is None:
            fixed = segment.duration

        memmap = open_memmap(self.get_path(current_file), mode='r')
        swf = SlidingWindowFeature(memmap, self.sliding_window_)
        result = swf.crop(segment, mode=mode, fixed=fixed)
        del memmap
        return result

    def shape(self, item):
        """Faster version of precomputed(item).data.shape"""
        memmap = open_memmap(self.get_path(item), mode='r')
        shape = memmap.shape
        del memmap
        return shape

    def dump(self, item, features):
        path = Path(self.get_path(item))
        mkdir_p(path.parent)
        np.save(path, features.data)


class PrecomputedHTK(object):

    def __init__(self, root_dir=None, duration=0.025, step=None):
        super(PrecomputedHTK, self).__init__()
        self.root_dir = root_dir
        self.duration = duration

        # load any htk file in root_dir/database
        path = '{root_dir}/*/*.htk'.format(root_dir=root_dir)
        found = glob(path)

        # FIXME switch to Py3.5 and use glob 'recursive' parameter
        # http://stackoverflow.com/questions/2186525/
        # use-a-glob-to-find-files-recursively-in-python

        if len(found) > 0:
            file_htk = found[0]
        else:
            msg = "Could not find any HTK file in '{root_dir}'."
            raise ValueError(msg.format(root_dir=root_dir))

        X, sample_period = self.load_htk(file_htk)
        self.dimension_ = X.shape[1]
        self.step = sample_period * 1e-7

        # don't trust HTK header when 'step' is provided by the user.
        # HACK remove this when Pepe's HTK files are fixed...
        if step is not None:
            self.step = step

        self.sliding_window_ = SlidingWindow(start=0.,
                                             duration=self.duration,
                                             step=self.step)
    @property
    def sliding_window(self):
        return self.sliding_window_

    @property
    def dimension(self):
        return self.dimension_

    @staticmethod
    def get_path(root_dir, item):
        uri = get_unique_identifier(item)
        path = '{root_dir}/{uri}.htk'.format(root_dir=root_dir, uri=uri)
        return path

    # http://codereview.stackexchange.com/questions/
    # 1496/reading-a-binary-file-containing-periodic-samples
    @staticmethod
    def load_htk(file_htk):
        with open(file_htk, 'rb') as fp:
            data = fp.read(12)
            num_samples, sample_period, sample_size, _ = unpack('>iihh', data)
            num_features = int(sample_size / 4)
            num_samples = int(num_samples)
            X = np.empty((num_samples, num_features))
            for i in range(num_samples):
                data = fp.read(sample_size)
                X[i, :] = unpack('>' + ('f' * (sample_size // 4)), data)
        return X, sample_period

    def __call__(self, item):
        file_htk = self.get_path(self.root_dir, item)
        X, _ = self.load_htk(file_htk)
        return SlidingWindowFeature(X, self.sliding_window_)
