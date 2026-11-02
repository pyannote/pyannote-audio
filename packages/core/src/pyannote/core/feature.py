# The MIT License (MIT)

# Copyright (c) 2014-2019 CNRS

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


"""
########
Features
########

See :class:`pyannote.core.SlidingWindowFeature` for the complete reference.
"""

import numbers
import warnings
from collections.abc import Iterator
from typing import Union

import numpy as np

from pyannote.core.utils.types import Alignment

from .segment import Segment, SlidingWindow
from .timeline import Timeline


class SlidingWindowFeature(np.lib.mixins.NDArrayOperatorsMixin):
    """Periodic feature vectors

    Parameters
    ----------
    data : (n_frames, n_features) numpy array
    sliding_window : SlidingWindow
    labels : list, optional
        Textual description of each dimension.
    """

    def __init__(
        self,
        data: np.ndarray,
        sliding_window: SlidingWindow,
        labels: list[str] | None = None,
    ):
        self.sliding_window: SlidingWindow = sliding_window
        self.data = data
        self.labels = labels
        self.__i: int = -1

    def __len__(self):
        """Number of feature vectors"""
        return self.data.shape[0]

    @property
    def extent(self):
        return self.sliding_window.range_to_segment(0, len(self))

    @property
    def dimension(self):
        """Dimension of feature vectors"""
        return self.data.shape[1]

    def getNumber(self):
        warnings.warn("This is deprecated in favor of `__len__`", DeprecationWarning)
        return self.data.shape[0]

    def getDimension(self):
        warnings.warn(
            "This is deprecated in favor of `dimension` property", DeprecationWarning
        )
        return self.dimension

    def getExtent(self):
        warnings.warn(
            "This is deprecated in favor of `extent` property", DeprecationWarning
        )
        return self.extent

    def __getitem__(self, i: int) -> np.ndarray:
        """Get ith feature vector"""
        return self.data[i]

    def __iter__(self):
        self.__i = -1
        return self

    def __next__(self) -> tuple[Segment, np.ndarray]:
        self.__i += 1
        try:
            return self.sliding_window[self.__i], self.data[self.__i]
        except IndexError:
            raise StopIteration()

    def next(self):
        return self.__next__()

    def iterfeatures(
        self, window: bool | None = False
    ) -> Iterator[tuple[np.ndarray, Segment] | np.ndarray]:
        """Feature vector iterator

        Parameters
        ----------
        window : bool, optional
            When True, yield both feature vector and corresponding window.
            Default is to only yield feature vector

        """
        n_samples = self.data.shape[0]
        for i in range(n_samples):
            if window:
                yield self.data[i], self.sliding_window[i]
            else:
                yield self.data[i]

    def crop(
        self,
        focus: Segment | Timeline,
        mode: Alignment = "loose",
        fixed: float | None = None,
        return_data: bool = True,
    ) -> Union[np.ndarray, "SlidingWindowFeature"]:
        """Extract frames

        Parameters
        ----------
        focus : Segment or Timeline
        mode : {'loose', 'strict', 'center'}, optional
            In 'strict' mode, only frames fully included in 'focus' support are
            returned. In 'loose' mode, any intersecting frames are returned. In
            'center' mode, first and last frames are chosen to be the ones
            whose centers are the closest to 'focus' start and end times.
            Defaults to 'loose'.
        fixed : float, optional
            Overrides `Segment` 'focus' duration and ensures that the number of
            returned frames is fixed (which might otherwise not be the case
            because of rounding errors).
        return_data : bool, optional
            Return a numpy array (default). For `Segment` 'focus', setting it
            to False will return a `SlidingWindowFeature` instance.

        Returns
        -------
        data : `numpy.ndarray` or `SlidingWindowFeature`
            Frame features.

        See also
        --------
        SlidingWindow.crop

        """

        if (not return_data) and (not isinstance(focus, Segment)):
            msg = (
                '"focus" must be a "Segment" instance when "return_data"'
                "is set to False."
            )
            raise ValueError(msg)

        if (not return_data) and (fixed is not None):
            msg = '"fixed" cannot be set when "return_data" is set to False.'
            raise ValueError(msg)

        ranges = self.sliding_window.crop(
            focus, mode=mode, fixed=fixed, return_ranges=True
        )

        # total number of samples in features
        n_samples = self.data.shape[0]

        # 1 for vector features (e.g. MFCC in pyannote.audio)
        # 2 for matrix features (e.g. grey-level frames in pyannote.video)
        # 3 for 3rd order tensor (e.g. RBG frames in pyannote.video)
        n_dimensions = len(self.data.shape) - 1

        # clip ranges
        clipped_ranges, repeat_first, repeat_last = [], 0, 0
        for start, end in ranges:
            # count number of requested samples before first sample
            repeat_first += min(end, 0) - min(start, 0)
            # count number of requested samples after last sample
            repeat_last += max(end, n_samples) - max(start, n_samples)
            # if all requested samples are out of bounds, skip
            if end < 0 or start >= n_samples:
                continue
            else:
                # keep track of non-empty clipped ranges
                clipped_ranges += [[max(start, 0), min(end, n_samples)]]

        if clipped_ranges:
            data = np.vstack([self.data[start:end, :] for start, end in clipped_ranges])
        else:
            # if all ranges are out of bounds, just return empty data
            shape = (0,) + self.data.shape[1:]
            data = np.empty(shape)

        # corner case when "fixed" duration cropping is requested:
        # correct number of samples even with out-of-bounds indices
        if fixed is not None:
            data = np.vstack(
                [
                    # repeat first sample as many times as needed
                    np.tile(self.data[0], (repeat_first,) + (1,) * n_dimensions),
                    data,
                    # repeat last sample as many times as needed
                    np.tile(
                        self.data[n_samples - 1], (repeat_last,) + (1,) * n_dimensions
                    ),
                ]
            )

        # return data
        if return_data:
            return data

        # wrap data in a SlidingWindowFeature and return
        sliding_window = SlidingWindow(
            start=self.sliding_window[clipped_ranges[0][0]].start,
            duration=self.sliding_window.duration,
            step=self.sliding_window.step,
        )

        return SlidingWindowFeature(data, sliding_window, labels=self.labels)

    def _repr_png_(self):
        from .notebook import MATPLOTLIB_IS_AVAILABLE, MATPLOTLIB_WARNING

        if not MATPLOTLIB_IS_AVAILABLE:
            warnings.warn(MATPLOTLIB_WARNING.format(klass=self.__class__.__name__))
            return None

        from .notebook import repr_feature

        return repr_feature(self)

    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array__(self) -> np.ndarray:
        return self.data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get("out", ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use SlidingWindowFeature instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle SlidingWindowFeature objects.
            if not isinstance(x, self._HANDLED_TYPES + (SlidingWindowFeature,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(
            x.data if isinstance(x, SlidingWindowFeature) else x for x in inputs
        )
        if out:
            kwargs["out"] = tuple(
                x.data if isinstance(x, SlidingWindowFeature) else x for x in out
            )
        data = getattr(ufunc, method)(*inputs, **kwargs)

        if type(data) is tuple:
            # multiple return values
            return tuple(
                type(self)(x, self.sliding_window, labels=self.labels) for x in data
            )
        elif method == "at":
            # no return value
            return None
        else:
            # one return value
            return type(self)(data, self.sliding_window, labels=self.labels)

    def align(self, to: "SlidingWindowFeature") -> "SlidingWindowFeature":
        """Align features by linear temporal interpolation

        Parameters
        ----------
        to : SlidingWindowFeature
            Features to align with.

        Returns
        -------
        aligned : SlidingWindowFeature
            Aligned features
        """

        old_start = self.sliding_window.start
        old_step = self.sliding_window.step
        old_duration = self.sliding_window.duration
        old_samples = len(self)
        old_t = old_start + 0.5 * old_duration + np.arange(old_samples) * old_step

        new_start = to.sliding_window.start
        new_step = to.sliding_window.step
        new_duration = to.sliding_window.duration
        new_samples = len(to)
        new_t = new_start + 0.5 * new_duration + np.arange(new_samples) * new_step

        new_data = np.hstack(
            [
                np.interp(new_t, old_t, old_data)[:, np.newaxis]
                for old_data in self.data.T
            ]
        )
        return SlidingWindowFeature(new_data, to.sliding_window, labels=self.labels)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
