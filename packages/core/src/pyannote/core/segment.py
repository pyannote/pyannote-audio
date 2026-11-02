# The MIT License (MIT)

# Copyright (c) 2014-2021 CNRS

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
#######
Segment
#######

.. plot:: pyplots/segment.py

:class:`pyannote.core.Segment` instances describe temporal fragments (*e.g.* of an audio file). The segment depicted above can be defined like that:

.. code-block:: ipython

  In [1]: from pyannote.core import Segment

  In [2]: segment = Segment(start=5, end=15)

  In [3]: print(segment)

It is nothing more than 2-tuples augmented with several useful methods and properties:

.. code-block:: ipython

  In [4]: start, end = segment

  In [5]: start

  In [6]: segment.end

  In [7]: segment.duration  # duration (read-only)

  In [8]: segment.middle  # middle (read-only)

  In [9]: segment & Segment(3, 12)  # intersection

  In [10]: segment | Segment(3, 12)  # union

  In [11]: segment.overlaps(3)  # does segment overlap time t=3?


Use `Segment.set_precision(ndigits)` to automatically round start and end timestamps to `ndigits` precision after the decimal point.
To ensure consistency between `Segment` instances, it is recommended to call this method only once, right after importing `pyannote.core.Segment`.

.. code-block:: ipython

  In [12]: Segment(1/1000, 330/1000) == Segment(1/1000, 90/1000+240/1000)
  Out[12]: False

  In [13]: Segment.set_precision(ndigits=4)

  In [14]: Segment(1/1000, 330/1000) == Segment(1/1000, 90/1000+240/1000)
  Out[14]: True

See :class:`pyannote.core.Segment` for the complete reference.
"""

import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .timeline import Timeline

import numpy as np

from .utils.types import Alignment


# setting 'frozen' to True makes it hashable and immutable
@dataclass(frozen=True, order=True)
class Segment:
    """
    Time interval

    Parameters
    ----------
    start : float
        interval start time, in seconds.
    end : float
        interval end time, in seconds.


    Segments can be compared and sorted using the standard operators:

    >>> Segment(0, 1) == Segment(0, 1.)
    True
    >>> Segment(0, 1) != Segment(3, 4)
    True
    >>> Segment(0, 1) < Segment(2, 3)
    True
    >>> Segment(0, 1) < Segment(0, 2)
    True
    >>> Segment(1, 2) < Segment(0, 3)
    False

    Note
    ----
    A segment is smaller than another segment if one of these two conditions is verified:

      - `segment.start < other_segment.start`
      - `segment.start == other_segment.start` and `segment.end < other_segment.end`

    """

    start: float = 0.0
    end: float = 0.0

    @staticmethod
    def set_precision(ndigits: int | None = None):
        """Automatically round start and end timestamps to `ndigits` precision after the decimal point

        To ensure consistency between `Segment` instances, it is recommended to call this method only
        once, right after importing `pyannote.core.Segment`.

        Usage
        -----
        >>> from pyannote.core import Segment
        >>> Segment.set_precision(2)
        >>> Segment(1/3, 2/3)
        <Segment(0.33, 0.67)>
        """
        global AUTO_ROUND_TIME
        global SEGMENT_PRECISION

        if ndigits is None:
            # backward compatibility
            AUTO_ROUND_TIME = False
            # 1 μs (one microsecond)
            SEGMENT_PRECISION = 1e-6
        else:
            AUTO_ROUND_TIME = True
            SEGMENT_PRECISION = 10 ** (-ndigits)

    def __bool__(self):
        """Emptiness

        >>> if segment:
        ...    # segment is not empty.
        ... else:
        ...    # segment is empty.

        Note
        ----
        A segment is considered empty if its end time is smaller than its
        start time, or its duration is smaller than 1μs.
        """
        return bool((self.end - self.start) > SEGMENT_PRECISION)

    def __post_init__(self):
        """Round start and end up to SEGMENT_PRECISION precision (when required)"""
        if AUTO_ROUND_TIME:
            object.__setattr__(
                self,
                "start",
                int(self.start / SEGMENT_PRECISION + 0.5) * SEGMENT_PRECISION,
            )
            object.__setattr__(
                self, "end", int(self.end / SEGMENT_PRECISION + 0.5) * SEGMENT_PRECISION
            )

    @property
    def duration(self) -> float:
        """Segment duration (read-only)"""
        return self.end - self.start if self else 0.0

    @property
    def middle(self) -> float:
        """Segment mid-time (read-only)"""
        return 0.5 * (self.start + self.end)

    def __iter__(self) -> Iterator[float]:
        """Unpack segment boundaries
        >>> segment = Segment(start, end)
        >>> start, end = segment
        """
        yield self.start
        yield self.end

    def copy(self) -> "Segment":
        """Get a copy of the segment

        Returns
        -------
        copy : Segment
            Copy of the segment.
        """
        return Segment(start=self.start, end=self.end)

    # ------------------------------------------------------- #
    # Inclusion (in), intersection (&), union (|) and gap (^) #
    # ------------------------------------------------------- #

    def __contains__(self, other: "Segment"):
        """Inclusion

        >>> segment = Segment(start=0, end=10)
        >>> Segment(start=3, end=10) in segment:
        True
        >>> Segment(start=5, end=15) in segment:
        False
        """
        return (self.start <= other.start) and (self.end >= other.end)

    def __and__(self, other):
        """Intersection

        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(5, 15)
        >>> segment & other_segment
        <Segment(5, 10)>

        Note
        ----
        When the intersection is empty, an empty segment is returned:

        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(15, 20)
        >>> intersection = segment & other_segment
        >>> if not intersection:
        ...    # intersection is empty.
        """
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return Segment(start=start, end=end)

    def intersects(self, other: "Segment") -> bool:
        """Check whether two segments intersect each other

        Parameters
        ----------
        other : Segment
            Other segment

        Returns
        -------
        intersect : bool
            True if segments intersect, False otherwise
        """

        return (
            (self.start < other.start and other.start < self.end - SEGMENT_PRECISION)
            or (self.start > other.start and self.start < other.end - SEGMENT_PRECISION)
            or (self.start == other.start)
        )

    def overlaps(self, t: float) -> bool:
        """Check if segment overlaps a given time

        Parameters
        ----------
        t : float
            Time, in seconds.

        Returns
        -------
        overlap: bool
            True if segment overlaps time t, False otherwise.
        """
        return self.start <= t and self.end >= t

    def __or__(self, other: "Segment") -> "Segment":
        """Union

        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(5, 15)
        >>> segment | other_segment
        <Segment(0, 15)>

        Note
        ----
        When a gap exists between the segment, their union covers the gap as well:

        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(15, 20)
        >>> segment | other_segment
        <Segment(0, 20)
        """

        # if segment is empty, union is the other one
        if not self:
            return other
        # if other one is empty, union is self
        if not other:
            return self

        # otherwise, do what's meant to be...
        start = min(self.start, other.start)
        end = max(self.end, other.end)
        return Segment(start=start, end=end)

    def __xor__(self, other: "Segment") -> "Segment":
        """Gap

        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(15, 20)
        >>> segment ^ other_segment
        <Segment(10, 15)

        Note
        ----
        The gap between a segment and an empty segment is not defined.

        >>> segment = Segment(0, 10)
        >>> empty_segment = Segment(11, 11)
        >>> segment ^ empty_segment
        ValueError: The gap between a segment and an empty segment is not defined.
        """

        # if segment is empty, xor is not defined
        if (not self) or (not other):
            raise ValueError(
                "The gap between a segment and an empty segment is not defined."
            )

        start = min(self.end, other.end)
        end = max(self.start, other.start)
        return Segment(start=start, end=end)

    def _str_helper(self, seconds: float) -> str:
        from datetime import timedelta

        negative = seconds < 0
        seconds = abs(seconds)
        td = timedelta(seconds=seconds)
        seconds = td.seconds + 86400 * td.days
        microseconds = td.microseconds
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        sign = "-" if negative else " "
        return (
            f"{sign}{hours:02d}:{minutes:02d}:{seconds:02d}.{microseconds // 1000:03d}"
        )

    def __str__(self):
        """Human-readable representation

        >>> print(Segment(1337, 1337 + 0.42))
        [ 00:22:17.000 -->  00:22:17.420]

        Note
        ----
        Empty segments are printed as "[]"
        """
        if self:
            return f"[{self._str_helper(self.start)} --> {self._str_helper(self.end)}]"
        return "[]"

    def __repr__(self):
        """Computer-readable representation

        >>> Segment(1337, 1337 + 0.42)
        <Segment(1337, 1337.42)>
        """
        return f"<Segment({self.start:g}, {self.end:g})>"

    def _repr_png_(self):
        """IPython notebook support

        See also
        --------
        :mod:`pyannote.core.notebook`
        """
        from .notebook import MATPLOTLIB_IS_AVAILABLE, MATPLOTLIB_WARNING

        if not MATPLOTLIB_IS_AVAILABLE:
            warnings.warn(MATPLOTLIB_WARNING.format(klass=self.__class__.__name__))
            return None

        from .notebook import repr_segment

        try:
            return repr_segment(self)
        except ImportError:
            warnings.warn(
                f"Couldn't import matplotlib to render the vizualization for object {self}. To enable, install the required dependencies with 'pip install pyannore.core[notebook]'"
            )
            return None


class SlidingWindow:
    """Sliding window

    Parameters
    ----------
    duration : float > 0, optional
        Window duration, in seconds. Default is 30 ms.
    step : float > 0, optional
        Step between two consecutive position, in seconds. Default is 10 ms.
    start : float, optional
        First start position of window, in seconds. Default is 0.
    end : float > `start`, optional
        Default is infinity (ie. window keeps sliding forever)

    Examples
    --------

    >>> sw = SlidingWindow(duration, step, start)
    >>> frame_range = (a, b)
    >>> frame_range == sw.toFrameRange(sw.toSegment(*frame_range))
    ... True

    >>> segment = Segment(A, B)
    >>> new_segment = sw.toSegment(*sw.toFrameRange(segment))
    >>> abs(segment) - abs(segment & new_segment) < .5 * sw.step

    >>> sw = SlidingWindow(end=0.1)
    >>> print(next(sw))
    [ 00:00:00.000 -->  00:00:00.030]
    >>> print(next(sw))
    [ 00:00:00.010 -->  00:00:00.040]
    """

    def __init__(self, duration=0.030, step=0.010, start=0.000, end=None):

        # duration must be a float > 0
        if duration <= 0:
            raise ValueError("'duration' must be a float > 0.")
        self.__duration = duration

        # step must be a float > 0
        if step <= 0:
            raise ValueError("'step' must be a float > 0.")
        self.__step: float = step

        # start must be a float.
        self.__start: float = start

        # if end is not provided, set it to infinity
        if end is None:
            self.__end: float = np.inf
        else:
            # end must be greater than start
            if end <= start:
                raise ValueError("'end' must be greater than 'start'.")
            self.__end: float = end

        # current index of iterator
        self.__i: int = -1

    @property
    def start(self) -> float:
        """Sliding window start time in seconds."""
        return self.__start

    @property
    def end(self) -> float:
        """Sliding window end time in seconds."""
        return self.__end

    @property
    def step(self) -> float:
        """Sliding window step in seconds."""
        return self.__step

    @property
    def duration(self) -> float:
        """Sliding window duration in seconds."""
        return self.__duration

    def closest_frame(self, t: float) -> int:
        """Closest frame to timestamp.

        Parameters
        ----------
        t : float
            Timestamp, in seconds.

        Returns
        -------
        index : int
            Index of frame whose middle is the closest to `timestamp`

        """
        return int(np.rint((t - self.__start - 0.5 * self.__duration) / self.__step))

    def samples(self, from_duration: float, mode: Alignment = "strict") -> int:
        """Number of frames

        Parameters
        ----------
        from_duration : float
            Duration in seconds.
        mode : {'strict', 'loose', 'center'}
            In 'strict' mode, computes the maximum number of consecutive frames
            that can be fitted into a segment with duration `from_duration`.
            In 'loose' mode, computes the maximum number of consecutive frames
            intersecting a segment with duration `from_duration`.
            In 'center' mode, computes the average number of consecutive frames
            where the first one is centered on the start time and the last one
            is centered on the end time of a segment with duration
            `from_duration`.

        """
        if mode == "strict":
            return int(np.floor((from_duration - self.duration) / self.step)) + 1

        elif mode == "loose":
            return int(np.floor((from_duration + self.duration) / self.step))

        elif mode == "center":
            return int(np.rint(from_duration / self.step))

    def crop(
        self,
        focus: Union[Segment, "Timeline"],
        mode: Alignment = "loose",
        fixed: float | None = None,
        return_ranges: bool | None = False,
    ) -> np.ndarray | list[list[int]]:
        """Crop sliding window

        Parameters
        ----------
        focus : `Segment` or `Timeline`
        mode : {'strict', 'loose', 'center'}, optional
            In 'strict' mode, only indices of segments fully included in
            'focus' support are returned. In 'loose' mode, indices of any
            intersecting segments are returned. In 'center' mode, first and
            last positions are chosen to be the positions whose centers are the
            closest to 'focus' start and end times. Defaults to 'loose'.
        fixed : float, optional
            Overrides `Segment` 'focus' duration and ensures that the number of
            returned frames is fixed (which might otherwise not be the case
            because of rounding erros).
        return_ranges : bool, optional
            Return as list of ranges. Defaults to indices numpy array.

        Returns
        -------
        indices : np.array (or list of ranges)
            Array of unique indices of matching segments
        """

        from .timeline import Timeline

        if not isinstance(focus, (Segment, Timeline)):
            msg = '"focus" must be a `Segment` or `Timeline` instance.'
            raise TypeError(msg)

        if isinstance(focus, Timeline):
            if fixed is not None:
                msg = "'fixed' is not supported with `Timeline` 'focus'."
                raise ValueError(msg)

            if return_ranges:
                ranges = []

                for i, s in enumerate(focus.support()):
                    rng = self.crop(s, mode=mode, fixed=fixed, return_ranges=True)

                    # if first or disjoint segment, add it
                    if i == 0 or rng[0][0] > ranges[-1][1]:
                        ranges += rng

                    # if overlapping segment, update last range
                    else:
                        ranges[-1][1] = rng[0][1]

                return ranges

            # concatenate all indices
            indices = np.hstack(
                [
                    self.crop(s, mode=mode, fixed=fixed, return_ranges=False)
                    for s in focus.support()
                ]
            )

            # remove duplicate indices
            return np.unique(indices)

        # 'focus' is a `Segment` instance

        if mode == "loose":
            # find smallest integer i such that
            # self.start + i x self.step + self.duration >= focus.start
            i_ = (focus.start - self.duration - self.start) / self.step
            i = int(np.ceil(i_))

            if fixed is None:
                # find largest integer j such that
                # self.start + j x self.step <= focus.end
                j_ = (focus.end - self.start) / self.step
                j = int(np.floor(j_))
                rng = (i, j + 1)

            else:
                n = self.samples(fixed, mode="loose")
                rng = (i, i + n)

        elif mode == "strict":
            # find smallest integer i such that
            # self.start + i x self.step >= focus.start
            i_ = (focus.start - self.start) / self.step
            i = int(np.ceil(i_))

            if fixed is None:
                # find largest integer j such that
                # self.start + j x self.step + self.duration <= focus.end
                j_ = (focus.end - self.duration - self.start) / self.step
                j = int(np.floor(j_))
                rng = (i, j + 1)

            else:
                n = self.samples(fixed, mode="strict")
                rng = (i, i + n)

        elif mode == "center":
            # find window position whose center is the closest to focus.start
            i = self.closest_frame(focus.start)

            if fixed is None:
                # find window position whose center is the closest to focus.end
                j = self.closest_frame(focus.end)
                rng = (i, j + 1)
            else:
                n = self.samples(fixed, mode="center")
                rng = (i, i + n)

        else:
            msg = "'mode' must be one of {'loose', 'strict', 'center'}."
            raise ValueError(msg)

        if return_ranges:
            return [list(rng)]

        return np.array(range(*rng), dtype=np.int64)

    def segmentToRange(self, segment: Segment) -> tuple[int, int]:
        warnings.warn("Deprecated in favor of `segment_to_range`", DeprecationWarning)
        return self.segment_to_range(segment)

    def segment_to_range(self, segment: Segment) -> tuple[int, int]:
        """Convert segment to 0-indexed frame range

        Parameters
        ----------
        segment : Segment

        Returns
        -------
        i0 : int
            Index of first frame
        n : int
            Number of frames

        Examples
        --------

            >>> window = SlidingWindow()
            >>> print window.segment_to_range(Segment(10, 15))
            i0, n

        """
        # find closest frame to segment start
        i0 = self.closest_frame(segment.start)

        # number of steps to cover segment duration
        n = int(segment.duration / self.step) + 1

        return i0, n

    def rangeToSegment(self, i0: int, n: int) -> Segment:
        warnings.warn(
            "This is deprecated in favor of `range_to_segment`", DeprecationWarning
        )
        return self.range_to_segment(i0, n)

    def range_to_segment(self, i0: int, n: int) -> Segment:
        """Convert 0-indexed frame range to segment

        Each frame represents a unique segment of duration 'step', centered on
        the middle of the frame.

        The very first frame (i0 = 0) is the exception. It is extended to the
        sliding window start time.

        Parameters
        ----------
        i0 : int
            Index of first frame
        n : int
            Number of frames

        Returns
        -------
        segment : Segment

        Examples
        --------

            >>> window = SlidingWindow()
            >>> print window.range_to_segment(3, 2)
            [ --> ]

        """

        # frame start time
        # start = self.start + i0 * self.step
        # frame middle time
        # start += .5 * self.duration
        # subframe start time
        # start -= .5 * self.step
        start = self.__start + (i0 - 0.5) * self.__step + 0.5 * self.__duration
        duration = n * self.__step
        end = start + duration

        # extend segment to the beginning of the timeline
        if i0 == 0:
            start = self.start

        return Segment(start, end)

    def samplesToDuration(self, nSamples: int) -> float:
        warnings.warn(
            "This is deprecated in favor of `samples_to_duration`", DeprecationWarning
        )
        return self.samples_to_duration(nSamples)

    def samples_to_duration(self, n_samples: int) -> float:
        """Returns duration of samples"""
        return self.range_to_segment(0, n_samples).duration

    def durationToSamples(self, duration: float) -> int:
        warnings.warn(
            "This is deprecated in favor of `duration_to_samples`", DeprecationWarning
        )
        return self.duration_to_samples(duration)

    def duration_to_samples(self, duration: float) -> int:
        """Returns samples in duration"""
        return self.segment_to_range(Segment(0, duration))[1]

    def __getitem__(self, i: int) -> Segment:
        """
        Parameters
        ----------
        i : int
            Index of sliding window position

        Returns
        -------
        segment : :class:`Segment`
            Sliding window at ith position

        """

        # window start time at ith position
        start = self.__start + i * self.__step

        # in case segment starts after the end,
        # return an empty segment
        if start >= self.__end:
            return None

        return Segment(start=start, end=start + self.__duration)

    def next(self) -> Segment:
        return self.__next__()

    def __next__(self) -> Segment:
        self.__i += 1
        window = self[self.__i]

        if window:
            return window
        else:
            raise StopIteration()

    def __iter__(self) -> "SlidingWindow":
        """Sliding window iterator

        Use expression 'for segment in sliding_window'

        Examples
        --------

        >>> window = SlidingWindow(end=0.1)
        >>> for segment in window:
        ...     print(segment)
        [ 00:00:00.000 -->  00:00:00.030]
        [ 00:00:00.010 -->  00:00:00.040]
        [ 00:00:00.020 -->  00:00:00.050]
        [ 00:00:00.030 -->  00:00:00.060]
        [ 00:00:00.040 -->  00:00:00.070]
        [ 00:00:00.050 -->  00:00:00.080]
        [ 00:00:00.060 -->  00:00:00.090]
        [ 00:00:00.070 -->  00:00:00.100]
        [ 00:00:00.080 -->  00:00:00.110]
        [ 00:00:00.090 -->  00:00:00.120]
        """

        # reset iterator index
        self.__i = -1
        return self

    def __len__(self) -> int:
        """Number of positions

        Equivalent to len([segment for segment in window])

        Returns
        -------
        length : int
            Number of positions taken by the sliding window
            (from start times to end times)

        """
        if np.isinf(self.__end):
            raise ValueError("infinite sliding window.")

        # start looking for last position
        # based on frame closest to the end
        i = self.closest_frame(self.__end)

        while self[i]:
            i += 1
        length = i

        return length

    def copy(self) -> "SlidingWindow":
        """Duplicate sliding window"""
        duration = self.duration
        step = self.step
        start = self.start
        end = self.end
        sliding_window = self.__class__(
            duration=duration, step=step, start=start, end=end
        )
        return sliding_window

    def __call__(
        self, support: Union[Segment, "Timeline"], align_last: bool = False
    ) -> Iterable[Segment]:
        """Slide window over support

        Parameter
        ---------
        support : Segment or Timeline
            Support on which to slide the window.
        align_last : bool, optional
            Yield a final segment so that it aligns exactly with end of support.

        Yields
        ------
        chunk : Segment

        Example
        -------
        >>> window = SlidingWindow(duration=2., step=1.)
        >>> for chunk in window(Segment(3, 7.5)):
        ...     print(tuple(chunk))
        (3.0, 5.0)
        (4.0, 6.0)
        (5.0, 7.0)
        >>> for chunk in window(Segment(3, 7.5), align_last=True):
        ...     print(tuple(chunk))
        (3.0, 5.0)
        (4.0, 6.0)
        (5.0, 7.0)
        (5.5, 7.5)
        """

        from pyannote.core import Timeline

        if isinstance(support, Timeline):
            segments = support

        elif isinstance(support, Segment):
            segments = Timeline(segments=[support])

        else:
            msg = (
                f'"support" must be either a Segment or a Timeline '
                f"instance (is {type(support)})"
            )
            raise TypeError(msg)

        for segment in segments:
            if segment.duration < self.duration:
                continue

            window = SlidingWindow(
                duration=self.duration,
                step=self.step,
                start=segment.start,
                end=segment.end,
            )

            for s in window:
                # ugly hack to account for floating point imprecision
                if s in segment:
                    yield s
                    last = s

            if align_last and last.end < segment.end:
                yield Segment(start=segment.end - self.duration, end=segment.end)
