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


from collections.abc import Iterable, Iterator

import numpy as np

from pyannote.core import Segment


def random_segment(
    segments: Iterable[Segment], weighted: bool = False
) -> Iterator[Segment]:
    """Generate segment with probability proportional to its duration

    Parameters
    ----------
    segments : Segment iterable
        Segments.
    weighted : bool, optional
        When True, yield segments with probability proportional to their
        duration. Defaults to yield segments with uniform probability.

    Yields
    ------
    segment : `Segment`
    """

    p = None
    if weighted:
        total = float(sum(s.duration for s in segments))
        p = [s.duration / total for s in segments]

    n_segments = len(segments)
    while True:
        i = np.random.choice(n_segments, p=p)
        yield segments[i]


def random_subsegment(
    segment: Segment, duration: float, min_duration: float | None = None
) -> Iterator[Segment]:
    """Pick a subsegment at random

    Parameters
    ----------
    segment : Segment
    duration : float
        Duration of random subsegment
    min_duration : float, optional
        When provided, choose segment duration at random between `min_duration`
        and `duration` (instead of fixed `duration`).

    Yields
    ------
    segment : `Segment`

    Usage
    -----
    >>> for subsegment in random_subsegment(segment, duration):
    >>> ... # do something with subsegment
    >>> ... pass

    >>> generator = random_subsegment(segment, duration)
    >>> subsegment = next(generator)
    """
    if min_duration is None:
        if duration > segment.duration:
            msg = (
                f"`duration` (= {duration:g}) should be smaller "
                f"than `segment` duration (= {segment.duration:g})."
            )
            raise ValueError(msg)

        while True:
            # draw start time from [segment.start, segment.end - duration]
            t = segment.start + np.random.random() * (segment.duration - duration)
            yield Segment(t, t + duration)

    else:
        # make sure max duration is smaller than actual segment duration
        max_duration = min(segment.duration, duration)

        while True:
            # draw duration from [min_duration, max_duration] interval
            rnd_duration = min_duration + np.random.random() * (
                max_duration - min_duration
            )

            # draw start from [segment.start, segment.end - rnd_duration] interval
            t = segment.start + np.random.random() * (segment.duration - rnd_duration)
            yield Segment(t, t + rnd_duration)
