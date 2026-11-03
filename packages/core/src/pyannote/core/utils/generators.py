# The MIT License (MIT)

# Copyright (c) 2014-2018 CNRS

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

import itertools
from collections.abc import Iterable, Iterator
from string import ascii_uppercase


def pairwise(iterable: Iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def string_generator(skip: list | set | None = None) -> Iterator[str]:
    """Label generator

    Parameters
    ----------
    skip : list or set
        List of labels that must be skipped.
        This option is useful in case you want to make sure generated labels
        are different from a pre-existing set of labels.

    Usage
    -----
    t = string_generator()
    next(t) -> 'A'    # start with 1-letter labels
    ...               # from A to Z
    next(t) -> 'Z'
    next(t) -> 'AA'   # then 2-letters labels
    next(t) -> 'AB'   # from AA to ZZ
    ...
    next(t) -> 'ZY'
    next(t) -> 'ZZ'
    next(t) -> 'AAA'  # then 3-letters labels
    ...               # (you get the idea)
    """
    if skip is None:
        skip = []

    # label length
    r = 1

    # infinite loop
    while True:
        # generate labels with current length
        for c in itertools.product(ascii_uppercase, repeat=r):
            if c in skip:
                continue
            yield "".join(c)

        # increment label length when all possibilities are exhausted
        r = r + 1


def int_generator() -> Iterator[int]:
    i = 0
    while True:
        yield i
        i = i + 1
