# MIT License
#
# Copyright (c) 2023- CNRS
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


from typing import Any, Callable, Tuple, Union

from pyannote.audio.core.model import Specifications


def map_with_specifications(
    specifications: Union[Specifications, Tuple[Specifications]],
    func: Callable,
    *iterables,
) -> Union[Any, Tuple[Any]]:
    """Compute the function using arguments from each of the iterables

    Returns a tuple if provided `specifications` is a tuple,
    otherwise returns the function return value.

    Parameters
    ----------
    specifications : (tuple of) Specifications
        Specifications or tuple of specifications
    func : callable
        Function called for each specification with
        `func(*iterables[i], specifications=specifications[i])`
    *iterables :
        List of iterables with same length as `specifications`.

    Returns
    -------
    output : (tuple of) `func` return value(s)
    """

    if isinstance(specifications, Specifications):
        return func(*iterables, specifications=specifications)

    return tuple(
        func(*i, specifications=s) for s, *i in zip(specifications, *iterables)
    )
