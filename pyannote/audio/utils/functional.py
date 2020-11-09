from functools import wraps
from typing import Callable, List


def apply_to_array(func: Callable):
    """
    Returns the same function but it will iterate through an Iterable
    and apply the function to each item, returning an array

    Parameters
    ----------
    func : Callable
        The

    Usage
    -----
    >>> @apply_to_array
    ... def array_print(s:str):
    ...     print(s)
    ...     return s
    >>> array_print("Hello World")
    >>> array_print(["Hello", "World"])
    """

    @wraps(func)
    def _inner(arg, *args, **kwargs):
        if isinstance(arg, list):
            return [func(a, *args, **kwargs) for a in arg]
        return func(arg, *args, **kwargs)

    return _inner


def chain(funcs: List[Callable], x: any):
    """Chains methods of a function to be used on the input
    in the order they appear in the list

    Usage
    -----
    >>> f = lambda x : x * 2
    >>> g = lambda x : x + 5
    >>> chain([g,f], 2) # f(g(x))
    14
    """

    current = x
    for f in funcs:
        current = f(current)
    return current
