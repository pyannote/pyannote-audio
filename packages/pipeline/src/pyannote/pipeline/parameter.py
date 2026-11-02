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
# Hadrien TITEUX - https://github.com/hadware


from collections.abc import Iterable, Mapping
from typing import Any

from optuna.trial import Trial

from .pipeline import Pipeline


class Parameter:
    """Base hyper-parameter"""


class Categorical(Parameter):
    """Categorical hyper-parameter

    The value is sampled from `choices`.

    Parameters
    ----------
    choices : iterable
        Candidates of hyper-parameter value.
    """

    def __init__(self, choices: Iterable):
        super().__init__()
        self.choices = list(choices)

    def __call__(self, name: str, trial: Trial):
        return trial.suggest_categorical(name, self.choices)


class DiscreteUniform(Parameter):
    """Discrete uniform hyper-parameter

    The value is sampled from the range [low, high],
    and the step of discretization is `q`.

    Parameters
    ----------
    low : `float`
        Lower endpoint of the range of suggested values.
        `low` is included in the range.
    high : `float`
        Upper endpoint of the range of suggested values.
        `high` is included in the range.
    q : `float`
        A step of discretization.
    """

    def __init__(self, low: float, high: float, q: float):
        super().__init__()
        self.low = float(low)
        self.high = float(high)
        self.q = float(q)

    def __call__(self, name: str, trial: Trial):
        return trial.suggest_discrete_uniform(name, self.low, self.high, self.q)


class Integer(Parameter):
    """Integer hyper-parameter

    The value is sampled from the integers in [low, high].

    Parameters
    ----------
    low : `int`
        Lower endpoint of the range of suggested values.
        `low` is included in the range.
    high : `int`
        Upper endpoint of the range of suggested values.
        `high` is included in the range.
    """

    def __init__(self, low: int, high: int):
        super().__init__()
        self.low = int(low)
        self.high = int(high)

    def __call__(self, name: str, trial: Trial):
        return trial.suggest_int(name, self.low, self.high)


class LogUniform(Parameter):
    """Log-uniform hyper-parameter

    The value is sampled from the range [low, high) in the log domain.

    Parameters
    ----------
    low : `float`
        Lower endpoint of the range of suggested values.
        `low` is included in the range.
    high : `float`
        Upper endpoint of the range of suggested values.
        `high` is excluded from the range.
    """

    def __init__(self, low: float, high: float):
        super().__init__()
        self.low = float(low)
        self.high = float(high)

    def __call__(self, name: str, trial: Trial):
        return trial.suggest_loguniform(name, self.low, self.high)


class Uniform(Parameter):
    """Uniform hyper-parameter

    The value is sampled from the range [low, high) in the linear domain.

    Parameters
    ----------
    low : `float`
        Lower endpoint of the range of suggested values.
        `low` is included in the range.
    high : `float`
        Upper endpoint of the range of suggested values.
        `high` is excluded from the range.
    """

    def __init__(self, low: float, high: float):
        super().__init__()
        self.low = float(low)
        self.high = float(high)

    def __call__(self, name: str, trial: Trial):
        return trial.suggest_uniform(name, self.low, self.high)


class Frozen(Parameter):
    """Frozen hyper-parameter

    The value is fixed a priori

    Parameters
    ----------
    value :
        Fixed value.
    """

    def __init__(self, value: Any):
        super().__init__()
        self.value = value

    def __call__(self, name: str, trial: Trial):
        return self.value


class ParamDict(Pipeline, Mapping):
    """Dict-like structured hyper-parameter

    Usage
    -----
    >>> params = ParamDict(param1=Uniform(0.0, 1.0), param2=Uniform(-1.0, 1.0))
    >>> params = ParamDict(**{"param1": Uniform(0.0, 1.0), "param2": Uniform(-1.0, 1.0)})
    """

    def __init__(self, **params):
        super().__init__()
        self.__params = params
        for param_name, param_value in params.items():
            setattr(self, param_name, param_value)

    def __len__(self):
        return len(self.__params)

    def __iter__(self):
        return iter(self.__params)

    def __getitem__(self, param_name):
        return getattr(self, param_name)
