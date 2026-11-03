# The MIT License (MIT)

# Copyright (c) 2018-2022 CNRS

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
# Hadrien TITEUX

from typing import Any

import numpy as np
import pytest
from optuna.pruners import MedianPruner
from optuna.samplers import GridSampler, TPESampler

from pyannote.pipeline import Optimizer, Pipeline
from pyannote.pipeline.parameter import Integer, ParamDict
from pyannote.pipeline.typing import Direction


def optimizer_tester(pipeline: Pipeline, target: Any):
    dataset = np.ones(10)
    sampler = TPESampler(seed=4577)
    optimizer = Optimizer(pipeline, sampler=sampler)
    optimizer.tune(dataset, n_iterations=100, show_progress=False)
    assert optimizer.best_params == target


@pytest.mark.parametrize(
    "direction",
    ["minimize", ("minimize", "maximize")],
    ids=["single-objective", "multi-objective"],
)
def test_default_sampler(direction):
    class TestPipeline(Pipeline):
        def get_direction(self):
            return direction

    optimizer = Optimizer(TestPipeline())

    assert isinstance(optimizer.sampler, TPESampler)
    assert optimizer.study_.sampler is optimizer.sampler


@pytest.mark.parametrize(
    "target, direction",
    [
        ({"param_a": 10, "param_b": 10}, "maximize"),
        ({"param_a": 0, "param_b": 0}, "minimize"),
    ],
)
def test_basic_optimization(target, direction: Direction):
    class SumPipeline(Pipeline):
        def __init__(self):
            super().__init__()
            self.param_a: int = Integer(0, 10)
            self.param_b: int = Integer(0, 10)

        def __call__(self, data: float) -> float:
            return data + self.param_a + self.param_b

        def loss(self, data: float, y_preds: float) -> float:
            return y_preds

        def get_direction(self) -> Direction:
            return direction

    optimizer_tester(pipeline=SumPipeline(), target=target)


@pytest.mark.parametrize(
    "target, direction",
    [
        ({"param_dict": {"param_a": 10, "param_b": 10}}, "maximize"),
        ({"param_dict": {"param_a": 0, "param_b": 0}}, "minimize"),
    ],
)
def test_structured_dict_param_optim(target, direction: Direction):
    class SumPipeline(Pipeline):
        def __init__(self):
            super().__init__()
            self.param_dict: dict[str, int] = ParamDict(
                param_a=Integer(0, 10), param_b=Integer(0, 10)
            )

        def __call__(self, data: float) -> float:
            return data + self.param_dict["param_b"] + self.param_dict["param_a"]

        def loss(self, data: float, y_preds: float) -> float:
            return y_preds

        def get_direction(self) -> Direction:
            return direction

    optimizer_tester(pipeline=SumPipeline(), target=target)


def test_multi_objective_loss_optimization():
    class TradeoffPipeline(Pipeline):
        def __init__(self):
            super().__init__()
            self.param = Integer(0, 4)

        def __call__(self, data: float) -> float:
            return self.param

        def loss(self, data: float, y_pred: float):
            return y_pred, abs(y_pred - 2)

        def get_direction(self):
            return "minimize", "minimize"

    optimizer = Optimizer(
        TradeoffPipeline(), sampler=GridSampler({"param": list(range(5))})
    )
    result = optimizer.tune([0.0], n_iterations=5, show_progress=False)

    pareto_front = optimizer.pareto_front
    assert result == {"pareto_front": pareto_front}
    assert {point["params"]["param"] for point in pareto_front} == {0, 1, 2}
    assert all(set(point) == {"number", "values", "params"} for point in pareto_front)

    with pytest.raises(RuntimeError, match="pareto_front"):
        _ = optimizer.best_loss
    with pytest.raises(RuntimeError, match="pareto_front"):
        _ = optimizer.best_params
    with pytest.raises(RuntimeError, match="pareto_front"):
        _ = optimizer.best_pipeline


def test_multi_objective_metrics():
    class AccumulatedMetric:
        def __init__(self, name, transform):
            self.name = name
            self.transform = transform
            self.values = []

        def __call__(self, reference, hypothesis, uem=None):
            value = self.transform(hypothesis)
            self.values.append(value)
            return value

        def __abs__(self):
            return np.mean(self.values)

        def confidence_interval(self, alpha=0.9):
            value = abs(self)
            return value, (value, value)

    class TradeoffPipeline(Pipeline):
        def __init__(self):
            super().__init__()
            self.param = Integer(0, 4)

        def __call__(self, data) -> float:
            return self.param

        def get_metric(self):
            return (
                AccumulatedMetric("score", lambda value: value),
                AccumulatedMetric("distance", lambda value: abs(value - 2)),
            )

        def get_direction(self):
            return "maximize", "minimize"

    optimizer = Optimizer(
        TradeoffPipeline(),
        sampler=GridSampler({"param": list(range(5))}),
        average_case=True,
    )
    optimizer.tune(
        [{"annotation": None, "annotated": None}],
        n_iterations=5,
        show_progress=False,
    )

    assert optimizer.study_.metric_names == ["score", "distance"]
    assert {point["params"]["param"] for point in optimizer.pareto_front} == {
        2,
        3,
        4,
    }


def test_multi_objective_rejects_pruning():
    class MultiObjectivePipeline(Pipeline):
        def get_direction(self):
            return "minimize", "maximize"

    with pytest.raises(ValueError, match="does not support trial pruning"):
        Optimizer(MultiObjectivePipeline(), pruner=MedianPruner())


def test_objective_count_must_match_direction_count():
    class InvalidPipeline(Pipeline):
        def __call__(self, data):
            return data

        def loss(self, data, output):
            return 0.0

        def get_direction(self):
            return "minimize", "maximize"

    optimizer = Optimizer(InvalidPipeline())
    with pytest.raises(ValueError, match="1 loss values.*2 directions"):
        optimizer.tune([0.0], n_iterations=1, show_progress=False)
