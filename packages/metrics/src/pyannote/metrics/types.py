from typing import Literal

MetricComponent = str
CalibrationMethod = Literal["isotonic", "sigmoid"]
MetricComponents = list[MetricComponent]
Details = dict[MetricComponent, float]
