from collections.abc import Hashable, Iterator
from typing import TYPE_CHECKING, Literal, Union

if TYPE_CHECKING:
    from ..annotation import Annotation
    from ..feature import SlidingWindowFeature
    from ..segment import Segment
    from ..timeline import Timeline

Label = Hashable
Support = Union["Segment", "Timeline"]
LabelGeneratorMode = Literal["int", "string"]
LabelGenerator = LabelGeneratorMode | Iterator[Label]
TrackName = str | int
Key = Union["Segment", tuple["Segment", TrackName]]
Resource = Union["Segment", "Timeline", "SlidingWindowFeature", "Annotation"]
CropMode = Literal["intersection", "loose", "strict"]
Alignment = Literal["center", "loose", "strict"]
LabelStyle = tuple[str, int, tuple[float, float, float]]
