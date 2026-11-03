
#### Description
<!-- Example: DiarizationPurity does not return the expected value -->

#### Steps/Code to Reproduce
<!--
Example:

```
from pyannote.core import Annotation, Segment
reference = Annotation()
reference[Segment(0, 10)] = 'A'
reference[Segment(12, 20)] = 'B'
reference[Segment(24, 27)] = 'A'
reference[Segment(30, 40)] = 'C'
hypothesis = Annotation()
hypothesis[Segment(2, 13)] = 'a'
hypothesis[Segment(13, 14)] = 'd'
hypothesis[Segment(14, 20)] = 'b'
hypothesis[Segment(22, 38)] = 'c'
hypothesis[Segment(38, 40)] = 'd'

from pyannote.metrics.diarization import DiarizationPurity
purity = DiarizationPurity()
print "Purity = {0:.3f}".format(purity(reference, hypothesis))
```
If the code is too long, feel free to put it in a public gist and link
it in the issue: https://gist.github.com
-->

#### Expected Results
<!-- Example: Output should be "Purity = 0.667". Please paste or describe the expected results.-->

#### Actual Results
<!-- Please paste or specifically describe the actual output or traceback. -->

#### Versions
<!-- Please paste the output of `pip freeze | grep pyannote` below -->

<!-- Thanks for contributing! -->
