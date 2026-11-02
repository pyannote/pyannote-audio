#########
Tutorial
#########

This tutorial will guide you through a simple example on how to use `pyannote.metrics` for evaluation purposes.

`pyannote.metrics` internally relies on :class:`pyannote.core.Annotation` data structure to store reference and hypothesis annotations.

.. plot:: pyplots/tutorial.py


.. ipython::

   In [10]: from pyannote.core import Segment, Timeline, Annotation

   In [11]: reference = Annotation()
      ....: reference[Segment(0, 10)] = 'A'
      ....: reference[Segment(12, 20)] = 'B'
      ....: reference[Segment(24, 27)] = 'A'
      ....: reference[Segment(30, 40)] = 'C'

   In [12]: hypothesis = Annotation()
      ....: hypothesis[Segment(2, 13)] = 'a'
      ....: hypothesis[Segment(13, 14)] = 'd'
      ....: hypothesis[Segment(14, 20)] = 'b'
      ....: hypothesis[Segment(22, 38)] = 'c'
      ....: hypothesis[Segment(38, 40)] = 'd'


Several evaluation metrics are available, including the diarization error rate:


.. ipython::
   :okwarning:

   In [13]: from pyannote.metrics.diarization import DiarizationErrorRate

   In [14]: metric = DiarizationErrorRate()

   In [15]: metric(reference, hypothesis)
   Out[15]: 0.516

That's it for the tutorial.
`pyannote.metrics` can do much more than that! Keep reading...
