Principles
==========

`pyannote.metrics` provides a set of classes to compare the output of speaker diarization (hereafter called `hypothesis`) systems to manual annotations (`reference`). Let us first instantiate a sample `reference` and `hypothesis`.

.. ipython::

   In [10]: from pyannote.core import Segment, Timeline, Annotation


   In [11]: reference = Annotation(uri='file1')
      ....: reference[Segment(0, 10)] = 'A'
      ....: reference[Segment(12, 20)] = 'B'
      ....: reference[Segment(24, 27)] = 'A'
      ....: reference[Segment(30, 40)] = 'C'

   In [12]: hypothesis = Annotation(uri='file1')
      ....: hypothesis[Segment(2, 13)] = 'a'
      ....: hypothesis[Segment(13, 14)] = 'd'
      ....: hypothesis[Segment(14, 20)] = 'b'
      ....: hypothesis[Segment(22, 38)] = 'c'
      ....: hypothesis[Segment(38, 40)] = 'd'


.. plot:: pyplots/tutorial.py

This basically tells us that, according to the manual annotation, speaker `A` speaks in timeranges [0s, 10s] and [24s, 27s].


.. note::

    Overlapping segments are supported. See :mod:`pyannote.core` documentation for more details.

`pyannote.metrics` follows an object-oriented paradigm.
Most evaluation metrics (e.g. :class:`DiarizationErrorRate` below) inherit from :class:`BaseMetric`.
As such, they share a common set of methods.

For instance, once instantiated, they can be called directly to compute the value of the evaluation metric.

.. ipython::
   :okwarning:

   In [10]: from pyannote.metrics.diarization import DiarizationErrorRate

   In [1]: metric = DiarizationErrorRate()

   In [1]: metric(reference, hypothesis)


Accumulation & reporting
------------------------

The same metric instance can be used to evaluate multiple files.

.. ipython::
   :okwarning:

   In [11]: other_reference = Annotation(uri='file2')
      ....: other_reference[Segment(0, 5)] = 'A'
      ....: other_reference[Segment(6, 10)] = 'B'
      ....: other_reference[Segment(12, 13)] = 'B'
      ....: other_reference[Segment(15, 20)] = 'A'

   In [12]: other_hypothesis = Annotation(uri='file2')
      ....: other_hypothesis[Segment(1, 6)] = 'a'
      ....: other_hypothesis[Segment(6, 7)] = 'b'
      ....: other_hypothesis[Segment(7, 10)] = 'c'
      ....: other_hypothesis[Segment(11, 19)] = 'b'
      ....: other_hypothesis[Segment(19, 20)] = 'a'

   In [12]: metric = DiarizationErrorRate()

   In [12]: metric(reference, hypothesis)

   In [12]: metric(other_reference, other_hypothesis)


You do not need to keep track of the result of each call yourself: this is done automatically.
For instance, once you have evaluated all files, you can use the overriden :func:`~pyannote.metrics.base.BaseMetric.__abs__` operator to get the accumulated value:

.. ipython::

   In [12]: abs(metric)

:func:`~pyannote.metrics.base.BaseMetric.report` provides a convenient summary of the result:

.. ipython::

   In [12]: report = metric.report(display=True)


The internal accumulator can be reset using the :func:`~pyannote.metrics.base.BaseMetric.report` method:

.. ipython::

   In [12]: metric.reset()


Evaluation map
--------------

Though audio files can always be processed entirely (from beginning to end), there are cases where reference annotations are only available for some regions of the audio files.
All metrics support the provision of an evaluation map that indicate which part of the audio file should be evaluated.

.. ipython::

   In [2]: uem = Timeline([Segment(0, 10), Segment(15, 20)])

   In [2]: metric(reference, hypothesis, uem=uem)


Components
----------

Most metrics are computed as the combination of several components.
For instance, the diarization error rate is the combination of false alarm (non-speech regions classified as speech), missed detection (speech regions classified as non-speech) and confusion between speakers.

Using ``detailed=True`` will return the value of each component:

.. ipython::
   :okwarning:

   In [13]: metric(reference, hypothesis, detailed=True)

The accumulated value of each component can also be obtained using the overriden :func:`~pyannote.metrics.base.BaseMetric.__getitem__` operator:

.. ipython::
   :okwarning:

   In [13]: metric(other_reference, other_hypothesis)

   In [13]: metric['confusion']

   In [13]: metric[:]


Define your own metric
----------------------

It is possible (and encouraged) to develop and contribute new evaluation metrics.

All you have to do is inherit from :class:`BaseMetric` and implement a few methods:
``metric_name``, ``metric_components``, ``compute_components``, and ``compute_metric``:

.. code-block:: python

    def is_male(speaker_name):
        # black magic that returns True if speaker is a man, False otherwise
        pass

    class MyMetric(BaseMetric):
        # This dummy metric computes the ratio between male and female speakers.
        # It does not actually use the reference annotation...

        @classmethod
        def metric_name(cls):
            # Return human-readable name of the metric

            return 'male / female ratio'

        @classmethod:
        def metric_components(cls):
            # Return component names from which the metric is computed

            return ['male', 'female']

        def compute_components(self, reference, hypothesis, **kwargs):
            # Actually compute the value of each component

            components = {'male': 0., 'female': 0.}

            for segment, _, speaker_name in hypothesis.itertracks(yield_label=True):
                if is_male(speaker_name):
                    components['male'] += segment.duration
                else:
                    components['female'] += segment.duration

            return components

        def compute_metric(self, components):
            # Actually compute the metric based on the component values

            return components['male'] / components['female']


See :class:`pyannote.metrics.base.BaseMetric` for more details.
