# CHANGELOG

## Version 6.1.0 (2026-10-15)

- BREAKING: require Python 3.11 or newer; Python 3.10 is no longer supported
- BREAKING: `from pyannote.core import *` now exports only public classes, constants,
  and `notebook` when available; incidental module imports are no longer exported
- fix(cluster): restore pooling clustering with NumPy 2
- fix(cluster): place the larger cluster first in linkage rows, including
  must-link merges; cluster membership and merge distances are unchanged

## Version 6.0.1 (2025-09-16)

- feat: add `Annotation.__iter__` (turn: Segment, speaker: str) iterator

## Version 6.0.0 (2025-09-09)

- BREAKING: drop support to `Python` < 3.10
- BREAKING: switch to native namespace package 
- setup: switch to `uv`

## Version 5.0.1 (2024-06-14)

- fix: fix return type of Timeline.__iter__ (@domsmrz)
- improve: remove MatplotlibDeprecationWarning (@kimdwkimdw)

## Version 5.0.0 (2022-12-15)

- BREAKING: remove support for JSON serialization

## Version 4.5 (2022-08-24)

- feat: add Annotation.write_lab method (@FrancescoBonzi)
- feat: add Annotation.to_{rttm|lab} and Timeline.to_uem serializers (@juanmc2005)
- setup: exclude tests from package (@kimdwkimdw)

## Version 4.4 (2022-03-09)

- BREAKING: remove empty segments in Timeline.__init__
- BREAKING: Timeline.extent() returns Segment(0.0, 0.0) for empty timelines
- feat: add "duration" option to Annotation.discretize
- fix: handle various corner cases in 1D pdist and cdist
- fix: fix documentation of {Timeline | Annotation}.__bool__
- test: check robustness to Segment.set_precision

## Version 4.3 (2021-10-11)

- feat: add support for visualizing 3D SlidingWindowFeatures (#66)
- feat: add Annotation.discretize (#65)

## Version 4.2.1 (2021-09-27)

- fix: fix automatic deployment of documentation

## Version 4.2 (2021-09-24)

- feat: add Segment.set_precision to automatically round start/end timestamps (@kimdwkimdw)
- feat: add methods {Timeline|Annotation}.{extrude|get_overlap} (@hadware)
- feat: add support for `Segment` in `one_hot_encoding`
- feat: make notebook visualization support an optional installation via the [notebook] extra_requires (@hadware)
- setup: add [doc] and [testing] extra_requires (@hadware)
- fix: several type annotation fixes (@kimdwkimdw, @PaulLerner)

## Version 4.1 (2020-05-27)

- feat: add "must_link_method" option to "pool" clustering

## Version 4.0 (2020-05-15)

- feat: add support for cannot/must link constraints in "pool" clustering
- BREAKING: make one_hot_encoding return SlidingWindowFeature with labels
- feat: add "labels" optional attribute to SlidingWindowFeature

## Version 3.7.1 (2020-03-23)

- fix: fix corner case in pyannote.core.utils.numpy.one_hot_encoding

## Version 3.7 (2020-03-21)

- feat: add new method SlidingWindowFeature.align

## Version 3.6 (2020-03-03)

- feat: add collar argument to Timeline.support (@nryant)
- feat: add new method Timeline.covers (@PaulLerner)
- fix: check for spaces before writing with RTTM or UEM format (@PaulLerner)
- improve: speed up Annotation.{from_df | from_json} (@nryant)

## Version 3.5 (2020-01-23)

- feat: add align_last option to SlidingWindow.__call__

## Version 3.4 (2020-01-21)

- feat: add SlidingWindow.__call__ to slide over a specific support
- feat: add pyannote.core.utils.random

## Version 3.3 (2020-01-08)

- BREAKING: remove pyannote.core.Scores
- fix: fix legend corner case with empty Annotation instances
- fix: fix {Timeline|Annotation}.crop with overlapping segments in support

## Version 3.2.2 (2020-01-06)

- fix: fix Segment.__bool__ return type

## Version 3.2 (2019-12-13)

- feat: add type hints (@hadware)
- chore: Python3-ize code base (@hadware)

## Version 3.1.10 (2019-12-12)

- feat: add Annotation.write_rttm and Timeline.write_uem (@PaulLerner)
- feat: add numpy interface to SlidingWindowFeature (experimental)
- feat: setup continuous integration
- improve: speed up SlidingWindowFeature plotting
- doc: update notebooks to Python 3 (@PaulLerner)
- BREAKING: remove deprecated Segment.pretty()

## Version 3.0 (2019-06-24)

- BREAKING: `Annotation.__mul__` now returns `np.ndarray` instance
- BREAKING: remove `Annotation` methods (that were deprecated version 1.0)
- BREAKING: `Scores.retrack` has been renamed `rename_tracks`
- improve: remove `six` and `xarray` dependencies
- BREAKING: remove `pyannote.core.time` module

## Version 2.2.2 (2019-04-18)

- improve: speed-up several `Segment` methods
- improve: decrease `one_hot_encoding` memory usage
- feat: add support for "fixed" option in all cropping modes
- setup: switch to scipy 1.1

## Version 2.1 (2019-01-17)

- feat: add basic implementation of Chinese Whispers clustering
- feat: add "utils.hierarchy.fcluster_auto" parameter-free "fcluster"

## Version 2.0.3 (2018-11-23)

- BREAKING: move pyannote.core.util to pyannote.core.utils.generators
- BREAKING: remove support for Python 2
- feat: add one_hot_{encoding | decoding} functions
- feat: add get_class_by_name utiltiy function
- feat: add custom pdist and cdist
- feat: add custom hierachical clustering "linkage" function
- setup: add scipy dependency

## Version 1.4.1 (2018-09-13)

- feat: add unit tests for feature cropping
- fix: fix out-of-bounds feature cropping

## Version 1.4 (2018-07-09)

- feat: expose `SlidingWindow.closest_frame`

## Version 1.3.2 (2018-06-12)

- feat: add support for multi-dimensional SlidingWindowFeatures
- setup: switch to sortedcontainers 2.x
- fix: update SWF.crop docstring
- chore: remove support for "segment" option in Annotation.argmax

## Version 1.3.1 (2017-12-14)

- feat: add "return_ranges" option to SlidingWindow.crop
- improve: faster SlidingWindowFeature.crop
- fix: fix documentation

## Version 1.2 (2017-10-05)

- BREAKING: remove all things "Transcription"
- feat: add return_data parameter to SlidingWindowFeature.crop (SWF)
- feat: add support for len(SWF)
- feat: add ylim parameter to plot_feature
- fix: fix corner case where notebook.crop is larger than SWF extent

## Version 1.1 (2017-09-19)

- feat: make iterators out of SlidingWindowFeature instances
- improve: make Annotation.__mul__ (much) faster

## Version 1.0.5 (2017-09-13)

- fix: make iterators out of SlidingWindow instances

## Version 1.0.4 (2017-07-18)

- fix: add missing import
- fix: fix corner case in Timeline.crop_iter

## Version 1.0.2 (2017-07-15)

- fix: use Timeline.support() instead of .coverage()

## Version 1.0.1 (2017-07-04)

- improve: switch from banyan to sortedcontainers
- feat: add Timeline.{remove|discard|overlapping_iter} methods
- BREAKING: Timeline.__init__ now raises ValueError in case of empty segment
- BREAKING: Timeline.crop now raises ValueError for bad mode
- BREAKING: rename "mapping" arguments to "returns_mapping" in Timeline.crop
- test: add more tests

## Version 0.13.3 (2017-06-29)

- fix: fix SlidingWindowFeature.iterfeatures()

## Version 0.13.2 (2017-03-29)

- setup: add dependencies for notebook visualization

## Version 0.13.1 (2017-02-20)

- fix: fix Annotation.rename_labels

## Version 0.13 (2017-02-05)

- improve: faster Annotation.subset and Annotation.rename_labels

## Version 0.12.1 (2017-30-01)

- fix: fix Annotation.uri setter

## Version 0.12 (2017-29-01)

- feat: add Timeline.to_annotation()

## Version 0.11.1 (2017-25-01)

- fix: fix (deprecated) "smooth" method

## Version 0.11 (2017-24-01)

- feat: add (Sphinx-based) documentation
- chore: move sample notebooks to /notebook
- feat: add unit tests
- BREAKING: rename some Timeline and Annotation methods

## Version 0.10 (2017-18-01)

-  feat: add 'copy' parameter to Annotation.{label|get}_timeline()
-  improve: speed-up Timeline.extent()
-  chore: move tests at root directory
-  chore: remove support for Unknown labels

## Version 0.9 (2017-01-17)

-  improve: speed up Timeline and Annotation

## Version 0.8 (2016-11-05)

-  feat: add "copy" option to Annotation.update

## Version 0.7.3 (2016-11-01)

-  feat: SlidingWindowFeature notebook display

## Version 0.7.2 (2016-07-12)

-  feat: new SlidingWindow.{samples|crop} methods
-  feat: new 'mode' parameter to SlidingWindowFeature.crop method
-  doc: updated notebooks for SlidingWindow and SlidingWindowFeature

## Version 0.6.6 (2016-06-23)

-  fix: force internal timeline update after copy

## Version 0.6.5 (2016-06-13)

-  BREAKING: make segmentToRange deterministic wrt. segment duration

## Version 0.6.4 (2016-06-06)

-  fix: Python 3 support in pyannote.core.features

## Version 0.6.3 (2016-03-29)

-  setup: versioneer 0.15

## Version 0.6.1 (2016-03-20)

-  fix: prevent adding empty segments in Annotation and Scores

## Version 0.6 (2016-02-25)

-  BREAKING: pyannote.core.json.{load|dump} expects file handles
-  feat: load_from, dump_to

## Version 0.5.2 (2016-02-19)

-  feat: Annotation * Annotation returns cooccurrence matrix
-  fix: Annotation.itertracks would raise a UnicodeDecodeError in some
   cases

## Version 0.5.1 (2016-02-17)

-  improve: notebook display
-  improve: Annotation.anonymize_{tracks|labels} no longer use Unknown
   instances
-  improve: empty segments are now printed as "[]"

## Version 0.4.7 (2016-02-04)

-  feat: deterministic order in Annotation.co_iter
-  fix: LabelMatrix.argmax corner case
-  setup: update dependencies

## Version 0.4.4 (2015-11-02)

-  feat: Travis continuous integration

## Version 0.4.3 (2015-10-28)

-  fix: Python 2/3 notebook representations
-  fix: bug in Scores with integer-values segments

## Version 0.4.1 (2015-10-27)

-  fix: update Scores.from_df to pandas 0.17

## Version 0.4 (2015-10-26)

-  feat: Python 3 support
-  feat: pytest test suite
-  fix: Annotation comparison
-  fix: deterministic order in Annotation.itertracks

## Version 0.3.6 (2015-05-06)

-  feat: LabelMatrix save/load methods

## Version 0.3.4 (2015-03-04)

-  fix: MAJOR bug in Annotation lazy-update

## Version 0.3.3 (2015-02-27)

-  fix: Scores IPython display

## Version 0.3.1 (2015-01-26)

-  feat: new Annotation.update method
-  improve: Annotation.subset support for any label iterable

## Version 0.3 (2014-12-04)

-  refactor: rewrote Scores internals
-  setup: use pandas 0.15.1+

## Version 0.2.5 (2014-11-21)

-  setup: use pyannote-banyan 0.1.6

## Version 0.2.4 (2014-11-18)

-  fix: extent of empty Timeline

## Version 0.2.3 (2014-11-14)

-  fix: force revert to pandas 0.13.1 for Scores to work again...

## Version 0.2.2 (2014-11-12)

-  setup: use banyan 0.1.5.1 from GitHub

## Version 0.2.1 (2014-10-30)

-  feat: pyannote/core Docker image
-  feat(Timeline): add from_df constructor

## Version 0.2 (2014-10-24)

-  breaking change: new PyAnnote JSON format

## Version 0.1 (2014-08-05)

-  fix(Transcription): fix potential edge/key conflict during alignment

## Version 0.0.5 (2014-07-23)

-  feat(SlidingWindow): add durationToSamples (and vice-versa)
-  fix(Transcription): fix loading from JSON
-  fix(Transcription): fix cropping corner cases
-  docs: add installation instruction for IPython display support
-  docs(Scores): add IPython documentation for Scores

## Version 0.0.3 (2014-06-02)

-  feat(Annotation): add 'collar' param to .smooth()
-  refactor(Annotation): remove support for >> operator
-  maintain(Mapping): remove label mapping data structure
-  feat(LabelMatrix): add IPython display
-  improve(LabelMatrix): 10x faster cooccurrence matrix
-  feat(Scores): add IPython display
-  feat(Transcription): add edge timerange prediction
-  feat(Transcription): add node temporal sort
-  fix(Transcription): make label_timeline return a copy
-  fix(Transcription): fix IPython display
-  docs(Transcription): add IPython documentation for Transcription

## Version 0.0.2 (2014-05-06)

-  feat: Transcription data structure (annotation graph)

## Version 0.0.1 (2014-05-02)

-  first public version
