# Changelog

## Version 4.2.0 (2026-10-16)

- BREAKING: require Python 3.11 or newer; Python 3.10 is no longer supported
- feat: add diarization speaker count metrics
- fix: fix metrics for empty references (@Atishyy27)
- fix: restore score-distribution plots with current Matplotlib versions
- fix: report a helpful installation error when transcription metrics are used
  without the optional `transcription` dependencies

## Version 4.1.0 (2026-05-06)

- feat: add transcription metrics

## Version 4.0.0 (2025-09-09)

- fix: remove deprecated use of `np.NaN`
- BREAKING: drop support to `Python` < 3.10
- BREAKING: switch to native namespace package
- setup: switch to `uv`

## Version 3.3.0 (2025-01-12)

- BREAKING: improve diarization purity and coverage to account for overlapping regions
- chore: use `bool` instead of deprecated `np.bool`

## Version 3.2.1 (2022-06-20)

- fix: fix corner case for confidence interval
- doc: add type hinting (@hadware)

## Version 3.2 (2022-01-12)

- feat: add option to override existing "uri"
- feat: add support for missing "uri"

## Version 3.1 (2021-09-27)

- BREAKING: remove (buggy) support for parallel processing
- fix: fix documentation deployment

## Version 3.0.1 (2020-07-02)

- setup: switch to pyannote.database 4.0+

## Version 3.0 (2020-06-15)

- feat: add DetectionCostFunction detection metric (@nryant)
- BREAKING: rename pyannote-metrics.py CLI to pyannote-metrics

## Version 2.3 (2020-02-26)

- feat: add DetectionPrecisionRecallFMeasure compound metric (@MarvinLvn)
- fix: fix corner "in f-measure" case when both precision and recall are zero (@MarvinLvn)
- fix: fix a typo in documentation (@wq2012)

## Version 2.2 (2019-12-13)

- feat: add support for evaluation of overlapped speech detection
- feat: setup continuous integration
- setup: switch to pyannote.core 3.2

## Version 2.1 (2019-06-24)

- chore: rewrite mapping and matching routines
- fix: remove buggy xarray dependency
- setup: switch to pyannote.core 3.0

## Version 2.0.2 (2019-04-15)

- fix: avoid division by zero

## Version 2.0.1 (2019-03-20)

- BREAKING: drop support for all file formats but RTTM
- BREAKING: drop Python 2.7 support
- setup: switch to pyannote.database 2.0
- setup: switch to pyannote.core 2.1

## Version 1.8.1 (2018-11-19)

- setup: switch to pyannote.core 2.0

## Version 1.8 (2018-09-03)

- feat: add compound segmentation metric SegmentationPurityCoverageFMeasure (@diego-fustes)
- fix: fix typo in IdentificationErrorAnalysis (@benjisympa)

## Version 1.7.1 (2018-09-03)

- fix: fix broken images in documentation

## Version 1.7 (2018-03-17)

- feat: add option to filter out target trials in "spotting" mode
- chore: default to "parallel=False"

## Version 1.6.1 (2018-02-05)

- fix: fix Diarization{Purity | Coverage} with empty references
- improve: improve support for speaker spotting experiments
- chore: (temporarily?) remove parallel processing in pyannote.metrics.py
- setup: drop support for Python 2

## Version 1.5 (2017-10-20)

- feat: add fixed vs. variable latency switch for LLSS

## Version 1.4.3 (2017-10-17)

- fix: add more safety checks to pyannote-metrics.py "spotting" mode
- setup: switch to pyannote.core 1.2, pyannote.database 1.1, pyannote.parser 0.7

## Version 1.4.2 (2017-10-13)

- improve: set latency of missed detections to maximum possible value
- improve: improve instructions in pyannote-metrics.py --help

## Version 1.4.1 (2017-10-02)

- feat: add LowLatencySpeakerSpotting metric
- feat: add "spotting" mode to pyannote-metrics.py
- setup: switch to pyannote.database 1.0

## Version 1.3 (2017-09-19)

- feat: add "skip_overlap" option to not evaluate overlapping speech regions
- improve: bring performance improvement to diarization metrics
- fix: fix a bug where collar was applied twice in DiarizationErrorRate
- fix: add collar support to purity/coverage/homogeneity/completeness
- fix: fix a bug happening in 'uemify' when both reference and hypothesis are empty
- fix: fix a "division by zero" error in homogeneity/completeness
- setup: switch to pyannote.core 1.1 (major performance improvements)

## Version 1.2 (2017-07-21)

- feat: add method DiarizationPurityCoverageFMeasure.compute_metrics to get
  purity, coverage, and their F-measure (all at once)

## Version 1.1 (2017-07-20)

- feat: add new metric 'DiarizationPurityCoverageFMeasure'
- doc: update installation instructions
- setup: switch to pyannote.core 1.0.4

## Version 1.0 (2017-07-04)

- setup: switch to pyannote.core 1.0
- feat: add score calibration for binary classification tasks
- doc: update citation

## Version 0.14.4 (2017-03-27)

- doc: update notebook to latest version

## Version 0.14.3 (2017-03-27)

- doc: add Sphinx documentation

## Version 0.14.2 (2017-03-21)

- feat: better README and technical report

## Version 0.14.1 (2017-03-16)

- chore: rename SegmentationError to SegmentationErrorAnalysis
- fix: fix DetectionErrorRate support for kwargs

## Version 0.14 (2017-02-06)

- feat: add "parallel" option to not use multiprocessing
- feat: add "accuracy" in "detection" report
- setup: switch to pyannote.core 0.13
- setup: switch to pyannote.parser 0.6.5

## Version 0.13.2 (2017-01-30)

- feat: add pyannote-metrics.py evaluation script
- fix: fix BaseMetric.report() for metric without a 'total' component
- fix: fix (Greedy)DiarizationErrorRate uem handling
- fix: fix (Greedy)DiarizationErrorRate parallel processing
- setup: switch to pyannote.core 0.12
- setup: update munkres & matplotlib dependencies

## Version 0.12.1 (2017-01-27)

- feat: support for multiprocessing
- feat: add report() method
- feat: travis continuous integration (finally!)
- improve: speed up detection metrics
- feat: add unit tests for detection metrics
- fix: fix python 3 support
- setup: remove dependency to pyannote.algorithms
- setup: switch to pyannote.core 0.11

## Version 0.11 (2016-12-13)

- feat: add pyannote.metrics.binary_classification module

## Version 0.10.3 (2016-11-28)

- fix: fix (greedy) diarization error rate
- feat: add support for 'collar' to (greedy) diarization error rate

## Version 0.10.2 (2016-11-10)

- fix: fix default "xlim" in "plot_distributions"
- setup: switch to pyannote.core 0.8 and pyannote.algorithms 0.6.6

## Version 0.10.1 (2016-11-05)

- feat: add "uem" support to diarization metrics

## Version 0.9 (2016-09-23)

- feat: add plotting functions for binary classification tasks

## Version 0.8 (2016-08-25)

- feat: detection accuracy
- refactor: detection metrics
- setup: update to pyannote.core 0.7.2

## Version 0.7.1 (2016-06-24)

- setup: update to pyannote.core 0.6.6

## Version 0.7 (2016-04-04)

- feat: greedy diarization error rate

## Version 0.6.0 (2016-03-29)

- feat: Python 3 support
- feat: unit tests
- wip: travis

## Version 0.5.1 (2016-02-19)

- refactor: diarization metrics

## Version 0.4.1 (2014-11-20)

- fix: identification error analysis matrix confusion

## Version 0.4 (2014-10-31)

- feat(error): identification regression analysis
- feat: new pyannote_eval.py CLI

## Version 0.3 (2014-10-01)

- feat(error): segmentation error analysis

## Version 0.2 (2014-08-05)

- feat(detection): add precision and recall
- fix(identification): fix precision and recall

## Version 0.1 (2014-06-27)

- feat(segmentation): add precision and recall
- feat(identification): add support for NIST collar
- feat(error): add module for detailed error analysis

## Version 0.0.1 (2014-06-04)

- first public version
