### Version 1.1 (WIP)

  - BREAKING: switch to new pyannote.pipeline package
  - BREAKING: add unified FeatureExtraction base class
  - feat: add support for on-the-fly data augmentation
  - setup: switch to librosa 0.6

### Version 1.0.1 (2018--07-19)

  - fix: fix regression in Precomputed.__call__ (#110, #105)

### Version 1.0 (2018-07-03)

  - chore: switch from keras to pytorch (with tensorboard support)
  - improve: faster & better traning (`AutoLR`, advanced learning rate schedulers, improved batch generators)
  - feat: add tunable speaker diarization pipeline (with its own tutorial)
  - chore: drop support for Python 2 (use Python 3.6 or later)

### Version 0.3.1 (2017-07-06)

  - feat: add python 3 support
  - chore: rewrite neural speaker embedding using autograd
  - feat: add new embedding architectures
  - feat: add new embedding losses
  - chore: switch to Keras 2
  - doc: add tutorial for (MFCC) feature extraction
  - doc: add tutorial for (LSTM-based) speech activity detection
  - doc: add tutorial for (LSTM-based) speaker change detection
  - doc: add tutorial for (TristouNet) neural speaker embedding

### Version 0.2.1 (2017-03-28)

  - feat: add LSTM-based speech activity detection
  - feat: add LSTM-based speaker change detection
  - improve: refactor LSTM-based speaker embedding
  - feat: add librosa basic support
  - feat: add SMORMS3 optimizer

### Version 0.1.4 (2016-09-26)

  - feat: add 'covariance_type' option to BIC segmentation

### Version 0.1.3 (2016-09-23)

  - chore: rename sequence generator in preparation of the release of
    TristouNet reproducible research package.

### Version 0.1.2 (2016-09-22)

  - first public version
