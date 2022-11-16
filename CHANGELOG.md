# Changelog

## Version 2.1.1 (2022-10-27)

  - BREAKING(pipeline): rewrite speaker diarization pipeline
  - feat(pipeline): add option to optimize for DER variant
  - feat(clustering): add support for NeMo speaker embedding
  - feat(clustering): add FINCH clustering
  - feat(clustering): add min_cluster_size hparams to AgglomerativeClustering
  - feat(hub): add support for private/gated models
  - setup(hub): switch to latest hugginface_hub API
  - fix(pipeline): fix support for missing reference in Resegmentation pipeline
  - fix(clustering) fix corner case where HMM.fit finds too little states

## Version 2.0.1 (2022-07-20)

  - BREAKING: complete rewrite
  - feat: much better performance
  - feat: Python-first API
  - feat: pretrained pipelines (and models) on Huggingface model hub
  - feat: multi-GPU training with pytorch-lightning
  - feat: data augmentation with torch-audiomentations
  - feat: Prodigy recipe for model-assisted audio annotation

## Version 1.1.2 (2021-01-28)

  - fix: make sure master branch is used to load pretrained models (#599)

## Version 1.1 (2020-11-08)

  - last release before complete rewriting

## Version 1.0.1 (2018--07-19)

  - fix: fix regression in Precomputed.__call__ (#110, #105)

## Version 1.0 (2018-07-03)

  - chore: switch from keras to pytorch (with tensorboard support)
  - improve: faster & better traning (`AutoLR`, advanced learning rate schedulers, improved batch generators)
  - feat: add tunable speaker diarization pipeline (with its own tutorial)
  - chore: drop support for Python 2 (use Python 3.6 or later)

## Version 0.3.1 (2017-07-06)

  - feat: add python 3 support
  - chore: rewrite neural speaker embedding using autograd
  - feat: add new embedding architectures
  - feat: add new embedding losses
  - chore: switch to Keras 2
  - doc: add tutorial for (MFCC) feature extraction
  - doc: add tutorial for (LSTM-based) speech activity detection
  - doc: add tutorial for (LSTM-based) speaker change detection
  - doc: add tutorial for (TristouNet) neural speaker embedding

## Version 0.2.1 (2017-03-28)

  - feat: add LSTM-based speech activity detection
  - feat: add LSTM-based speaker change detection
  - improve: refactor LSTM-based speaker embedding
  - feat: add librosa basic support
  - feat: add SMORMS3 optimizer

## Version 0.1.4 (2016-09-26)

  - feat: add 'covariance_type' option to BIC segmentation

## Version 0.1.3 (2016-09-23)

  - chore: rename sequence generator in preparation of the release of
    TristouNet reproducible research package.

## Version 0.1.2 (2016-09-22)

  - first public version
