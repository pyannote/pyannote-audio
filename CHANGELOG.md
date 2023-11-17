# Changelog

## `develop` branch

## Version 3.1.0 (2023-11-16)

### TL;DR

[`pyannote/speaker-diarization-3.1`](https://hf.co/pyannote/speaker-diarization-3.1) no longer requires [unpopular](https://github.com/pyannote/pyannote-audio/issues/1537) ONNX runtime

### New features

- feat(model): add WeSpeaker embedding wrapper based on PyTorch
- feat(model): add support for multi-speaker statistics pooling
- feat(pipeline): add `TimingHook` for profiling processing time
- feat(pipeline): add `ArtifactHook` for saving internal steps
- feat(pipeline): add support for list of hooks with `Hooks`
- feat(utils): add `"soft"` option to `Powerset.to_multilabel`

### Fixes

- fix(pipeline): add missing "embedding" hook call in `SpeakerDiarization`
- fix(pipeline): fix `AgglomerativeClustering` to honor `num_clusters` when provided
- fix(pipeline): fix frame-wise speaker count exceeding `max_speakers` or detected `num_speakers` in `SpeakerDiarization` pipeline

### Improvements

- improve(pipeline): compute `fbank` on GPU when requested

### Breaking changes

- BREAKING(pipeline): rename `WeSpeakerPretrainedSpeakerEmbedding` to `ONNXWeSpeakerPretrainedSpeakerEmbedding`
- BREAKING(setup): remove `onnxruntime` dependency.
  You can still use ONNX `hbredin/wespeaker-voxceleb-resnet34-LM` but you will have to install `onnxruntime` yourself.
- BREAKING(pipeline): remove `logging_hook` (use `ArtifactHook` instead)
- BREAKING(pipeline): remove `onset` and `offset` parameter in `SpeakerDiarizationMixin.speaker_count`
  You should now binarize segmentations before passing them to `speaker_count`

## Version 3.0.1 (2023-09-28)

- fix(pipeline): fix WeSpeaker GPU support

## Version 3.0.0 (2023-09-26)

### Features and improvements

- feat(pipeline): send pipeline to device with `pipeline.to(device)`
- feat(pipeline): add `return_embeddings` option to `SpeakerDiarization` pipeline
- feat(pipeline): make `segmentation_batch_size` and `embedding_batch_size` mutable in `SpeakerDiarization` pipeline (they now default to `1`)
- feat(pipeline): add progress hook to pipelines
- feat(task): add [powerset](https://www.isca-speech.org/archive/interspeech_2023/plaquet23_interspeech.html) support to `SpeakerDiarization` task
- feat(task): add support for multi-task models
- feat(task): add support for label scope in speaker diarization task
- feat(task): add support for missing classes in multi-label segmentation task
- feat(model): add segmentation model based on torchaudio self-supervised representation
- feat(pipeline): check version compatibility at load time
- improve(task): load metadata as tensors rather than pyannote.core instances
- improve(task): improve error message on missing specifications

### Breaking changes

- BREAKING(task): rename `Segmentation` task to `SpeakerDiarization`
- BREAKING(pipeline): pipeline defaults to CPU (use `pipeline.to(device)`)
- BREAKING(pipeline): remove `SpeakerSegmentation` pipeline (use `SpeakerDiarization` pipeline)
- BREAKING(pipeline): remove `segmentation_duration` parameter from `SpeakerDiarization` pipeline (defaults to `duration` of segmentation model)
- BREAKING(task): remove support for variable chunk duration for segmentation tasks
- BREAKING(pipeline): remove support for `FINCHClustering` and `HiddenMarkovModelClustering`
- BREAKING(setup): drop support for Python 3.7
- BREAKING(io): channels are now 0-indexed (used to be 1-indexed)
- BREAKING(io): multi-channel audio is no longer downmixed to mono by default.
  You should update how `pyannote.audio.core.io.Audio` is instantiated:
  - replace `Audio()` by `Audio(mono="downmix")`;
  - replace `Audio(mono=True)` by `Audio(mono="downmix")`;
  - replace `Audio(mono=False)` by `Audio()`.
- BREAKING(model): get rid of (flaky) `Model.introspection`
  If, for some weird reason, you wrote some custom code based on that,
  you should instead rely on `Model.example_output`.
- BREAKING(interactive): remove support for Prodigy recipes

### Fixes and improvements

- fix(pipeline): fix reproducibility issue with Ampere CUDA devices
- fix(pipeline): fix support for IOBase audio
- fix(pipeline): fix corner case with no speaker
- fix(train): prevent metadata preparation to happen twice
- fix(task): fix support for "balance" option
- improve(task): shorten and improve structure of Tensorboard tags

### Dependencies update

- setup: switch to torch 2.0+, torchaudio 2.0+, soundfile 0.12+, lightning 2.0+, torchmetrics 0.11+
- setup: switch to pyannote.core 5.0+, pyannote.database 5.0+, and pyannote.pipeline 3.0+
- setup: switch to speechbrain 0.5.14+

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

## Version 1.0.1 (2018-07-19)

- fix: fix regression in Precomputed.**call** (#110, #105)

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
