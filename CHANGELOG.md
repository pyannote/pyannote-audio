# CHANGELOG

## Version 4.0.2 (2025-11-19)

- BREAKING(util): make `Binarize.__call__` return `string` tracks (instead of `int`) [@benniekiss](https://github.com/benniekiss/)
- fix(torch): pin `torch`, `torchcodec`, and `torchaudio` versions to [avoid segmentation fault](https://github.com/meta-pytorch/torchcodec/issues/995) 
- fix(pyannoteAI): update pyannoteAI wrapper to return both regular and exclusive diarization
- feat(pipeline): add `Pipeline.cuda()` convenience method [@tkanarsky](https://github.com/tkanarsky/)
- feat(pipeline): add `preload` option to base `Pipeline.__call__` to force preloading audio in memory ([@antoinelaurent](https://github.com/antoinelaurent/))
- feat(cli): add option to apply pipeline on a directory of audio files
- improve(util): make `permutate` faster thanks to vectorized cost function

## Version 4.0.1 (2025-10-10)

- feat: allow passing preloaded pipeline config to `get_pipeline`
- setup: update `pyannoteai-sdk` dependency to `0.3.0`
- fix: relax version constraint on `OpenTelemetry` dependencies
- improve: warn (instead of raise) when passing unsupported arguments to speaker diarization pipeline

## Version 4.0.0 (2025-09-29) 

### TL;DR

#### Improved speaker assignment and counting

`pyannote/speaker-diarization-community-1` pretrained pipeline relies on VBx clustering instead of agglomerative hierarchical clustering (as suggested by [BUT Speech@FIT](https://speech.fit.vut.cz/) researchers [Petr Pálka](https://github.com/Selesnyan) and [Jiangyu Han](https://github.com/jyhan03)).

#### *Exclusive* speaker diarization

`pyannote/speaker-diarization-community-1` pretrained pipeline returns a new *exclusive* speaker diarization, on top of the regular speaker diarization.
This is a feature which is [backported from our latest commercial model](https://www.pyannote.ai/blog/precision-2) that simplifies the reconciliation between fine-grained speaker diarization timestamps and (sometimes not so precise) transcription timestamps.

```python
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1", token="huggingface-access-token")
output = pipeline("/path/to/conversation.wav")
print(output.speaker_diarization)            # regular speaker diarization
print(output.exclusive_speaker_diarization)  # exclusive speaker diarization
```

#### Faster training

Metadata caching and optimized dataloaders make training on large scale datasets much faster.  
This led to a 15x speed up on [pyannoteAI](https://www.pyannote.ai) internal large scale training.

#### [pyannoteAI](https://www.pyannote.ai) premium speaker diarization

Change one line of code to use [pyannoteAI](https://docs.pyannote.ai) premium models and enjoy **more accurate speaker diarization**.

```diff
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
-    "pyannote/speaker-diarization-community-1", token="huggingface-access-token")
+    "pyannote/speaker-diarization-precision-2, token="pyannoteAI-api-key")
diarization = pipeline("/path/to/conversation.wav")
```

#### Offline (air-gapped) use

Pipelines can now be stored alongside their internal models in the same repository, streamlining fully offline use.

1. Accept `pyannote/speaker-diarization-community-1` pipeline [user agreement](https://hf.co/pyannote/speaker-diarization-community-1)
2. Clone the pipeline repository from Huggingface (if prompted for a password, use a Huggingface access token with correct permissions)

    ```bash
    $ git lfs install
    $ git clone https://hf.co/pyannote/speaker-diarization-community-1 /path/to/directory/pyannote-speaker-diarization-community-1
    ```

3. Enjoy!

    ```python
    # load pipeline from disk (works without internet connection)
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained('/path/to/directory/pyannote-speaker-diarization-community-1')

    # run the pipeline locally on your computer
    diarization = pipeline("audio.wav")
    ```

#### Telemetry

With the optional telemetry feature in `pyannote.audio`, you can choose to send anonymous usage metrics to help the `pyannote` team improve the library.

### Breaking changes

- BREAKING(io): remove support for `sox` and `soundfile` audio I/O backends (only `ffmpeg` or in-memory audio is supported)
- BREAKING(setup): drop support to `Python` < 3.10
- BREAKING(hub): rename `use_auth_token` to `token`
- BREAKING(hub): drop support for `{pipeline_name}@{revision}` syntax in `Model.from_pretrained(...)` and `Pipeline.from_pretrained(...)` -- use new `revision` keyword argument instead
- BREAKING(task): remove `OverlappedSpeechDetection` task (part of `SpeakerDiarization` task)
- BREAKING(pipeline): remove `OverlappedSpeechDetection` and `Resegmentation` unmaintained pipelines (part of `SpeakerDiarization`)
- BREAKING(cache): rely on `huggingface_hub` caching directory (`PYANNOTE_CACHE` is no longer used)
- BREAKING(inference): `Inference` now only supports already instantiated models
- BREAKING(task): drop support for `multilabel` training in `SpeakerDiarization` task
- BREAKING(task): drop support for `warm_up` option in `SpeakerDiarization` task
- BREAKING(task): drop support for `weigh_by_cardinality` option in `SpeakerDiarization` task
- BREAKING(task): drop support for `vad_loss` option in `SpeakerDiarization` task
- BREAKING(chore): switch to native namespace package
- BREAKING(cli): remove deprecated `pyannote-audio-train` CLI

### New features

- feat(io): switch from `torchaudio` to `torchcodec` for audio I/O
- feat(pipeline): add support for VBx clustering ([@Selesnyan](https://github.com/Selesnyan) and [jyhan03](https://github.com/jyhan03))
- feat(pyannoteAI): add wrapper around pyannoteAI SDK
- improve(hub): add support for pipeline repos that also include underlying models
- feat(clustering): add support for `k-means` clustering
- feat(model): add `wav2vec_frozen` option to freeze/unfreeze `wav2vec` in `SSeRiouSS` architecture
- feat(task): add support for manual optimization in `SpeakerDiarization` task
- feat(utils): add `hidden` option to `ProgressHook`
- feat(utils): add `FilterByNumberOfSpeakers` protocol files filter
- feat(core): add `Calibration` class to calibrate logits/distances into probabilities
- feat(metric): add `DetectionErrorRate`, `SegmentationErrorRate`, `DiarizationPrecision`, and `DiarizationRecall` metrics
- feat(cli): add CLI to download, apply, benchmark, and optimize pipelines
- feat(cli): add CLI to strip checkpoints to their bare inference minimum 
 
### Improvements

- improve(model): improve WavLM (un)freezing support for `SSeRiouSS` architecture ([@clement-pages](https://github.com/clement-pages/))
- improve(task): improve `SpeakerDiarization` training with manual optimization ([@clement-pages](https://github.com/clement-pages/))
- improve(train): speed up dataloaders
- improve(setup): switch to `uv`
- improve(setup): switch to `lightning` from `pytorch-lightning`
- improve(utils): improve dependency check when loading pretrained models and/or pipeline
- improve(utils): add option to skip dependency check
- improve(utils): add option to load a pretrained model checkpoint from an `io.BytesIO` buffer
- improve(pipeline): add option to load a pretrained pipeline from a `dict` ([@benniekiss](https://github.com/benniekiss/))

### Fixes

- fix(model): improve WavLM (un)freezing support for `ToTaToNet` architecture ([@clement-pages](https://github.com/clement-pages/))
- fix(separation): fix clipping issue in speech separation pipeline ([@joonaskalda](https://github.com/joonaskalda/))
- fix(separation): fix alignment between separated sources and diarization ([@Lebourdais](https://github.com/Lebourdais/) and [@clement-pages](https://github.com/clement-pages/))
- fix(separation): prevent leakage removal collar from being applied to diarization ([@clement-pages](https://github.com/clement-pages/))
- fix(separation): fix `PixIT` training with manual optimization ([@clement-pages](https://github.com/clement-pages/))
- fix(doc): fix link to pytorch ([@emmanuel-ferdman](https://github.com/emmanuel-ferdman/))
- fix(task): fix corner case with small (<9) number of validation samples ([@antoinelaurent](https://github.com/antoinelaurent/))
- fix(doc): fix default embedding in `SpeechSeparation` and `SpeakerDiarization` docstring ([@razi-tm](https://github.com/razi-tm/)).

## Version 3.4.0 (2025-09-09)

- setup: pin pyannote.{core,database,metrics,pipeline} dependencies as future releases of these packages will break the 3.x branch

## Version 3.3.2 (2024-09-11)

### Fixes

- fix: (really) fix support for `numpy==2.x` ([@metal3d](https://github.com/metal3d/))
- doc: fix `Pipeline` docstring ([@huisman](https://github.com/huisman/))

## Version 3.3.1 (2024-06-19)

### Breaking changes

- setup: drop support for Python 3.8

### Fixes

- fix: fix support for `numpy==2.x` ([@ibevers](https://github.com/ibevers/))
- fix: fix support for `speechbrain==1.x` ([@Adel-Moumen](https://github.com/Adel-Moumen/))


## Version 3.3.0 (2024-06-14)

### TL;DR

`pyannote.audio` does [speech separation](https://hf.co/pyannote/speech-separation-ami-1.0): multi-speaker audio in, one audio channel per speaker out!

```bash
pip install pyannote.audio[separation]==3.3.0
```

### New features

- feat(task): add `PixIT` joint speaker diarization and speech separation task (with [@joonaskalda](https://github.com/joonaskalda/))
- feat(model): add `ToTaToNet` joint speaker diarization and speech separation model (with [@joonaskalda](https://github.com/joonaskalda/))
- feat(pipeline): add `SpeechSeparation` pipeline (with [@joonaskalda](https://github.com/joonaskalda/))
- feat(io): add option to select torchaudio `backend`

### Fixes

- fix(task): fix wrong train/development split when training with (some) meta-protocols ([#1709](https://github.com/pyannote/pyannote-audio/issues/1709))
- fix(task): fix metadata preparation with missing validation subset ([@clement-pages](https://github.com/clement-pages/))

### Improvements

- improve(io): when available, default to using `soundfile` backend
- improve(pipeline): do not extract embeddings when `max_speakers` is set to 1
- improve(pipeline): optimize memory usage of most pipelines ([#1713](https://github.com/pyannote/pyannote-audio/pull/1713) by [@benniekiss](https://github.com/benniekiss/))

## Version 3.2.0 (2024-05-08)

### New features

- feat(task): add option to cache task training metadata to speed up training (with [@clement-pages](https://github.com/clement-pages/))
- feat(model): add `receptive_field`, `num_frames` and `dimension` to models (with [@Bilal-Rahou](https://github.com/Bilal-Rahou))
- feat(model): add `fbank_only` property to `WeSpeaker` models
- feat(util): add `Powerset.permutation_mapping` to help with permutation in powerset space (with [@FrenchKrab](https://github.com/FrenchKrab))
- feat(sample): add sample file at `pyannote.audio.sample.SAMPLE_FILE`
- feat(metric): add `reduce` option to `diarization_error_rate` metric (with [@Bilal-Rahou](https://github.com/Bilal-Rahou))
- feat(pipeline): add `Waveform` and `SampleRate` preprocessors

### Fixes

- fix(task): fix random generators and their reproducibility (with [@FrenchKrab](https://github.com/FrenchKrab))
- fix(task): fix estimation of training set size (with [@FrenchKrab](https://github.com/FrenchKrab))
- fix(hook): fix `torch.Tensor` support in `ArtifactHook`
- fix(doc): fix typo in `Powerset` docstring (with [@lukasstorck](https://github.com/lukasstorck))
- fix(doc): remove mention of unsupported `numpy.ndarray` waveform (with [@Purfview](https://github.com/Purfview))

### Improvements

- improve(metric): add support for number of speakers mismatch in `diarization_error_rate` metric
- improve(pipeline): track both `Model` and `nn.Module` attributes in `Pipeline.to(device)`
- improve(io): switch to `torchaudio >= 2.2.0`
- improve(doc): update tutorials (with [@clement-pages](https://github.com/clement-pages/))

### Breaking changes

- BREAKING(model): get rid of `Model.example_output` in favor of `num_frames` method, `receptive_field` property, and `dimension` property
- BREAKING(task): custom tasks need to be updated (see "Add your own task" tutorial)

### Community contributions

- community: add tutorial for offline use of `pyannote/speaker-diarization-3.1` (by [@simonottenhauskenbun](https://github.com/simonottenhauskenbun))

## Version 3.1.1 (2023-12-01)

### TL;DR

Providing `num_speakers` to [`pyannote/speaker-diarization-3.1`](https://hf.co/pyannote/speaker-diarization-3.1) now [works as expected](https://github.com/pyannote/pyannote-audio/issues/1567).

### Fixes

- fix(pipeline): fix support for setting `num_speakers` in [`pyannote/speaker-diarization-3.1`](https://hf.co/pyannote/speaker-diarization-3.1) pipeline

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
