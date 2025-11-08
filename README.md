<p align="center">
  <a href="https://pyannote.ai/" target="blank"><img src="https://avatars.githubusercontent.com/u/162698670" width="64" /></a>
</p>

<div align="center">
    <h1><code>pyannote</code> speaker diarization toolkit</h1>
</div>


`pyannote.audio` is an open-source toolkit written in Python for speaker diarization. Based on [PyTorch](https://pytorch.org) machine learning framework, it comes with state-of-the-art [pretrained models and pipelines](https://hf.co/pyannote), that can be further finetuned to your own data for even better performance.

<p align="center">
 <a href="https://www.youtube.com/watch?v=37R_R82lfwA"><img src="https://img.youtube.com/vi/37R_R82lfwA/0.jpg"></a>
</p>

## Highlights

- :exploding_head: state-of-the-art performance (see [Benchmark](#benchmark))
- :hugs: pretrained [pipelines](https://hf.co/models?other=pyannote-audio-pipeline) (and [models](https://hf.co/models?other=pyannote-audio-model)) on [:hugs: model hub](https://huggingface.co/pyannote)
- :rocket: built-in support for [pyannoteAI](https://pyannote.ai) premium speaker diarization
- :snake: Python-first API
- :zap: multi-GPU training with [pytorch-lightning](https://pytorchlightning.ai/)

## `community-1` open-source speaker diarization

1. Make sure [`ffmpeg`](https://ffmpeg.org/) is installed on your machine (needed by [`torchcodec`](https://docs.pytorch.org/torchcodec/) audio decoding library)
2. Install with [`uv`](https://docs.astral.sh/uv/)`add pyannote.audio` (recommended) or `pip install pyannote.audio`
3. Accept [`pyannote/speaker-diarization-community-1`](https://hf.co/pyannote/speaker-diarization-community-1) user conditions
4. Create Huggingface access token at [`hf.co/settings/tokens`](https://hf.co/settings/tokens)

```python
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

# Community-1 open-source speaker diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token="HUGGINGFACE_ACCESS_TOKEN")

# send pipeline to GPU (when available)
pipeline.to(torch.device("cuda"))

# apply pretrained pipeline (with optional progress hook)
with ProgressHook() as hook:
    output = pipeline("audio.wav", hook=hook)  # runs locally

# print the result
for turn, speaker in output.speaker_diarization:
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
# start=0.2s stop=1.5s speaker_0
# start=1.8s stop=3.9s speaker_1
# start=4.2s stop=5.7s speaker_0
# ...

```

## `precision-2` premium speaker diarization

1. Create pyannoteAI API key at [`dashboard.pyannote.ai`](https://dashboard.pyannote.ai) 
2. Enjoy free credits!

```python
from pyannote.audio import Pipeline

# Precision-2 premium speaker diarization service
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-precision-2", token="PYANNOTEAI_API_KEY")

output = pipeline("audio.wav")  # runs on pyannoteAI servers

# print the result
for turn, speaker in output.speaker_diarization:
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s {speaker}")
# start=0.2s stop=1.6s SPEAKER_00
# start=1.8s stop=4.0s SPEAKER_01 
# start=4.2s stop=5.6s SPEAKER_00
# ...
```

Visit [`docs.pyannote.ai`](https://docs.pyannote.ai) to learn about other pyannoteAI features (voiceprinting, confidence scores, ...)

## Benchmark

| Benchmark (last updated in 2025-09) | <a href="https://hf.co/pyannote/speaker-diarization-3.1">`legacy` (3.1)</a>| <a href="https://hf.co/pyannote/speaker-diarization-community-1">`community-1`</a> | <a href="https://docs.pyannote.ai">`precision-2`</a> | 
| --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | -------------------------------------------------| ------------------------------------------------ |
| [AISHELL-4](https://arxiv.org/abs/2104.03603)                                                                               | 12.2 | 11.7 | 11.4 |
| [AliMeeting](https://www.openslr.org/119/) (channel 1)                                                                      | 24.5 | 20.3 | 15.2 |
| [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) (IHM)                                                                        | 18.8 | 17.0 | 12.9 |
| [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) (SDM)                                                                        | 22.7 | 19.9 | 15.6 |
| [AVA-AVD](https://arxiv.org/abs/2111.14448)                                                                                 | 49.7 | 44.6 | 37.1 |
| [CALLHOME](https://catalog.ldc.upenn.edu/LDC2001S97) ([part 2](https://github.com/BUTSpeechFIT/CALLHOME_sublists/issues/1)) | 28.5 | 26.7 | 16.6 |
| [DIHARD 3](https://catalog.ldc.upenn.edu/LDC2022S14) ([full](https://arxiv.org/abs/2012.01477))                             | 21.4 | 20.2 | 14.7 |
| [Ego4D](https://arxiv.org/abs/2110.07058) (dev.)                                                                            | 51.2 | 46.8 | 39.0 |
| [MSDWild](https://github.com/X-LANCE/MSDWILD)                                                                               | 25.4 | 22.8 | 17.3 |
| [RAMC](https://www.openslr.org/123/)                                                                                        | 22.2 | 20.8 | 10.5 |
| [REPERE](https://www.islrn.org/resources/360-758-359-485-0/) (phase2)                                                       | 7.9  |  8.9 |  7.4 |
| [VoxConverse](https://github.com/joonson/voxconverse) (v0.3)                                                                | 11.2 | 11.2 |  8.5 |

__[Diarization error rate](http://pyannote.github.io/pyannote-metrics/reference.html#diarization) (in %, the lower, the better)__

Compared to the [`3.1`](https://hf.co/pyannote/speaker-diarization-3.1) legacy pipeline, [`community-1`](https://hf.co/pyannote/speaker-diarization-community-1) brings significant improvement in terms of speaker counting and assignment.
[`precision-2`](https://www.pyannote.ai/blog/precision-2) premium pipeline further improves accuracy as well as processing speed (in its self-hosted version).

| Benchmark (last updated in 2025-09) | <a href="https://hf.co/pyannote/speaker-diarization-community-1">`community-1`</a> | <a href="https://docs.pyannote.ai">`precision-2`</a> | Speed up
| -------------- | ----------- | ----------- | ------ |
| [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) (IHM), ~1h files                                                     | 31s per hour of audio | 14s per hour of audio | 2.2x faster
| [DIHARD 3](https://catalog.ldc.upenn.edu/LDC2022S14) ([full](https://arxiv.org/abs/2012.01477)), ~5min files        | 37s per hour of audio | 14s per hour of audio | 2.6x faster

__Self-hosted speed on a NVIDIA H100 80GB HBM3__

## Telemetry

With the optional telemetry feature in `pyannote.audio`, you can choose to send anonymous usage metrics to help the `pyannote` team improve the library.

### What we track

For each call to `Pipeline.from_pretrained({origin})` (or `Model.from_pretrained({origin})`), we track information about `{origin}` in the following privacy-preserving way:

* If `{origin}` is an official `pyannote` or `pyannoteAI` pipeline (or model) hosted on `Huggingface`, we track it as `{origin}`.
* If `{origin}` is a pipeline (or model) hosted on `Huggingface` from any other organization, we track it as `huggingface`.
* If `{origin}` is a path to a local file or directory, we track it as `local`.

We also track the pipeline Python class (e.g. `pyannote.audio.pipelines.SpeakerDiarization`).

For each file processed with a pipeline, we track
* the file duration in seconds
* the value of `num_speakers`, `min_speakers`, and `max_speakers` for speaker diarization pipelines

**We [do not track](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/telemetry/metrics.py) any information that could identify who the user is.**

### Configuring telemetry

Telemetry can be configured in three ways:
1. Using an environment variable
2. Within the current Python session only
3. Globally across sessions

All of these options will modify the value of the environment variable for consistency.
If the environment variable is not set, `pyannote.audio` will read the default value in the telemetry config.
The default config can also be changed from Python.

#### Using environment variable

You can control telemetry by setting the `PYANNOTE_METRICS_ENABLED` environment variable:

```bash
# enable metrics
export PYANNOTE_METRICS_ENABLED=1

# disable metrics
export PYANNOTE_METRICS_ENABLED=0
```

#### For current session

To control telemetry for your current Python kernel session:

```python
from pyannote.audio.telemetry import set_telemetry_metrics

# enable metrics for current session
set_telemetry_metrics(True)

# disable metrics for current session
set_telemetry_metrics(False)
```

#### Global configuration

To set telemetry preferences that persist across sessions:

```python
from pyannote.audio.telemetry import set_telemetry_metrics

# enable metrics globally
set_telemetry_metrics(True, save_choice_as_default=True)

# disable metrics globally
set_telemetry_metrics(False, save_choice_as_default=True)
```

## Documentation

- [Changelog](CHANGELOG.md)
- Videos
  - [Speaker diarization, a ~~love~~ loss story](https://www.youtube.com/watch?v=CtjDotATEI0&list=PLSeS0sl8xpTwz7h5iJSniiF89iUdZXNJ2&index=13) / JSALT 2025 plenary talk / 60 min 
  - [Introduction to speaker diarization](https://umotion.univ-lemans.fr/video/9513-speech-segmentation-and-speaker-diarization/) / JSALT 2023 summer school / 90 min
  - [Speaker segmentation model](https://www.youtube.com/watch?v=wDH2rvkjymY) / Interspeech 2021 / 3 min
  - [First release of pyannote.audio](https://www.youtube.com/watch?v=37R_R82lfwA) / ICASSP 2020 / 8 min
- Blog
  - 2022-12-02 > ["How I reached 1st place at Ego4D 2022, 1st place at Albayzin 2022, and 6th place at VoxSRC 2022 speaker diarization challenges"](tutorials/adapting_pretrained_pipeline.ipynb)
  - 2022-10-23 > ["One speaker segmentation model to rule them all"](https://herve.niderb.fr/fastpages/2022/10/23/One-speaker-segmentation-model-to-rule-them-all)
  - 2021-08-05 > ["Streaming voice activity detection with pyannote.audio"](https://herve.niderb.fr/fastpages/2021/08/05/Streaming-voice-activity-detection-with-pyannote.html)
- Community contributions (not maintained by the core team)
  - 2024-04-05 > [Offline speaker diarization (speaker-diarization-3.1)](tutorials/community/offline_usage_speaker_diarization.ipynb) by [Simon Ottenhaus](https://github.com/simonottenhauskenbun)
  - 2024-09-24 > [Evaluating `pyannote` pretrained speech separation pipelines](tutorials/community/eval_separation_pipeline.ipynb) by  [Clément Pagés](https://github.com/)
- Tutorials  
*Those tutorials were written for older versions of pyannote.audio and should be updated. Interested in working for pyannoteAI as a community manager or developer advocate? This might be a nice place to start!*
  - [Applying a pretrained pipeline](tutorials/applying_a_pipeline.ipynb)
  - [Adapting a pretrained pipeline to your own data](tutorials/adapting_pretrained_pipeline.ipynb)
  - [Training a pipeline](tutorials/voice_activity_detection.ipynb)
  - [Applying a pretrained model](tutorials/applying_a_model.ipynb)
  - [Training, fine-tuning, and transfer learning](tutorials/training_a_model.ipynb)
  - [Adding a new model](tutorials/add_your_own_model.ipynb)
  - [Adding a new task](tutorials/add_your_own_task.ipynb)
  - [Frequently asked questions](FAQ.md)

## Citations

If you use `pyannote.audio` please use the following citations:

```bibtex
@inproceedings{Plaquet23,
  author={Alexis Plaquet and Hervé Bredin},
  title={{Powerset multi-class cross entropy loss for neural speaker diarization}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}
```

```bibtex
@inproceedings{Bredin23,
  author={Hervé Bredin},
  title={{pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}
```

## Development

The commands below will setup pre-commit hooks and packages needed for developing the `pyannote.audio` library.

```bash
pip install -e .[dev,testing]
pre-commit install
```

## Test

```bash
pytest
```
