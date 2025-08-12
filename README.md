Using `pyannote.audio` open-source toolkit in production?
Consider switching to [pyannoteAI](https://www.pyannote.ai) for better and faster options.

# `pyannote` speaker diarization toolkit

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

## Open-source speaker diarization pipeline

1. Install [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) with `pip install pyannote.audio`
2. Accept [`pyannote/segmentation-3.0`](https://hf.co/pyannote/segmentation-3.0) user conditions
3. Accept [`pyannote/speaker-diarization-3.1`](https://hf.co/pyannote/speaker-diarization-3.1) user conditions
4. Create Huggingface access token at [`hf.co/settings/tokens`](https://hf.co/settings/tokens).

```python
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

# Open-source pyannote speaker diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token="HUGGINGFACE_ACCESS_TOKEN")

# send pipeline to GPU (when available)
pipeline.to(torch.device("cuda"))

# apply pretrained pipeline (with optional progress hook)
with ProgressHook() as hook:
    diarization = pipeline("audio.wav", hook=hook)  # runs locally

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
# start=0.2s stop=1.5s speaker_0
# start=1.8s stop=3.9s speaker_1
# start=4.2s stop=5.7s speaker_0
# ...

```

## Premium pyannoteAI speaker diarization pipeline

1. Install [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) with `pip install pyannote.audio`
2. Create pyannoteAI API key at [`dashboard.pyannote.ai`](https://dashboard.pyannote.ai)

```python
from pyannote.audio import Pipeline

# Premium pyannoteAI speaker diarization service
pipeline = Pipeline.from_pretrained(
    "pyannoteAI/speaker-diarization-precision", token="PYANNOTEAI_API_KEY")

diarization = pipeline("audio.wav")  # runs on pyannoteAI servers

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s {speaker}")
# start=0.2s stop=1.6s SPEAKER_00
# start=1.8s stop=4.0s SPEAKER_01 
# start=4.2s stop=5.6s SPEAKER_00
# ...
```

Visit [`docs.pyannote.ai`](https://docs.pyannote.ai) to learn about other pyannoteAI features (voiceprinting, confidence scores, ...)

## Benchmark
Out of the box, <img src="https://avatars.githubusercontent.com/u/7559051" width="20" style="vertical-align:text-bottom;" /> `pyannote.audio` speaker diarization [pipeline v4.0](https://hf.co/pyannote/speaker-diarization-4.0) is expected to be much better than v3.1.

<img src="https://avatars.githubusercontent.com/u/162698670" width="20" style="vertical-align:text-bottom;" /> `pyannoteAI` premium model are even better (and faster) and can be testedby creating an account on [`pyannoteAI` website](https://dashboard.pyannote.ai) (you will get 150 hours of free credits). 

| Benchmark (2025-08) | <a href="https://hf.co/pyannote/speaker-diarization-3.1"><img src="https://avatars.githubusercontent.com/u/7559051" width="32" /><br/>v3.1</a> | <a href="https://hf.co/pyannote/speaker-diarization-4.0"><img src="https://avatars.githubusercontent.com/u/7559051" width="32" /><br/> v4.0</a> | <a href="https://docs.pyannote.ai"><img src="https://avatars.githubusercontent.com/u/162698670" width="32" /><br/>API</a> | <a href="https://docs.pyannote.ai"><img src="https://avatars.githubusercontent.com/u/162698670" width="32" /><br/>labs</a> | 
| --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | -------------------------------------------------| ------------------------------------------------ | --- |
| [AISHELL-4](https://arxiv.org/abs/2104.03603)                                                                               | 12.2 | 11.6 | 11.9 | 11.4 | 
| [AliMeeting](https://www.openslr.org/119/) (channel 1)                                                                      | 24.5 | 20.7 | 16.6 | 15.2 | 
| [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) (IHM)                                                                        | 18.8 | 17.4 | 13.2 | 12.9 |
| [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) (SDM)                                                                        | 22.7 | 19.8 | 15.8 | 15.6 |
| [AVA-AVD](https://arxiv.org/abs/2111.14448)                                                                                 | 49.7 | 45.3 | 39.9 | 37.1 |
| [CALLHOME](https://catalog.ldc.upenn.edu/LDC2001S97) ([part 2](https://github.com/BUTSpeechFIT/CALLHOME_sublists/issues/1)) | 28.4 | 27.5 | 17.8 | 16.6 |
| [DIHARD 3](https://catalog.ldc.upenn.edu/LDC2022S14) ([full](https://arxiv.org/abs/2012.01477))                             | 21.4 | 20.5 | 15.7 | 14.7 |
| [Ego4D](https://arxiv.org/abs/2110.07058) (dev.)                                                                            | 51.2 | 46.5 | 42.8 | 39.0 |
| [MSDWild](https://github.com/X-LANCE/MSDWILD)                                                                               | 25.4 | 23.0 | 17.8 | 17.3 |
| [RAMC](https://www.openslr.org/123/)                                                                                        | 22.2 | 20.9 | 10.6 | 10.5 |
| [REPERE](https://www.islrn.org/resources/360-758-359-485-0/) (phase2)                                                       | 7.9  |  8.9 |  7.3 |  7.4 |
| [VoxConverse](https://github.com/joonson/voxconverse) (v0.3)                                                                | 11.2 | 11.2 |  8.9 |  8.5 |

[Diarization error rate](http://pyannote.github.io/pyannote-metrics/reference.html#diarization) (in %, the lower, the better)


## Documentation

- [Changelog](CHANGELOG.md)
- [Frequently asked questions](FAQ.md)
- Models
  - Available tasks explained
  - [Applying a pretrained model](tutorials/applying_a_model.ipynb)
  - [Training, fine-tuning, and transfer learning](tutorials/training_a_model.ipynb)
- Pipelines
  - Available pipelines explained
  - [Applying a pretrained pipeline](tutorials/applying_a_pipeline.ipynb)
  - [Adapting a pretrained pipeline to your own data](tutorials/adapting_pretrained_pipeline.ipynb)
  - [Training a pipeline](tutorials/voice_activity_detection.ipynb)
- Contributing
  - [Adding a new model](tutorials/add_your_own_model.ipynb)
  - [Adding a new task](tutorials/add_your_own_task.ipynb)
  - Adding a new pipeline
  - Sharing pretrained models and pipelines
- Blog
  - 2022-12-02 > ["How I reached 1st place at Ego4D 2022, 1st place at Albayzin 2022, and 6th place at VoxSRC 2022 speaker diarization challenges"](tutorials/adapting_pretrained_pipeline.ipynb)
  - 2022-10-23 > ["One speaker segmentation model to rule them all"](https://herve.niderb.fr/fastpages/2022/10/23/One-speaker-segmentation-model-to-rule-them-all)
  - 2021-08-05 > ["Streaming voice activity detection with pyannote.audio"](https://herve.niderb.fr/fastpages/2021/08/05/Streaming-voice-activity-detection-with-pyannote.html)
- Videos
  - [Introduction to speaker diarization](https://umotion.univ-lemans.fr/video/9513-speech-segmentation-and-speaker-diarization/) / JSALT 2023 summer school / 90 min
  - [Speaker segmentation model](https://www.youtube.com/watch?v=wDH2rvkjymY) / Interspeech 2021 / 3 min
  - [First release of pyannote.audio](https://www.youtube.com/watch?v=37R_R82lfwA) / ICASSP 2020 / 8 min
- Community contributions (not maintained by the core team)
  - 2024-04-05 > [Offline speaker diarization (speaker-diarization-3.1)](tutorials/community/offline_usage_speaker_diarization.ipynb) by [Simon Ottenhaus](https://github.com/simonottenhauskenbun)
  - 2024-09-24 > [Evaluating `pyannote` pretrained speech separation pipelines](tutorials/community/eval_separation_pipeline.ipynb) by  [Clément Pagés](https://github.com/)

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
