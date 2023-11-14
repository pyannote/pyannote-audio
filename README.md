Using `pyannote.audio` open-source toolkit in production?  
Make the most of it thanks to our [consulting services](https://herve.niderb.fr/consulting.html).

# `pyannote.audio` speaker diarization toolkit

`pyannote.audio` is an open-source toolkit written in Python for speaker diarization. Based on [PyTorch](pytorch.org) machine learning framework, it comes with state-of-the-art [pretrained models and pipelines](https://hf.co/pyannote), that can be further finetuned to your own data for even better performance.

<p align="center">
 <a href="https://www.youtube.com/watch?v=37R_R82lfwA"><img src="https://img.youtube.com/vi/37R_R82lfwA/0.jpg"></a>
</p>


## TL;DR

1. Install [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) `3.0` with `pip install pyannote.audio`
2. Accept [`pyannote/segmentation-3.0`](https://hf.co/pyannote/segmentation-3.0) user conditions
3. Accept [`pyannote/speaker-diarization-3.0`](https://hf.co/pyannote/speaker-diarization-3.0) user conditions
4. Create access token at [`hf.co/settings/tokens`](https://hf.co/settings/tokens).


```python
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE")

# send pipeline to GPU (when available)
import torch
pipeline.to(torch.device("cuda"))

# apply pretrained pipeline
diarization = pipeline("audio.wav")

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
# start=0.2s stop=1.5s speaker_0
# start=1.8s stop=3.9s speaker_1
# start=4.2s stop=5.7s speaker_0
# ...
```

## Highlights

- :hugs: pretrained [pipelines](https://hf.co/models?other=pyannote-audio-pipeline) (and [models](https://hf.co/models?other=pyannote-audio-model)) on [:hugs: model hub](https://huggingface.co/pyannote)
- :exploding_head: state-of-the-art performance (see [Benchmark](#benchmark))
- :snake: Python-first API
- :zap: multi-GPU training with [pytorch-lightning](https://pytorchlightning.ai/)


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
  - [First releaase of pyannote.audio](https://www.youtube.com/watch?v=37R_R82lfwA) / ICASSP 2020 /  8 min

## Benchmark

Out of the box, `pyannote.audio` speaker diarization [pipeline](https://hf.co/pyannote/speaker-diarization-3.0) v3.0 is expected to be much better (and faster) than v2.x.  
Those numbers are diarization error rates (in %):

| Dataset \ Version      | v1.1 | v2.0 | [v2.1](https://hf.co/pyannote/speaker-diarization-2.1) | [v3.0](https://hf.co/pyannote/speaker-diarization-3.0) |  <a href="mailto:herve-at-niderb-dot-fr?subject=Premium pyannote.audio pipeline&body=Looks like I got your attention! Drop me an email for more details. Hervé.">Premium</a>  |
| ---------------------- | ---- | ---- | ------ | ------ | --------- |
| AISHELL-4              | -    | 14.6 |  14.1  |  12.3  | 12.3      |
| AliMeeting (channel 1) | -    | -    |  27.4  |  24.3  | 19.4      |
| AMI (IHM)              | 29.7 | 18.2 |  18.9  |  19.0  | 16.7      |
| AMI (SDM)              | -    | 29.0 |  27.1  |  22.2  | 20.1      |
| AVA-AVD                | -    | -    |  -     |  49.1  | 42.7      |
| DIHARD 3 (full)        | 29.2 | 21.0 |  26.9  |  21.7  | 17.0      |
| MSDWild                | -    | -    |  -     |  24.6  | 20.4      |
| REPERE (phase2)        | -    | 12.6 |   8.2  |   7.8  |  7.8      |
| VoxConverse (v0.3)     | 21.5 | 12.6 |  11.2  |  11.3  |  9.5      |

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
