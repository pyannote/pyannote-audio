# Neural speaker diarization with `pyannote.audio`

`pyannote.audio` is an open-source toolkit written in Python for speaker diarization. Based on [PyTorch](pytorch.org) machine learning framework, it provides a set of trainable end-to-end neural building blocks that can be combined and jointly optimized to build speaker diarization pipelines.

<p align="center">
 <a href="https://www.youtube.com/watch?v=37R_R82lfwA"><img src="https://img.youtube.com/vi/37R_R82lfwA/0.jpg"></a>
</p>


## TL;DR [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyannote/pyannote-audio/blob/develop/tutorials/intro.ipynb)


```python
# 1. visit hf.co/pyannote/speaker-diarization and hf.co/pyannote/segmentation and accept user conditions (only if requested)
# 2. visit hf.co/settings/tokens to create an access token (only if you had to go through 1.)
# 3. instantiate pretrained speaker diarization pipeline
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token="ACCESS_TOKEN_GOES_HERE")

# 4. apply pretrained pipeline
diarization = pipeline("audio.wav")

# 5. print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
# start=0.2s stop=1.5s speaker_0
# start=1.8s stop=3.9s speaker_1
# start=4.2s stop=5.7s speaker_0
# ...
```

## What's new in `pyannote.audio` 2.x?

For version 2.x of `pyannote.audio`, [I](https://herve.niderb.fr) decided to rewrite almost everything from scratch.
Highlights of this release are:

- :exploding_head: much better performance (see [Benchmark](#benchmark))
- :snake: Python-first API
- :hugs: pretrained [pipelines](https://hf.co/models?other=pyannote-audio-pipeline) (and [models](https://hf.co/models?other=pyannote-audio-model)) on [:hugs: model hub](https://huggingface.co/pyannote)
- :zap: multi-GPU training with [pytorch-lightning](https://pytorchlightning.ai/)
- :control_knobs: data augmentation with [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)
- :boom: [Prodigy](https://prodi.gy/) recipes for model-assisted audio annotation

## Installation

Only Python 3.8+ is officially supported (though it might work with Python 3.7)

```bash
conda create -n pyannote python=3.8
conda activate pyannote

# pytorch 1.11 is required for speechbrain compatibility
# (see https://pytorch.org/get-started/previous-versions/#v1110)
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch

pip install pyannote.audio
```

## Documentation

- [Changelog](CHANGELOG.md)
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
- Miscellaneous
    - [Training with `pyannote-audio-train` command line tool](tutorials/training_with_cli.md)
    - [Annotating your own data with Prodigy](tutorials/prodigy.md)
    - [Speaker verification](tutorials/speaker_verification.ipynb)
    - Visualization and debugging

## Frequently asked questions

* [How does one capitalize and pronounce the name of this awesome library?](FAQ.md)
* [Can I use gated models (and pipelines) offline?](FAQ.md)
* [Pretrained pipelines do not produce good results on my data. What can I do?](FAQ.md)

## Benchmark

Out of the box, `pyannote.audio` default speaker diarization [pipeline](https://hf.co/pyannote/speaker-diarization) is expected to be much better (and faster) in v2.x than in v1.1. Those numbers are diarization error rates (in %)

| Dataset \ Version      | v1.1 | v2.0 | v2.1.1 (finetuned) |
| ---------------------- | ---- | ---- | ------------------ |
| AISHELL-4              | -    | 14.6 | 14.1 (14.5)        |
| AliMeeting (channel 1) | -    | -    | 27.4 (23.8)        |
| AMI (IHM)              | 29.7 | 18.2 | 18.9 (18.5)        |
| AMI (SDM)              | -    | 29.0 | 27.1 (22.2)        |
| CALLHOME (part2)       | -    | 30.2 | 32.4 (29.3)        |
| DIHARD 3 (full)        | 29.2 | 21.0 | 26.9 (21.9)        |
| VoxConverse (v0.3)     | 21.5 | 12.6 | 11.2 (10.7)        |
| REPERE (phase2)        | -    | 12.6 | 8.2 ( 8.3)         |
| This American Life     | -    | -    | 20.8 (15.2)        |

## Citations

If you use `pyannote.audio` please use the following citations:

```bibtex
@inproceedings{Bredin2020,
  Title = {{pyannote.audio: neural building blocks for speaker diarization}},
  Author = {{Bredin}, Herv{\'e} and {Yin}, Ruiqing and {Coria}, Juan Manuel and {Gelly}, Gregory and {Korshunov}, Pavel and {Lavechin}, Marvin and {Fustes}, Diego and {Titeux}, Hadrien and {Bouaziz}, Wassim and {Gill}, Marie-Philippe},
  Booktitle = {ICASSP 2020, IEEE International Conference on Acoustics, Speech, and Signal Processing},
  Year = {2020},
}
```

```bibtex
@inproceedings{Bredin2021,
  Title = {{End-to-end speaker segmentation for overlap-aware resegmentation}},
  Author = {{Bredin}, Herv{\'e} and {Laurent}, Antoine},
  Booktitle = {Proc. Interspeech 2021},
  Year = {2021},
}
```

## Support

For commercial enquiries and scientific consulting, please contact [me](mailto:herve@niderb.fr).

## Development

The commands below will setup pre-commit hooks and packages needed for developing the `pyannote.audio` library.

```bash
pip install -e .[dev,testing]
pre-commit install
```

Tests rely on a set of debugging files available in [`test/data`](test/data) directory.
Set `PYANNOTE_DATABASE_CONFIG` environment variable to `test/data/database.yml` before running tests:

```bash
PYANNOTE_DATABASE_CONFIG=tests/data/database.yml pytest
```
