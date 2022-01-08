# Neural speaker diarization with `pyannote.audio`

`pyannote.audio` is an open-source toolkit written in Python for speaker diarization. Based on [PyTorch](pytorch.org) machine learning framework, it provides a set of trainable end-to-end neural building blocks that can be combined and jointly optimized to build speaker diarization pipelines.

<p align="center">
 <a href="https://www.youtube.com/watch?v=37R_R82lfwA"><img src="https://img.youtube.com/vi/37R_R82lfwA/0.jpg"></a>
</p>


## TL;DR

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyannote/pyannote-audio/blob/master/notebooks/introduction_to_pyannote_audio_speaker_diarization_toolkit.ipynb)


```python
# instantiate pretrained speaker diarization pipeline
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# apply pretrained pipeline
diarization = pipeline("audio.wav")

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
# start=0.2s stop=1.5s speaker_A
# start=1.8s stop=3.9s speaker_B
# start=4.2s stop=5.7s speaker_A
# ...
```

## What's new in `pyannote.audio` 2.0

For version 2.x of `pyannote.audio`, [I](https://herve.niderb.fr) decided to rewrite almost everything from scratch.
Highlights of this release are:

- :exploding_head: much better performance (see [Benchmark](#benchmark))
- :snake: Python-first API
- :hugs: pretrained [pipelines](https://hf.co/models?other=pyannote-audio-pipeline) (and [models](https://hf.co/models?other=pyannote-audio-model)) on [:hugs: model hub](https://huggingface.co/pyannote)
- :zap: multi-GPU training with [pytorch-lightning](https://pytorchlightning.ai/)
- :control_knobs: data augmentation with [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)

## Installation

```bash
conda create -n pyannote python=3.8.5
conda activate pyannote
conda install pytorch torchaudio -c pytorch

# pyannote.audio relies on torchaudio's soundfile backend,
# itself relying on libsndfile, sometimes tricky to install.
# This seems to work fine but is provided with no guarantee of success:
conda install numpy cffi
conda install libsndfile=1.0.28 -c conda-forge

pip install https://github.com/pyannote/pyannote-audio/archive/develop.zip
```

## Documentation

- Pipelines
    - [Applying a pretrained pipeline](tutorials/applying_a_pipeline.ipynb)
    - Pipelines illustrated
    - Tuning a pretrained pipeline
- Models
    - [Applying a pretrained model](tutorials/applying_a_model.ipynb)
    - Available tasks explained
    - [Training, fine-tuning, and transfer learning](tutorials/training_a_model.ipynb)
- Contributing
    - [Adding a new model](tutorials/add_your_own_model.ipynb)
    - [Adding a new task](tutorials/add_your_own_task.ipynb)
    - Adding a new pipeline
    - Sharing pretrained models and pipelines
- Miscellaneous
    - [Speaker verification](tutorials/speaker_verification.ipynb)
    - Visualization and debugging

## Benchmark

The pretrained speaker diarization pipeline with default parameters is expected to be much better in v2.0 than in v1.1:

| [Diarization error rate](http://pyannote.github.io/pyannote-metrics/reference.html#diarization) (%) | v1.1 | v2.0 | ∆DER |
| --------------------------------------------------------------------------------------------------- | ---- | ---- | ---- |
| [AMI `only_words` evaluation set](https://github.com/BUTSpeechFIT/AMI-diarization-setup)            | 29.7 | 21.3 | -28% |
| [DIHARD 3 evaluation set](https://arxiv.org/abs/2012.01477)                                         | 29.2 | 22.2 | -24% |
| [VoxConverse 0.0.2 evaluation set](https://github.com/joonson/voxconverse)                          | 21.5 | 13.0 | -39% |

Here is the (pseudo-)code used to obtain those numbers:

```python
# v1.1
import torch
pipeline = torch.hub.load("pyannote/pyannote-audio", "dia")
diarization = pipeline({"audio": "audio.wav"})

# v2.0
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
diarization = pipeline("audio.wav")

# evaluation
from pyannote.metrics.diarization import DiarizationErrorRate
metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
for audio, reference in evaluation_set:  # pseudo-code
    diarization = pipeline(audio)
    _ = metric(reference, diarization)
der = abs(metric)
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
