# Neural speaker diarization with `pyannote.audio`

`pyannote.audio` is an open-source toolkit written in Python for speaker diarization. Based on [PyTorch](pytorch.org) machine learning framework, it provides a set of trainable end-to-end neural building blocks that can be combined and jointly optimized to build speaker diarization pipelines.

<p align="center">
 <a href="https://www.youtube.com/watch?v=37R_R82lfwA"><img src="https://img.youtube.com/vi/37R_R82lfwA/0.jpg"></a>
</p>


## TL;DR [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyannote/pyannote-audio/blob/develop/tutorials/intro.ipynb)


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

- Models
    - Available tasks explained
    - [Applying a pretrained model](tutorials/applying_a_model.ipynb)
    - [Training, fine-tuning, and transfer learning](tutorials/training_a_model.ipynb)
- Pipelines
    - Available pipelines explained
    - [Applying a pretrained pipeline](tutorials/applying_a_pipeline.ipynb)
    - [Training a pipeline](tutorials/voice_activity_detection.ipynb)
- Contributing
    - [Adding a new model](tutorials/add_your_own_model.ipynb)
    - [Adding a new task](tutorials/add_your_own_task.ipynb)
    - Adding a new pipeline
    - Sharing pretrained models and pipelines
- Miscellaneous
    - [Training with `pyannote-audio-train` command line tool](tutorials/training_with_cli.md)
    - [Annotating your own data with Prodigy](tutorials/prodigy.md)
    - [Speaker verification](tutorials/speaker_verification.ipynb)
    - Visualization and debugging

## Frequently asked questions

#### How does one capitalize and pronounce the name of this awesome library?

📝 Written in lower case: `pyannote.audio` (or `pyannote` if you are lazy).  Not `PyAnnote` nor `PyAnnotate` (*sic*).
📢 [Pronounced](https://www.howtopronounce.com/french/pianote) like the french verb *pianoter*.  *pi* like in **pi**ano, not *py* like in **py**thon.
🎹 *pianoter* means *to play the piano* (hence the logo 🤯).

#### **[Pretrained pipelines](https://huggingface.co/models?other=pyannote-audio-pipeline) do not produce good results on my data. What can I do?**

1. [Annotate](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/prodigy.md) dozens of conversations manually and separate them into development and test subsets in [`pyannote.database`](https://github.com/pyannote/pyannote-database#speaker-diarization).
2. [Optimize the hyper-parameters](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/voice_activity_detection.ipynb) of the pretained pipeline using the development set. If performance is still not good enough, go to step 3.
3. Annotate hundreds of conversations manually and set them up as training subset in `pyannote.database`.
4. [Fine-tune](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/training_a_model.ipynb) the models (on which the pipeline relies) using the training set.
5. [Optimize the hyper-parameters](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/voice_activity_detection.ipynb) of the pipeline using the fine-tuned models using the development set. If performance is still not good enough, go back to step 3.


## Benchmark

Out of the box, `pyannote.audio` default speaker diarization pipeline is expected to be much better (and faster) in v2.0 than in v1.1.:

| Dataset     | DER% with v1.1 | DER% with v2.0 | Relative improvement |
| ----------- | -------------- | -------------- | -------------------- |
| AMI         | 29.7%          | 18.2%          | 38%                  |
| DIHARD      | 29.2%          | 21.0%          | 28%                  |
| VoxConverse | 21.5%          | 12.6%          | 41%                  |

A more detailed benchmark is available [here](https://hf.co/pyannote/speaker-diarization).

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
