# `pyannote-audio` | neural building blocks for speaker diarization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyannote/pyannote-audio/blob/develop/notebooks/introduction_to_pyannote_audio_speaker_diarization_toolkit.ipynb)

`pyannote.audio` is an open-source toolkit written in Python for speaker diarization. Based on [PyTorch](pytorch.org) machine learning framework, it provides a set of trainable end-to-end neural building blocks that can be combined and jointly optimized to build speaker diarization pipelines:

<p align="center"> 
<img src="pipeline.png">
</p>

`pyannote.audio` also comes with [pretrained models](https://github.com/pyannote/pyannote-audio-hub) covering a wide range of domains for voice activity detection, speaker change detection, overlapped speech detection, and speaker embedding:

![segmentation](tutorials/pretrained/model/segmentation.png)

## Installation

`pyannote.audio` only supports Python 3.7 (or later) on Linux and macOS. It might work on Windows but there is no garantee that it does, nor any plan to add official support for Windows.

The instructions below assume that `pytorch` has been installed using the instructions from https://pytorch.org.

Until a proper release of `pyannote.audio` is available on `PyPI`, it must be installed from source using the develop branch of the official repository:

```bash
$ git clone https://github.com/pyannote/pyannote-audio.git
$ cd pyannote-audio
$ git checkout develop
$ pip install .
```

## Documentation

Part of the API is described in [this](tutorials/pretrained/model) tutorial.  

Documentation is a work in progress and is scheduled to be ready by end of April 2020.

## Tutorials

* Use [pretrained](https://github.com/pyannote/pyannote-audio-hub) models and pipelines
  * [Apply pretrained pipelines on your own data](tutorials/pretrained/pipeline)
  * [Apply pretrained models on your own data](tutorials/pretrained/model)
* [Prepare your own dataset for training or fine-tuning](tutorials/data_preparation)
* [Fine-tune pretrained models to your own data](tutorials/finetune)
* Train models on your own data
  * [Speech activity detection](tutorials/models/speech_activity_detection)
  * [Speaker change detection](tutorials/models/speaker_change_detection)
  * [Overlapped speech detection](tutorials/models/overlap_detection)
  * [Speaker embedding](tutorials/models/speaker_embedding)
* Tune pipelines on your own data
  * [Speech activity detection pipeline](tutorials/pipelines/speech_activity_detection)
  * [Speaker diarization pipeline](tutorials/pipelines/speaker_diarization)

## Citation

If you use `pyannote.audio` please use the following citation

```bibtex
@inproceedings{Bredin2020,
  Title = {{pyannote.audio: neural building blocks for speaker diarization}},
  Author = {{Bredin}, Herv{\'e} and {Yin}, Ruiqing and {Coria}, Juan Manuel and {Gelly}, Gregory and {Korshunov}, Pavel and {Lavechin}, Marvin and {Fustes}, Diego and {Titeux}, Hadrien and {Bouaziz}, Wassim and {Gill}, Marie-Philippe},
  Booktitle = {ICASSP 2020, IEEE International Conference on Acoustics, Speech, and Signal Processing},
  Address = {Barcelona, Spain},
  Month = {May},
  Year = {2020},
}
```
