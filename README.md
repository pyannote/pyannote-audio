# pyannote-audio

Audio processing

## Installation

```bash
$ conda create --name pyannote python=3.6 anaconda
$ source activate pyannote
$ conda install -c conda-forge yaafe
$ pip install -U pip setuptools
$ pip install pyannote.audio
```

[install pytorch from source](https://github.com/pytorch/pytorch#from-source)

## Citation

If you use `pyannote.audio` in your research, please use the following citation (until a regular paper is published):

```bibtex
@misc{pyannote.audio,
  author = {Bredin, H.},
  title = {pyannote.audio},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/pyannote/pyannote-audio}},
}
```

## Tutorials

 * [Feature extraction](tutorials/feature-extraction)
 * [LSTM-based speech activity detection](tutorials/speech-activity-detection)
 * [LSTM-based speaker change detection](tutorials/change-detection)
 * [_TristouNet_ neural speech turn embedding](tutorials/speaker-embedding)


## Documentation

The API is unfortunately not documented yet.
