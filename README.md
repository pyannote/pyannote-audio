# pyannote-audio

Neural building blocks for speaker diarization

## Installation

```bash
$ conda create --name pyannote python=3.6 anaconda
$ source activate pyannote
$ conda install -c conda-forge yaafe
$ conda install cmake
$ pip install -U pip setuptools
$ pip install --process-dependency-links pyannote.audio
```

## Citation

If you use `pyannote.audio` in your research, please use the following citations.

  - Speech  activity and speaker change detection
    ```bibtex
    @inproceedings{Yin2017,
      Author = {Ruiqing Yin and Herv\'e Bredin and Claude Barras},
      Title = {{Speaker Change Detection in Broadcast TV using Bidirectional Long Short-Term Memory Networks}},
      Booktitle = {{18th Annual Conference of the International Speech Communication Association, Interspeech 2017}},
      Year = {2017},
      Month = {August},
      Address = {Stockholm, Sweden},
      Url = {https://github.com/yinruiqing/change_detection}
    }
    ```
  - Speaker embedding
    ```bibtex
    @inproceedings{Bredin2017,
        author = {Herv\'{e} Bredin},
        title = {{TristouNet: Triplet Loss for Speaker Turn Embedding}},
        booktitle = {42nd IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2017},
        year = {2017},
        url = {http://arxiv.org/abs/1609.04301},
    }
    ```

## Tutorials

 * [Feature extraction](tutorials/feature-extraction)
 * [LSTM-based speech activity detection](tutorials/speech-activity-detection)
 * [LSTM-based speaker change detection](tutorials/change-detection)
 * [_TristouNet_ neural speech turn embedding](tutorials/speaker-embedding)


## Documentation

The API is unfortunately not documented yet.
