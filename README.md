# pyannote-audio

Neural building blocks for speaker diarization: 
* speech activity detection
* speaker change detection
* overlapped speech detection
* speaker embedding
* speaker diarization pipeline

## Installation

```bash
# Step 1: This creates a conda environment with python3.6
$ conda create env -f environment.yml
$ source activate pyannote

# Step 2:  install from source in the "develop" branch
$ git clone https://github.com/pyannote/pyannote-audio.git
$ cd pyannote-audio
$ git checkout develop
$ pip install .
```

If you're using Shennong for feature extraction, use the `environment-shennong.yml` 
environment config file. If you're willing to use a regular pip environment, make sure you have PyYAML installed 
in that env, and run Step 2.

## Citation

If you use `pyannote.audio` please use the following citations.

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
  - Speaker diarization pipeline
    ```bibtex
    @inproceedings{Yin2018,
      Author = {Ruiqing Yin and Herv\'e Bredin and Claude Barras},
      Title = {{Neural Speech Turn Segmentation and Affinity Propagation for Speaker Diarization}},
      Booktitle = {{19th Annual Conference of the International Speech Communication Association, Interspeech 2018}},
      Year = {2018},
      Month = {September},
      Address = {Hyderabad, India},
    }
    ```

## Tutorials

:warning: These tutorials assumes that you installed the [`develop` branch](https://github.com/pyannote/pyannote-audio/issues/145) of `pyannote.audio`.  
:warning: They are most likely [broken](https://github.com/pyannote/pyannote-audio/issues/151) in `pyannote.audio 1.x`.

  * [Feature extraction](tutorials/feature_extraction)
  * Models
    * [Training LSTM-based speech activity detection](tutorials/models/speech_activity_detection)
    * [Training LSTM-based speaker change detection](tutorials/models/speaker_change_detection)
    * [Training LSTM-based overlapped speech detection](tutorials/models/overlap_detection)
    * [Training LSTM-based speaker embedding](tutorials/models/speaker_embedding)

  * Pipelines
    * [Tuning speech activity detection pipeline](tutorials/pipelines/speech_activity_detection)
    * [Tuning speaker diarization pipeline](tutorials/pipelines/speaker_diarization)
  
  * In-house datasets 
    * [Applying pre-trained models on your own data](tutorials/own_data/pretrained)
    * [Training models on your own data](tutorials/own_data/train)

  
## Documentation

Part of the API is described in [this](tutorials/models/pretrained) tutorial.  
Other than that, there is still a lot to do (contribute?) documentation-wise...
