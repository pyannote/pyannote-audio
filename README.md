# pyannote-audio

Neural building blocks for speaker diarization: 
* speech activity detection
* speaker change detection
* speaker embedding
* speaker diarization pipeline

## Installation

```bash
$ conda create --name pyannote python=3.6
$ source activate pyannote
$ pip install pyannote.audio

# support Yaafe feature extraction (optional)
$ conda install -c conda-forge yaafe
```

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
