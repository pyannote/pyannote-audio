> The MIT License (MIT)
>
> Copyright (c) 2017-2019 CNRS
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
>
> AUTHOR
> Hervé Bredin - http://herve.niderb.fr

# Speech activity detection with `pyannote.audio`

In this tutorial, you will learn how to train, validate, and apply a speech activity detector based on MFCCs and LSTMs, using `pyannote-speech-detection` command line tool.

## Table of contents
- [Citation](#citation)
- [Databases](#databases)
- [Configuration](#configuration)
- [Training](#training)
- [Validation](#validation)
- [Application](#application)
- [More options](#more-options)

## Citation
([↑up to table of contents](#table-of-contents))

If you use `pyannote-audio` for speech activity detection, please cite the following paper:

```bibtex
@inproceedings{Yin2017,
  Author = {Ruiqing Yin and Herv\'e Bredin and Claude Barras},
  Title = {{Speaker Change Detection in Broadcast TV using Bidirectional Long Short-Term Memory Networks}},
  Booktitle = {{Interspeech 2017, 18th Annual Conference of the International Speech Communication Association}},
  Year = {2017},
  Month = {August},
  Address = {Stockholm, Sweden},
  Url = {https://github.com/yinruiqing/change_detection}
}
```

## Databases
([↑up to table of contents](#table-of-contents))

```bash
$ source activate pyannote
$ pip install pyannote.db.odessa.ami
$ pip install pyannote.db.musan
```

This tutorial relies on the [AMI](http://groups.inf.ed.ac.uk/ami/corpus) and [MUSAN](http://www.openslr.org/17/) databases. We first need to tell `pyannote` where the audio files are located:

```bash
$ cat ~/.pyannote/database.yml
Databases:
  AMI: /path/to/ami/amicorpus/*/audio/{uri}.wav
  MUSAN: /path/to/musan/{uri}.wav
```

Have a look at `pyannote.database` [documentation](http://github.com/pyannote/pyannote-database) to learn how to use other datasets.

## Configuration
([↑up to table of contents](#table-of-contents))

To ensure reproducibility, `pyannote-speech-detection` relies on a configuration file defining the experimental setup:

```bash
$ cat tutorials/models/speech_activity_detection/config.yml
```
```yaml
task:
   name: SpeechActivityDetection
   params:
      duration: 2.0      # sequences are 2s long
      batch_size: 64     # 64 sequences per batch
      per_epoch: 1       # one epoch = 1 day of audio

data_augmentation:
   name: AddNoise                                   # add noise on-the-fly
   params:
      snr_min: 10                                   # using random signal-to-noise
      snr_max: 20                                   # ratio between 10 and 20 dBs
      collection: MUSAN.Collection.BackgroundNoise  # use background noise from MUSAN
                                                    # (needs pyannote.db.musan)
feature_extraction:
   name: LibrosaMFCC      # use MFCC from librosa
   params:
      e: False            # do not use energy
      De: True            # use energy 1st derivative
      DDe: True           # use energy 2nd derivative
      coefs: 19           # use 19 MFCC coefficients
      D: True             # use coefficients 1st derivative
      DD: True            # use coefficients 2nd derivative
      duration: 0.025     # extract MFCC from 25ms windows
      step: 0.010         # extract MFCC every 10ms
      sample_rate: 16000  # convert to 16KHz first (if needed)

architecture:
   name: StackedRNN
   params:
      instance_normalize: True  # normalize sequences
      rnn: LSTM                 # use LSTM (could be GRU)
      recurrent: [128, 128]     # two layers with 128 hidden states
      bidirectional: True       # bidirectional LSTMs
      linear: [32, 32]          # add two linear layers at the end 

scheduler:
   name: CyclicScheduler        # use cyclic learning rate (LR) scheduler
   params:
      learning_rate: auto       # automatically guess LR upper bound
      epochs_per_cycle: 14      # 14 epochs per cycle
```

## Training
([↑up to table of contents](#table-of-contents))

The following command will train the network using the training set of AMI database for 1000 epochs:

```bash
$ export EXPERIMENT_DIR=tutorials/models/speech_activity_detection
$ pyannote-speech-detection train --gpu --to=1000 --subset=train ${EXPERIMENT_DIR} AMI.SpeakerDiarization.MixHeadset
```

This will create a bunch of files in `TRAIN_DIR` (defined below).
One can follow along the training process using [tensorboard](https://github.com/tensorflow/tensorboard).
```bash
$ tensorboard --logdir=${EXPERIMENT_DIR}
```

![tensorboard screenshot](tb_train.png)


## Validation
([↑up to table of contents](#table-of-contents))

To get a quick idea of how the network is doing during training, one can use the `validate` mode.
It can (should!) be run in parallel to training and evaluates the model epoch after epoch.

```bash
$ export TRAIN_DIR=${EXPERIMENT_DIR}/train/AMI.SpeakerDiarization.MixHeadset.train
$ pyannote-speech-detection validate --subset=develop --gpu ${TRAIN_DIR} AMI.SpeakerDiarization.MixHeadset
```

In practice, it is tuning a simple speech activity detection pipeline (pyannote.audio.pipeline.speech_activity_detection.SpeechActivityDetection) after each epoch and stores the best hyper-parameter configuration on disk:

```bash
$ cat ${TRAIN_DIR}/validate/AMI.SpeakerDiarization.MixHeadset/params.yml
```
```yaml
epoch: 280
params:
  min_duration_off: 0.0
  min_duration_on: 0.0
  offset: 0.5503037490496294
  onset: 0.5503037490496294
  pad_offset: 0.0
  pad_onset: 0.0
```

One can also use [tensorboard](https://github.com/tensorflow/tensorboard) to follow the validation process.

![tensorboard screenshot](tb_validate.png)

## Application
([↑up to table of contents](#table-of-contents))

Now that we know how the model is doing, we can apply it on test files of the AMI database: 

```bash
$ export VALIDATE_DIR=${TRAIN_DIR}/validate/AMI.SpeakerDiarization.MixHeadset.development
$ pyannote-speech-detection apply --gpu --subset=test ${VALIDATE_DIR} AMI.SpeakerDiarization.MixHeadset 
```

Raw scores and speech activity detection results will be dumped into the following directory: `${VALIDATE_DIR}/apply/{BEST_EPOCH}`.

## More options

For more options, see:

```bash
$ pyannote-speech-detection --help
```

That's all folks!
