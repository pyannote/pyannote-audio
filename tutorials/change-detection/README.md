> The MIT License (MIT)
>
> Copyright (c) 2017 CNRS
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
> Ruiqing Yin

# Speaker change detection with `pyannote.audio`

In this tutorial, you will learn how to train and test a speaker change detection model based on MFCCs and LSTMs, using `pyannote-chang-detection` command line tool.

## Installation

```bash
$ conda create --name py35-pyannote-audio python=3.5 anaconda
$ ource activate py35-pyannote-audio
$ conda install gcc
$ conda install -c conda-forge yaafe
$ pip install "pyannote.audio==0.2.1"
$ pip install pyannote.db.etape
```

## Experimental setup

### ETAPE database

This tutorial relies on the [ETAPE database](http://islrn.org/resources/425-777-374-455-4/). We first need to tell `pyannote` where the audio files are located:

```bash
$ cat ~/.pyannote/db.yml
Etape: /path/to/Etape/corpus/{uri}.wav
```
### Configuration

To ensure reproducibility, `pyannote-change-detection` relies on a configuration file defining the experimental setup:

```bash
$ cat tutorials/change-detection/config.yml
feature_extraction:
   name: YaafeMFCC
   params:
      e: False                   # this experiments relies
      De: True                   # on 11 MFCC coefficients
      DDe: True                  # with 1st and 2nd derivatives
      D: True                    # without energy, but with
      DD: True                   # energy derivatives
      stack: 1

architecture:
   name: StackedLSTM
   params:                       # this experiments relies
     n_classes: 1                # on one LSTM layer (16 outputs)
     lstm: [16]                  # and one dense layer.
     mlp: [16]                   # LSTM is bidirectional
     bidirectional: 'concat'
     final_activation: 'sigmoid'

sequences:
   duration: 3.2                 # this experiments relies on sliding windows
   step: 0.8                     # of 3.2s with a step of 0.8s
   balance: 0.05                 # and balancing neighborhood size of 0.05s

```

### Training

The following command will train the network using the training set of the `TV` protocol of the ETAPE database. This may take a long time...

```bash
$ export EXPERIMENT_DIR=tutorials/change-detection
$ pyannote-change-detection train \       #  
          ${EXPERIMENT_DIR} \             # <experiment_dir>
          Etape.SpeakerDiarization.TV     # <database.task.protocol>
Epoch 1/100
62464/62464 [==============================] - 171s - loss: 0.1543 - acc: 0.9669   
Epoch 2/100
62464/62464 [==============================] - 117s - loss: 0.1375 - acc: 0.9692     
Epoch 3/100
62464/62464 [==============================] - 115s - loss: 0.1376 - acc: 0.9691     
...
Epoch 50/100
62464/62464 [==============================] - 112s - loss: 0.0903 - acc: 0.9724  
...

```

This will create a bunch of files in `TRAIN_DIR` (defined below), including plots showing the accuracy epoch after epoch.

### Evaluation
Now that the network is trained, we get different models for different epochs. We can evaluate a specific model:

```bash
$ export TRAIN_DIR=${EXPERIMENT_DIR}/train/Etape.SpeakerDiarization.TV.train
$ pyannote-change-detection evaluate \
         --epoch=49 \
         ${TRAIN_DIR} \               # <train_dir>
         Etape.SpeakerDiarization.TV  # <database.task.protocol> 
```

This is the expected output:

```
threshold purity  coverage
--------- ------- --------
0         95.720% 36.603%
0.0526316 95.526% 49.018%
0.105263  95.213% 57.660%
...
0.526316  92.396% 83.756%
...
0.894737  88.155% 91.139%
0.947368  87.468% 92.046%
1         86.929% 92.672%

```



### Testing

We can choose the best model according to the evaluation results and  apply it on the development set of the `TV` protocol of the ETAPE database:

```bash
$ pyannote-speech-detection apply \
          --epoch=49 \
          --threshold=0.1 \
          ${TRAIN_DIR} \               #<train_dir>
          Etape.SpeakerDiarization.TV  # <database.task.protocol>
```

this will create a list of files in `APPLY_DIR` (defined below) containing segmentation results.

```bash
$ export APPLY_DIR=${TRAIN_DIR}/segment/Etape.SpeakerDiarization.TV.development/0.1/
$ head -n 5 $APPLY_DIR/BFMTV_BFMStory_2011-03-17_175900.0.seg
            file_id              seg_id  channel start length
-------------------------------- ------- ------- ----- ------
BFMTV_BFMStory_2011-03-17_175900    0      1      -1     560
BFMTV_BFMStory_2011-03-17_175900    1      1      559    356
BFMTV_BFMStory_2011-03-17_175900    2      1      915    204
BFMTV_BFMStory_2011-03-17_175900    3      1      1119   160
BFMTV_BFMStory_2011-03-17_175900    4      1      1279   398
```

## Going further...

```bash
$ pyannote-change-detection --help
```
