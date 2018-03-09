> The MIT License (MIT)
>
> Copyright (c) 2017-2018 CNRS
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

# Neural speech turn embedding with `pyannote.audio`

In this tutorial, you will learn how to train a [_TristouNet_](http://arxiv.org/abs/1609.04301) speech turn embedding using `pyannote-speaker-embedding` command line tool.

## Table of contents
- [Citation](#citation)
- [ETAPE database](#etape-database)
- [Configuration](#configuration)
- [Training](#training)
- [Validation](#validation)
- [Application](#application)
- [More options](#more-options)

## Citation
([↑up to table of contents](#table-of-contents))

If you use `pyannote-audio` for speaker (or audio) neural embedding, please cite the following paper:

```bibtex
@inproceedings{Bredin2017,
    author = {Herv\'{e} Bredin},
    title = {{TristouNet: Triplet Loss for Speaker Turn Embedding}},
    booktitle = {42nd IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2017},
    year = {2017},
    url = {http://arxiv.org/abs/1609.04301},
}
```

## ETAPE database
([↑up to table of contents](#table-of-contents))

```bash
$ source activate pyannote
$ pip install pyannote.db.etape
```

This tutorial relies on the [ETAPE database](http://islrn.org/resources/425-777-374-455-4/). We first need to tell `pyannote` where the audio files are located:

```bash
$ cat ~/.pyannote/db.yml
Etape: /path/to/Etape/corpus/{uri}.wav
```

If you want to train the network using a different database, you might need to create your own [`pyannote.database`](http://github.com/pyannote/pyannote-database) plugin.
See [github.com/pyannote/pyannote-db-template](https://github.com/pyannote/pyannote-db-template) for details on how to do so.

## Configuration
([↑up to table of contents](#table-of-contents))

To ensure reproducibility, `pyannote-speaker-embedding` relies on a configuration file defining the experimental setup:

```bash
$ cat tutorials/speaker-embedding/config.yml
```
```yaml
# train the network with regular triplet loss
# see pyannote.audio.embedding.approaches for more details
approach:
   name: TripletLoss
   params:
     duration: 2
     sampling: all
     per_fold: 20
     per_label: 3
     parallel: 2

# use precomputed features (from feature extraction tutorials)
feature_extraction:
   name: Precomputed
   params:
      root_dir: /path/to/tutorials/feature-extraction

# use the TristouNet architecture.
# see pyannote.audio.embedding.models for more details
architecture:
   name: TristouNet
   params:
     rnn: LSTM
     recurrent: [16]
     bidirectional: True
     pooling: sum
     linear: [16, 16]
```

## Training
([↑up to table of contents](#table-of-contents))

The following command will train the network using the training set of the `TV` protocol of the ETAPE database for 50 epochs (one epoch = one minute per speaker).

```bash
$ export EXPERIMENT_DIR=tutorials/speaker-embedding
$ pyannote-speaker-embedding train --to=50 ${EXPERIMENT_DIR} Etape.SpeakerDiarization.TV
```
```
Epoch #0: 100%|█████████████████████████████████████| 36/36 [00:40<00:00,  1.12s/it]
Epoch #1: 100%|█████████████████████████████████████| 36/36 [00:24<00:00,  1.47it/s]
...
Epoch #50: 100%|████████████████████████████████████| 36/36 [00:24<00:00,  1.46it/s]
```

This will create a bunch of files in `TRAIN_DIR` (defined below).
One can follow along the training process using [tensorboard](https://github.com/tensorflow/tensorboard).
```bash
$ tensorboard --logdir=${EXPERIMENT_DIR}
```

Among other things, it allows to visualize the evolution of (intra/inter-speaker) distance distributions epoch after epoch:

![tensorboard screenshot](tb_train.png)


## Validation
([↑up to table of contents](#table-of-contents))

To get a quick idea of how the network is doing during training, one can use the `validate` mode.
It can (should!) be run in parallel to training and evaluates the model epoch after epoch.
One can use [tensorboard](https://github.com/tensorflow/tensorboard) to follow the validation process.

```bash
$ export TRAIN_DIR=${EXPERIMENT_DIR}/train/Etape.SpeakerDiarization.TV.train
$ pyannote-speaker-embedding validate ${TRAIN_DIR} Etape.SpeakerDiarization.TV
```
```
Epoch #17 : EER.2s = 24.173% [24.173%, #17]: : 18epoch [29:45, 92.73s/epoch]
```

## Application
([↑up to table of contents](#table-of-contents))

Now that we know how the model is doing, we can apply it on all files of the `TV` protocol of the ETAPE database and store raw SAD scores in `/path/to/emb`:

```bash
$ pyannote-speaker-embedding apply ${TRAIN_DIR}/weights/0050.pt Etape.SpeakerDiarization.TV /path/to/emb
```

We can then use these extracted embeddings like this:


```python
# first test file of ETAPE protocol
>>> from pyannote.database import get_protocol
>>> protocol = get_protocol('Etape.SpeakerDiarization.TV')
>>> test_file = next(protocol.test())

# precomputed embeddings as pyannote.core.SlidingWindowFeature
>>> from pyannote.audio.features import Precomputed
>>> precomputed = Precomputed('/path/to/emb')
>>> embeddings = precomputed(test_file)

# iterate over all embeddings
>>> for window, embedding in embeddings:
...     print(window)
...     print(embedding)
...     break
# [ 00:00:00.000 -->  00:00:02.000]
# [ 0.20038879  0.19046214 -0.01471622 -0.41413733 -0.07942801  0.09274344
#  -0.06193411 -0.05772949 -0.04593764 -0.5080932   0.08709101  0.5761991
#  -0.17651494  0.30997014 -0.00222487 -0.05157921]

# extract embedding from a specific segment
>>> from pyannote.core import Segment
>>> fX = embeddings.crop(Segment(10, 20))
>>> print(fX.shape)
# (13, 16)
```
## More options

For more options, including training on GPU, see:

```bash
$ pyannote-speaker-embedding --help
```

That's all folks!
