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
- [AMI database](#ami-database)
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

## AMI database
([↑up to table of contents](#table-of-contents))

```bash
$ source activate pyannote
$ pip install pyannote.db.odessa.ami
```

This tutorial relies on the [AMI database](http://groups.inf.ed.ac.uk/ami/corpus). We first need to tell `pyannote` where the audio files are located:

```bash
$ cat ~/.pyannote/db.yml | grep AMI
AMI: /path/to/ami/amicorpus/*/audio/{uri}.wav
```

If you want to use a different database, you might need to create your own [`pyannote.database`](http://github.com/pyannote/pyannote-database) plugin.
See [github.com/pyannote/pyannote-db-template](https://github.com/pyannote/pyannote-db-template) for details on how to do so. You might also use `pip search pyannote` to browse existing plugins.

## Configuration
([↑up to table of contents](#table-of-contents))

To ensure reproducibility, `pyannote-speaker-embedding` relies on a configuration file defining the experimental setup:

```bash
$ cat tutorials/speaker-embedding/config.yml
```
```yaml
# train the network using triplet loss
# see pyannote.audio.embedding.approaches for more details
approach:
  name: TripletLoss
  params:
    metric: cosine    # embeddings are optimized for cosine metric
    clamp: positive   # triplet loss variant
    margin: 0.2       # triplet loss margin
    duration: 2       # sequence duration
    sampling: all     # triplet sampling strategy
    per_fold: 40      # number of speakers per fold
    per_label: 3      # number of sequences per speaker

# use precomputed features (see feature extraction tutorial)
feature_extraction:
  name: Precomputed
  params:
     root_dir: tutorials/feature-extraction

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

# use cyclic learning rate scheduler
scheduler:
  name: CyclicScheduler
  params:
      learning_rate: auto
```

## Training
([↑up to table of contents](#table-of-contents))

The following command will train the network using the training set of the AMI database for 1000 epochs (one epoch = every speaker seen at least once)

```bash
$ export EXPERIMENT_DIR=tutorials/speaker-embedding
$ pyannote-speaker-embedding train --gpu --to=1000 ${EXPERIMENT_DIR} AMI.SpeakerDiarization.MixHeadset
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
$ export TRAIN_DIR=${EXPERIMENT_DIR}/train/AMI.SpeakerDiarization.MixHeadset.train
$ pyannote-speaker-embedding validate ${TRAIN_DIR} AMI.SpeakerDiarization.MixHeadset
```

## Application
([↑up to table of contents](#table-of-contents))

Now that we know how the model is doing, we can apply it on all files of the AMI database and store raw SAD scores in `/path/to/emb`:

```bash
$ pyannote-speaker-embedding apply ${TRAIN_DIR}/weights/0050.pt AMI.SpeakerDiarization.MixHeadset /path/to/emb
```

We can then use these extracted embeddings like this:


```python
# first test file of AMI protocol
>>> from pyannote.database import get_protocol
>>> protocol = get_protocol('AMI.SpeakerDiarization.MixHeadset')
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

# extract embedding from a specific segment
>>> from pyannote.core import Segment
>>> fX = embeddings.crop(Segment(10, 20))
>>> print(fX.shape)
```

## More options

For more options, see:

```bash
$ pyannote-speaker-embedding --help
```

That's all folks!
