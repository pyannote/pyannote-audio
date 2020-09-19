> The MIT License (MIT)
>
> Copyright (c) 2017-2020 CNRS
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

# End-to-end speaker embedding with `pyannote.audio`

This tutorial teaches how to train, validate, and apply a speaker embedding neural network with [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) dataset. It assumes that you have already followed both the [data preparation](../../data_preparation) tutorial and the [VoxCeleb installation instructions](https://github.com/pyannote/pyannote-db-voxceleb). 

## Table of contents
- [Citation](#citation)
- [Configuration](#configuration)
- [Training](#training)
- [Validation](#validation)
- [Application](#application)
- [More options](#more-options)

## Citation
([↑up to table of contents](#table-of-contents))

If you use `pyannote-audio` for speaker (or audio) neural embedding, please cite the following papers:

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

## Configuration
([↑up to table of contents](#table-of-contents))

To ensure reproducibility, `pyannote-audio` relies on a configuration file defining the experimental setup:

```bash
$ export EXP_DIR=tutorials/models/speaker_embedding
$ cat ${EXP_DIR}/config.yml
```
```yaml
# Since we are training an end-to-end model, the
# feature extraction step simply returns the raw
# waveform.
feature_extraction:
   name: pyannote.audio.features.RawAudio
   params:
      sample_rate: 16000

# add background noise and music from MUSAN
data_augmentation:
   name: pyannote.audio.augmentation.noise.AddNoise
   params:
     snr_min: 5
     snr_max: 15
     collection:
       - MUSAN.Collection.BackgroundNoise
       - MUSAN.Collection.Music

# use SincTDNN architecture (basically an x-vector
# but where filters are learned instead of being handcrafted)
architecture:
   name: pyannote.audio.models.SincTDNN
   params:
      sincnet:
         stride: [5, 1, 1]
         waveform_normalize: True
         instance_normalize: True
      tdnn:
         embedding_dim: 512
      embedding:
         batch_normalize: False
         unit_normalize: False

# we use additive angular margin loss
task:
   name: AdditiveAngularMarginLoss
   params:
      margin: 0.05
      s: 10
      duration: 2.0   # train on 2s audio chunks
      per_fold: 256   # number of speakers per batch
      per_label: 1    # number of sample per speaker
      per_epoch: 5    # one epoch = 5 days worth of audio

scheduler:
   name: ConstantScheduler
   params:
      learning_rate: 0.01
```

## Training
([↑up to table of contents](#table-of-contents))

The following command will train the network using the training subset of VoxCeleb2 database for 200 epochs:

```bash
$ pyannote-audio emb train --subset=train --to=250 --parallel=8 ${EXP_DIR} VoxCeleb.SpeakerVerification.VoxCeleb2
```

This will create a bunch of files in `TRN_DIR` (defined below). One can also follow along the training process using [tensorboard](https://github.com/tensorflow/tensorboard):
```bash
$ tensorboard --logdir=${EXP_DIR}
```

![tensorboard screenshot](tb_train.png)


## Validation
([↑up to table of contents](#table-of-contents))

To get a quick idea of how the network is doing on the development set, one can use the `validate` mode.

```bash
$ export TRN_DIR=${EXP_DIR}/train/VoxCeleb.SpeakerVerification.VoxCeleb1.train
$ pyannote-audio emb validate --subset=test --to=250 --every=5 ${TRN_DIR} VoxCeleb.SpeakerVerification.VoxCeleb1
```
It can be run while the model is still training and evaluates the model every 5 epochs. This will create a bunch of files in `VAL_DIR` (defined below). 

In practice, it is tuning a simple speaker verification experiment and stores the best hyper-parameter configuration on disk:

```bash
$ export VAL_DIR=${TRN_DIR}/validate_equal_error_rate/VoxCeleb.SpeakerVerification.VoxCeleb1.test
$ cat ${VAL_DIR}/params.yml
```
```yaml
epoch: 221
equal_error_rate: 0.043624072110286335
```

This model reaches 4.4% EER after 220 epochs (approximately 3 days of training). Note that this is obtained by simply using cosine distance between average embedding. This can be further reduced by using a different backend.

![tensorboard screenshot](tb_validate.png)


## Application
([↑up to table of contents](#table-of-contents))

Now that we know how the model is doing, we can apply it on test files of the AMI database: 

```bash
$ pyannote-audio emb apply --step=0.1 --subset=test ${VAL_DIR} AMI.SpeakerDiarization.MixHeadset 
```

Embeddings will be extracted in `${VAL_DIR}/apply/{BEST_EPOCH}`

In the above example, embeddings are extracted every 200ms (0.1 * 2s) using a 2s sliding window. We can then use these extracted embeddings like this:

```python
# first test file of AMI protocol
>>> from pyannote.database import get_protocol
>>> protocol = get_protocol('AMI.SpeakerDiarization.MixHeadset')
>>> test_file = next(protocol.test())

# precomputed embeddings as pyannote.core.SlidingWindowFeature
>>> from pyannote.audio.features import Precomputed
>>> precomputed = Precomputed('{VAL_DIR}/apply/{BEST_EPOCH}')
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
$ pyannote-audio --help
```

That's all folks!
