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
> AUTHORS
> Ruiqing Yin
> Hervé Bredin - http://herve.niderb.fr

# Speaker change detection with `pyannote.audio`

In this tutorial, you will learn how to train, validate, and apply a speaker change detector based on MFCCs and LSTMs, using `pyannote-change-detection` command line tool.

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

If you use `pyannote-audio` for speaker change detection, please cite the following paper:

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

To ensure reproducibility, `pyannote-change-detection` relies on a configuration file defining the experimental setup:

```bash
$ cat tutorials/change-detection/config.yml
```
```yaml
# train the network for speaker change detection
# see pyannote.audio.labeling.tasks for more details
task:
   name: SpeakerChangeDetection
   params:
      duration: 3.2     # sub-sequence duration
      per_epoch: 36000  # 10 hours of audio per epoch
      collar: 0.200     # up-sampling collar
      batch_size: 32    # number of sub-sequences per batch
      parallel: 4       # number of background batch generators

# use precomputed features (see feature extraction tutorial)
feature_extraction:
   name: Precomputed
   params:
      root_dir: tutorials/feature-extraction

# use the StackedRNN architecture.
# see pyannote.audio.labeling.models for more details
architecture:
   name: StackedRNN
   params:
     rnn: LSTM
     recurrent: [32, 20]
     bidirectional: True
     linear: [40, 10]

# use cyclic learning rate scheduler
scheduler:
   name: CyclicScheduler
   params:
       learning_rate: auto
```

## Training
([↑up to table of contents](#table-of-contents))

The following command will train the network using the training set of AMI database for 1000 epoch:

```bash
$ export EXPERIMENT_DIR=tutorials/change-detection
$ pyannote-change-detection train --gpu --to=1000 ${EXPERIMENT_DIR} AMI.SpeakerDiarization.MixHeadset
```

This will create a bunch of files in `TRAIN_DIR` (defined below).
One can follow along the training process using [tensorboard](https://github.com/tensorflow/tensorboard).
```bash
$ tensorboard --logdir=${EXPERIMENT_DIR}
```

## Validation
([↑up to table of contents](#table-of-contents))

To get a quick idea of how the network is doing during training, one can use the `validate` mode.
It can (should!) be run in parallel to training and evaluates the model epoch after epoch.
One can use [tensorboard](https://github.com/tensorflow/tensorboard) to follow the validation process.

```bash
$ export TRAIN_DIR=${EXPERIMENT_DIR}/train/AMI.SpeakerDiarization.MixHeadset.train
$ pyannote-change-detection validate --purity=0.8 ${TRAIN_DIR} AMI.SpeakerDiarization.MixHeadset
```

By default, this validation computes the segmentation metrics.  
You may prefer to use the option `--diarization` for diarization metrics.

## Application
([↑up to table of contents](#table-of-contents))

Now that we know how the model is doing, we can apply it on all files of the AMI database and store raw change scores in `/path/to/scd`:

```bash
$ pyannote-change-detection apply ${TRAIN_DIR}/weights/0050.pt AMI.SpeakerDiarization.MixHeadset /path/to/scd
```

We can then use these raw scores to perform actual speaker change detection, and [`pyannote.metrics`](http://pyannote.github.io/pyannote-metrics/) to evaluate the result:


```python
# AMI protocol
>>> from pyannote.database import get_protocol
>>> protocol = get_protocol('AMI.SpeakerDiarization.MixHeadset')

# precomputed scores
>>> from pyannote.audio.features import Precomputed
>>> precomputed = Precomputed('/path/to/scd')

# peak detection
>>> from pyannote.audio.signal import Peak
# alpha / min_duration are tunable parameters (and should be tuned for better performance)
# we use log_scale = True because of the final log-softmax in the StackedRNN model
>>> peak = Peak(alpha=0.5, min_duration=1.0, log_scale=True)

# evaluation metric
>>> from pyannote.metrics.diarization import DiarizationPurityCoverageFMeasure
>>> metric = DiarizationPurityCoverageFMeasure()

# loop on test files
>>> from pyannote.database import get_annotated
>>> for test_file in protocol.test():
...    # load reference annotation
...    reference = test_file['annotation']
...    uem = get_annotated(test_file)
...
...    # load precomputed change scores as pyannote.core.SlidingWindowFeature
...    scd_scores = precomputed(test_file)
...
...    # binarize scores to obtain speech regions as pyannote.core.Timeline
...    hypothesis = peak.apply(scd_scores, dimension=1)
...
...    # evaluate speech activity detection
...    metric(reference, hypothesis.to_annotation(), uem=uem)

>>> purity, coverage, fmeasure = metric.compute_metrics()
>>> print(f'Purity = {100*purity:.1f}% / Coverage = {100*coverage:.1f}%')
```

## More options

For more options, see:

```bash
$ pyannote-change-detection --help
```

That's all folks!
