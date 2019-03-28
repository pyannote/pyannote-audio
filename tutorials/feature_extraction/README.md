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

# Feature extraction with `pyannote.audio`

In this tutorial, you will learn how to perform feature extraction using `pyannote-speech-feature` command line tool.

## Table of contents
- [AMI database](#etape-database)
- [Configuration](#configuration)
- [Extraction](#extraction)
- [Usage](#usage)
  - [In other command line tools](#in-other-command-line-tools)
  - [In your own code](#in-your-own-code)

## AMI database
([↑up to table of contents](#table-of-contents))

```bash
$ source activate pyannote
$ pip install pyannote.db.odessa.ami
```

This tutorial relies on the [AMI database](http://groups.inf.ed.ac.uk/ami/corpus). We first need to tell `pyannote` where the audio files are located:

```bash
$ cat ~/.pyannote/database.yml
Databases:
  AMI: /path/to/ami/amicorpus/*/audio/{uri}.wav
```

Have a look at `pyannote.database` [documentation](http://github.com/pyannote/pyannote-database) to learn how to use other datasets.

## Configuration
([↑up to table of contents](#table-of-contents))

To ensure reproducibility, `pyannote-speech-feature` relies on a configuration file defining the experimental setup:

```bash
$ cat tutorials/feature-extraction/config.yml
feature_extraction:
   name: LibrosaMFCC                # extract MFCCs using Yaafe
   params:
      e: False                      # no energy
      De: True                      # energy 1st derivative
      DDe: True                     # energy 2nd derivative
      coefs: 19                     # 19 coefficients
      D: True                       # with 1st derivatives
      DD: True                      # and 2nd derivatives
      duration: 0.025               # one 25ms-long windows
      step: 0.010                   # and a step of 10ms
      sample_rate: 16000
```

## Extraction
([↑up to table of contents](#table-of-contents))

The following command will extract features for all files of the `MixHeadset` protocol of the AMI database.

```bash
$ export EXPERIMENT_DIR=tutorials/feature_extraction
$ pyannote-speech-feature ${EXPERIMENT_DIR} AMI.SpeakerDiarization.MixHeadset
Development set: 21it [01:28,  4.21s/it]
Test set: 22it [01:39,  4.53s/it]
Training set: 115it [09:33,  4.99s/it]
```

This will create one a bunch of files in `EXPERIMENT_DIR`.
```bash
$ ls $EXPERIMENT_DIR
AMI config.yml metadata.yml
$ ls $EXPERIMENT_DIR/AMI | head -n 5
EN2001a.Mix-Headset.npy
EN2001b.Mix-Headset.npy
EN2001d.Mix-Headset.npy
EN2001e.Mix-Headset.npy
EN2002b.Mix-Headset.npy
```

## Usage
([↑up to table of contents](#table-of-contents))

### In other command line tools
([↑up to table of contents](#table-of-contents))

Now that features are extracted, they can be used by other command line tools (instead of re-computing them on-the-fly).

For instance, the `feature_extraction` section of the configuration file of the [speech activity detection tutorial](/tutorials/models/speech_activity_detection) can be updated to look like that:

```bash
$ cat tutorials/models/speech_activity_detection/config.yml
feature_extraction:
   name: Precomputed
   params:
      root_dir: tutorials/feature_extraction

[...]
```

### In your own code
([↑up to table of contents](#table-of-contents))

```python
>>> from pyannote.audio.features import Precomputed
>>> precomputed = Precomputed('tutorials/feature_extraction')
>>> from pyannote.database import get_protocol
>>> protocol = get_protocol('AMI.SpeakerDiarization.MixHeadset')
>>> for current_file in protocol.test():
...     features = precomputed(current_file)
...     break
>>> X = features.data                   # numpy array containing all features
>>> X.shape
(178685, 59)
>>> from pyannote.core import Segment
>>> features.crop(Segment(10.2, 11.4))  # numpy array containing local features
array([[ 0.85389346,  0.71583151,  0.71233984, ..., -0.89612021,
        -0.76569814, -0.19767237],
       [-0.47338321, -0.20921302,  0.7786835 , ..., -0.25947172,
        -1.36994643, -0.68953601],
       [-0.06111027, -0.29888008,  0.2566882 , ..., -0.59178806,
        -0.15753769,  0.57210477],
       ...,
       [-1.61349947, -1.13563152, -1.24434275, ...,  0.49641144,
         0.25312351,  1.20094644],
       [-1.15335094, -1.22503884, -0.50867748, ...,  0.23089361,
         0.46149691, -0.29184605],
       [-1.13511339, -1.64100123, -0.9486918 , ...,  0.36467688,
         0.29080623, -1.65317099]])
```

## Going further...
([↑up to table of contents](#table-of-contents))

```bash
$ pyannote-speech-feature --help
```
