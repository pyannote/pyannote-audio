#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2020 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

"""
`pyannote.audio` provides

  * speech activity detection
  * speaker change detection
  * speaker embedding
  * speaker diarization pipeline

## Installation

```bash
$ pip install pyannote.audio
```

## Citation

If you use `pyannote.audio` please use the following citations.

  - Speech  activity and speaker change detection

        @inproceedings{Yin2017,
          Author = {Ruiqing Yin and Herv\'e Bredin and Claude Barras},
          Title = {{Speaker Change Detection in Broadcast TV using Bidirectional Long Short-Term Memory Networks}},
          Booktitle = {{18th Annual Conference of the International Speech Communication Association, Interspeech 2017}},
          Year = {2017},
          Month = {August},
          Address = {Stockholm, Sweden},
          Url = {https://github.com/yinruiqing/change_detection}
        }

  - Speaker embedding

        @inproceedings{Bredin2017,
            author = {Herv\'{e} Bredin},
            title = {{TristouNet: Triplet Loss for Speaker Turn Embedding}},
            booktitle = {42nd IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2017},
            year = {2017},
            url = {http://arxiv.org/abs/1609.04301},
        }

  - Speaker diarization pipeline

        @inproceedings{Yin2018,
          Author = {Ruiqing Yin and Herv\'e Bredin and Claude Barras},
          Title = {{Neural Speech Turn Segmentation and Affinity Propagation for Speaker Diarization}},
          Booktitle = {{19th Annual Conference of the International Speech Communication Association, Interspeech 2018}},
          Year = {2018},
          Month = {September},
          Address = {Hyderabad, India},
        }

"""

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
