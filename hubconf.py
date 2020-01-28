#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019-2020 CNRS

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

dependencies = ['pyannote.audio', 'torch']

import yaml
from pathlib import Path
from typing import Optional, Union
import functools


import torch
from pyannote.audio.features import Pretrained


MODELS = {
    # speech activity detection
    "sad": {
        "ami": "712d7e3184",
        "etape": "604053b0ac",
        "dihard": "ab30b4cbfb",
        "dihardx": "0116a70245",  # domain-adversarial
    },

    # speaker change detection
    "scd": {
        "ami": "4d326a90b5",
        "etape": "feec2e9fdf",
        "dihard": "0804daa63d",
    },

    # overlapped speech detection
    "ovl": {
        "ami": "7103e99f5b",
        "etape": "acd955e0c2",
        "dihard": "753394ba3b",
    },

    # speaker embedding
    "emb": {
        "voxceleb": "21ba139a32",
    },
}

_DEVICE = Union[str, torch.device]

def _generic(task: str = 'sad',
             corpus: str = 'AMI',
             device: Optional[_DEVICE] = None,
             batch_size: int = 32,
             step: float = 0.25) -> Pretrained:
    """Load pretrained model

    Parameters
    ----------
    task : {'sad', 'scd', 'ovl', 'emb'}, optional
        Use 'sad' for speech activity detection, 'scd' for speaker change
        detection, 'ovl' for overlapped speech detection, and 'emb' for speaker
        embedding. Defaults to 'sad'.
    corpus : {'ami', 'dihard', 'etape', 'voxceleb'}, optional
        Use 'ami' for model trained on AMI corpus, 'dihard' for DIHARD corpus,
        'etape' for ETAPE corpus, 'voxceleb' for VoxCeleb corpus.
    device : str or torch.device, optional
        Device used for inference.
        Defaults to GPU when available.
    batch_size : int, optional
        Batch size used for inference.
    step : float, optional
        Ratio of audio chunk duration used as step between two consecutive
        audio chunks. Defaults to 0.25.

    Returns
    -------
    pretrained : `Pretrained`

    Usage
    -----
    # TODO. change to 'pyannote/pyannote-audio' after 2.0 release
    >>> pretrained = torch.hub.load('pyannote/pyannote-audio:develop',
                                    '_generic', task='sad', corpus='ami',
    ...                             batch_size=32)
    >>> sad_scores = pretrained({'audio': '/path/to/audio.wav'})
    """

    # path where pre-trained model is downloaded by torch.hub
    hub_dir = Path(__file__).parent / 'models' / task / corpus / MODELS[task][corpus]

    # guess path to "params.yml"
    params_yml, = hub_dir.glob('*/*/*/*/params.yml')
    validate_dir = params_yml.parent

    # TODO: print a message to the user providing information about it
    # *_, train, _, development, _ = params_yml.parts
    # msg = 'Model trained on {train}'
    # print(msg)

    # initialize  extraction
    return Pretrained(validate_dir=validate_dir,
                      batch_size=batch_size,
                      device=device,
                      step=step)

_sad = functools.partial(_generic, task='sad')
sad_ami = functools.partial(_sad, corpus='ami')
sad_dihard = functools.partial(_sad, corpus='dihard')
sad_etape = functools.partial(_sad, corpus='etape')
sad_dihardx = functools.partial(_sad, corpus='dihardx')

_ovl = functools.partial(_generic, task='ovl')
ovl_ami = functools.partial(_ovl, corpus='ami')
ovl_dihard = functools.partial(_ovl, corpus='dihard')
ovl_etape = functools.partial(_ovl, corpus='etape')

_scd = functools.partial(_generic, task='scd')
scd_ami = functools.partial(_scd, corpus='ami')
scd_dihard = functools.partial(_scd, corpus='dihard')
scd_etape = functools.partial(_scd, corpus='etape')

_emb = functools.partial(_generic, task='emb')
emb_voxceleb = functools.partial(_emb, corpus='voxceleb')


if __name__ == '__main__':
    DOCOPT = """Create torch.hub zip file from validation directory

Usage:
  hubconf.py <validate_dir>
  hubconf.py (-h | --help)
  hubconf.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
    """
    from docopt import docopt
    from pyannote.audio.applications.base import create_zip
    arguments = docopt(DOCOPT, version='hubconf')
    validate_dir = Path(arguments['<validate_dir>'])
    hub_zip = create_zip(validate_dir)
    print(f'Created file "{hub_zip.name}" in directory "{validate_dir}".')
