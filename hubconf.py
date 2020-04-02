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

import os
import pathlib
import typing
import functools
import shutil
import zipfile
import torch
import yaml
from pyannote.audio.features import Pretrained as _Pretrained
from pyannote.pipeline import Pipeline as _Pipeline

dependencies = ['pyannote.audio', 'torch']

_HUB_REPO = 'https://github.com/pyannote/pyannote-audio-hub'
_ZIP_URL = f'{_HUB_REPO}/raw/master/{{kind}}s/{{name}}.zip'
_PRETRAINED_URL = f'{_HUB_REPO}/raw/master/pretrained.yml'

# path where pre-trained models and pipelines are downloaded and cached
_HUB_DIR = pathlib.Path(os.environ.get("PYANNOTE_AUDIO_HUB",
                                       "~/.pyannote/hub")).expanduser().resolve()

# download pretrained.yml if needed
_PRETRAINED_YML = _HUB_DIR / 'pretrained.yml'

if not _PRETRAINED_YML.exists():
    msg = (
        f'Downloading list of pretrained models and pipelines '
        f'to "{_PRETRAINED_YML}".'
    )
    print(msg)
    from pyannote.audio.utils.path import mkdir_p
    mkdir_p(_PRETRAINED_YML.parent)
    torch.hub.download_url_to_file(_PRETRAINED_URL,
                                   _PRETRAINED_YML,
                                   progress=True)

def _generic(name: str,
             duration: float = None,
             step: float = 0.25,
             batch_size: int = 32,
             device: typing.Optional[typing.Union[typing.Text, torch.device]] = None,
             pipeline: typing.Optional[bool] = None,
             force_reload: bool = False) -> typing.Union[_Pretrained, _Pipeline]:
    """Load pretrained model or pipeline

    Parameters
    ----------
    name : str
        Name of pretrained model or pipeline
    duration : float, optional
        Override audio chunks duration.
        Defaults to the one used during training.
    step : float, optional
        Ratio of audio chunk duration used for the internal sliding window.
        Defaults to 0.25 (i.e. 75% overlap between two consecutive windows).
        Reducing this value might lead to better results (at the expense of
        slower processing).
    batch_size : int, optional
        Batch size used for inference. Defaults to 32.
    device : torch.device, optional
        Device used for inference.
    pipeline : bool, optional
        Wrap pretrained model in a (not fully optimized) pipeline.
    force_reload : bool
        Whether to discard the existing cache and force a fresh download.
        Defaults to use existing cache.

    Returns
    -------
    pretrained: `Pretrained` or `Pipeline`

    Usage
    -----
    >>> sad_pipeline = torch.hub.load('pyannote/pyannote-audio', 'sad_ami')
    >>> scores = model({'audio': '/path/to/audio.wav'})
    """

    model_exists = name in _MODELS
    pipeline_exists = name in _PIPELINES

    if model_exists and pipeline_exists:

        if pipeline is None:
            msg = (
                f'Both a pretrained model and a pretrained pipeline called '
                f'"{name}" are available. Use option "pipeline=True" to '
                f'load the pipeline, and "pipeline=False" to load the model.')
            raise ValueError(msg)

        if pipeline:
            kind = 'pipeline'
            zip_url = _ZIP_URL.format(kind=kind, name=name)
            sha256 = _PIPELINES[name]
            return_pipeline = True

        else:
            kind = 'model'
            zip_url = _ZIP_URL.format(kind=kind, name=name)
            sha256 = _MODELS[name]
            return_pipeline = False

    elif pipeline_exists:

        if pipeline is None:
            pipeline = True

        if not pipeline:
            msg = (
                f'Could not find any pretrained "{name}" model. '
                f'A pretrained "{name}" pipeline does exist. '
                f'Did you mean "pipeline=True"?'
            )
            raise ValueError(msg)

        kind = 'pipeline'
        zip_url = _ZIP_URL.format(kind=kind, name=name)
        sha256 = _PIPELINES[name]
        return_pipeline = True

    elif model_exists:

        if pipeline is None:
            pipeline = False

        kind = 'model'
        zip_url = _ZIP_URL.format(kind=kind, name=name)
        sha256 = _MODELS[name]
        return_pipeline = pipeline

        if name.startswith('emb_') and return_pipeline:
            msg = (
                f'Pretrained model "{name}" has no associated pipeline. Use '
                f'"pipeline=False" or remove "pipeline" option altogether.'
            )
            raise ValueError(msg)

    else:
        msg = (
            f'Could not find any pretrained model nor pipeline called "{name}".'
        )
        raise ValueError(msg)

    if sha256 is None:
        msg = (
            f'Pretrained {kind} "{name}" is not available yet but will be '
            f'released shortly. Stay tuned...'
        )
        raise NotImplementedError(msg)

    pretrained_dir = _HUB_DIR / f'{kind}s'
    pretrained_subdir = pretrained_dir / f'{name}'
    pretrained_zip = pretrained_dir / f'{name}.zip'

    if not pretrained_subdir.exists() or force_reload:

        if pretrained_subdir.exists():
            shutil.rmtree(pretrained_subdir)

        from pyannote.audio.utils.path import mkdir_p
        mkdir_p(pretrained_zip.parent)
        try:
            msg = (
                f'Downloading pretrained {kind} "{name}" to "{pretrained_zip}".'
            )
            print(msg)
            torch.hub.download_url_to_file(zip_url,
                                           pretrained_zip,
                                           hash_prefix=sha256,
                                           progress=True)
        except RuntimeError as e:
            shutil.rmtree(pretrained_subdir)
            msg = (
                f'Failed to download pretrained {kind} "{name}".'
                f'Please try again.')
            raise RuntimeError(msg)

        # unzip downloaded file
        with zipfile.ZipFile(pretrained_zip) as z:
            z.extractall(path=pretrained_dir)

    if kind == 'model':

        params_yml, = pretrained_subdir.glob('*/*/*/*/params.yml')
        pretrained =  _Pretrained(validate_dir=params_yml.parent,
                                  duration=duration,
                                  step=step,
                                  batch_size=batch_size,
                                  device=device)

        if return_pipeline:
            if name.startswith('sad_'):
                from pyannote.audio.pipeline.speech_activity_detection import SpeechActivityDetection
                pipeline = SpeechActivityDetection(scores=pretrained)
            elif name.startswith('scd_'):
                from pyannote.audio.pipeline.speaker_change_detection import SpeakerChangeDetection
                pipeline = SpeakerChangeDetection(scores=pretrained)
            elif name.startswith('ovl_'):
                from pyannote.audio.pipeline.overlap_detection import OverlapDetection
                pipeline = OverlapDetection(scores=pretrained)
            else:
                # this should never happen
                msg = (
                    f'Pretrained model "{name}" has no associated pipeline. Use '
                    f'"pipeline=False" or remove "pipeline" option altogether.'
                )
                raise ValueError(msg)

            return pipeline.load_params(params_yml)

        return pretrained

    elif kind == 'pipeline':

        from pyannote.audio.pipeline.utils import load_pretrained_pipeline
        params_yml, *_ = pretrained_subdir.glob('*/*/params.yml')
        return load_pretrained_pipeline(params_yml.parent)

with open(_PRETRAINED_YML, 'r') as fp:
    _pretrained = yaml.load(fp, Loader=yaml.SafeLoader)

_MODELS = _pretrained['models']
for name in _MODELS:
    locals()[name] = functools.partial(_generic, name)

_PIPELINES = _pretrained['pipelines']
for name in _PIPELINES:
    locals()[name] = functools.partial(_generic, name)

_SHORTCUTS = _pretrained['shortcuts']
for shortcut, name in _SHORTCUTS.items():
    locals()[shortcut] = locals()[name]
