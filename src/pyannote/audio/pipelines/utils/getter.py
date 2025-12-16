# The MIT License (MIT)
#
# Copyright (c) 2017-2025 CNRS
# Copyright (c) 2025- pyannoteAI
#
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

import itertools
from pathlib import Path
from typing import Dict, Mapping

import torch
from pyannote.audio.core.calibration import Calibration
from pyannote.audio.core.model import Model
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.core.plda import PLDA
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.config import from_dict as augmentation_from_dict


def get_pipeline(
    pipeline: Pipeline | str | dict,
    token: str | None = None,
    cache_dir: Path | str | None = None,
) -> Pipeline:
    if isinstance(pipeline, Pipeline):
        _pipeline = pipeline

    elif isinstance(pipeline, str):
        _pipeline = Pipeline.from_pretrained(pipeline, token=token, cache_dir=cache_dir)

    elif isinstance(pipeline, dict):
        if "checkpoint" in pipeline:
            pipeline.setdefault("token", token)
            pipeline.setdefault("cache_dir", cache_dir)
            _pipeline = Pipeline.from_pretrained(**pipeline)

        else:
            _pipeline = Pipeline.from_pretrained(
                pipeline, token=token, cache_dir=cache_dir
            )

    else:
        raise TypeError(
            f"Unsupported type ({type(pipeline)}) for loading pipeline: "
            f"expected `str` or `dict`."
        )

    if _pipeline is None:
        raise ValueError(f"Could not load pipeline: {pipeline}.")

    return _pipeline


PipelineModel = Model | str | Mapping


def get_model(
    model: PipelineModel,
    token: str | None = None,
    cache_dir: Path | str | None = None,
) -> Model:
    """Load pretrained model and set it into `eval` mode.

    Parameter
    ---------
    model : Model, str, or dict
        When `Model`, returns `model` as is.
        When `str`, assumes that this is either the path to a checkpoint or the name of a
        pretrained model on Huggingface.co and loads with `Model.from_pretrained(model)`
        When `dict`, loads with `Model.from_pretrained(**model)`.
    token : str or bool, optional
        Huggingface token to be used for downloading from Huggingface hub.
    cache_dir: Path or str, optional
        Path to the folder where files downloaded from Huggingface hub are stored.

    Returns
    -------
    model : Model
        Model in `eval` mode.

    Examples
    --------
    >>> model = get_model("hbredin/VoiceActivityDetection-PyanNet-DIHARD")
    >>> model = get_model("/path/to/checkpoint.ckpt")
    >>> model = get_model({"checkpoint": "hbredin/VoiceActivityDetection-PyanNet-DIHARD",
    ...                    "map_location": torch.device("cuda")})

    See also
    --------
    pyannote.audio.core.model.Model.from_pretrained

    """

    if isinstance(model, Model):
        pass

    elif isinstance(model, str):
        _model = Model.from_pretrained(
            model,
            token=token,
            cache_dir=cache_dir,
            strict=False,
        )
        if _model:
            model = _model

    elif isinstance(model, Mapping):
        model.setdefault("token", token)
        model.setdefault("cache_dir", cache_dir)
        model = Model.from_pretrained(**model)

    else:
        raise TypeError(
            f"Unsupported type ({type(model)}) for loading model: "
            f"expected `str` or `dict`."
        )

    model.eval()
    return model


PipelineAugmentation = BaseWaveformTransform | Mapping


PipelineCalibration = Calibration | str | Dict


def get_calibration(
    calibration: PipelineCalibration,
    token: str | None = None,
    cache_dir: Path | str | None = None,
) -> Calibration | None:
    """Load pretrained calibration

    Parameters
    ----------
    calibration : Calibration, str, or dict
        When `Calibration`, returns `calibration` as is.
        When `str`, assumes that this is either the path to a checkpoint or the name of a
        pretrained calibration on Huggingface.co and loads with `Calibration.from_pretrained(calibration)`.
        When `dict`, loads with `Calibration.from_pretrained(**calibration)`.
    token : str or bool, optional
        Huggingface token to be used for downloading from Huggingface hub.
    cache_dir: Path or str, optional
        Path to the folder where files downloaded from Huggingface hub are stored.

    Returns
    -------
    calibration : Calibration
        Calibration.

    See also
    --------
    pyannote.audio.core.calibration.Calibration.from_pretrained
    """

    if isinstance(calibration, Calibration):
        loaded_calibration = calibration

    elif isinstance(calibration, str):
        loaded_calibration = Calibration.from_pretrained(
            calibration,
            token=token,
            cache_dir=cache_dir,
        )

    elif isinstance(calibration, Dict):
        calibration.setdefault("token", token)
        calibration.setdefault("cache_dir", cache_dir)
        loaded_calibration = Calibration.from_pretrained(**calibration)

    else:
        raise TypeError(
            f"Unsupported type ({type(calibration)}) for loading calibration: "
            f"expected `str` or `dict`."
        )

    return loaded_calibration


PipelinePLDA = PLDA | str | Dict


def get_plda(
    plda: PipelinePLDA,
    token: str | None = None,
    cache_dir: Path | str | None = None,
) -> PLDA | None:
    """Load pretrained calibration

    Parameters
    ----------
    plda : PLDA, str, or dict
        When `PLDA`, returns `plda` as is.
        When `str`, assumes that this is either the path to a checkpoint or the name of a
        pretrained PLDA on Huggingface.co and loads with `PLDA.from_pretrained(PLDA)`.
        When `dict`, loads with `PLDA.from_pretrained(**plda)`.
    token : str or bool, optional
        Huggingface token to be used for downloading from Huggingface hub.
    cache_dir: Path or str, optional
        Path to the folder where files downloaded from Huggingface hub are stored.

    Returns
    -------
    plda : PLDA
        PLDA.

    See also
    --------
    pyannote.audio.core.plda.PLDA.from_pretrained
    """

    if isinstance(plda, PLDA):
        loaded_plda = plda

    elif isinstance(plda, str):
        loaded_plda = PLDA.from_pretrained(plda, token=token, cache_dir=cache_dir)

    elif isinstance(plda, Dict):
        plda.setdefault("token", token)
        plda.setdefault("cache_dir", cache_dir)
        loaded_plda = PLDA.from_pretrained(**plda)

    else:
        raise TypeError(
            f"Unsupported type ({type(plda)}) for loading PLDA: "
            f"expected `str` or `dict`."
        )

    return loaded_plda


def get_augmentation(augmentation: PipelineAugmentation) -> BaseWaveformTransform:
    """Load augmentation

    Parameter
    ---------
    augmentation : BaseWaveformTransform, or dict
        When `BaseWaveformTransform`, returns `augmentation` as is.
        When `dict`, loads with `torch_audiomentations`'s `from_config` utility function.

    Returns
    -------
    augmentation : BaseWaveformTransform
        Augmentation.
    """

    if augmentation is None:
        return None

    if isinstance(augmentation, BaseWaveformTransform):
        return augmentation

    if isinstance(augmentation, Mapping):
        return augmentation_from_dict(augmentation)

    raise TypeError(
        f"Unsupported type ({type(augmentation)}) for loading augmentation: "
        f"expected `BaseWaveformTransform`, or `dict`."
    )


def get_devices(needs: int | None = None):
    """Get devices that can be used by the pipeline

    Parameters
    ----------
    needs : int, optional
        Number of devices needed by the pipeline

    Returns
    -------
    devices : list of torch.device
        List of available devices.
        When `needs` is provided, returns that many devices.
    """

    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        devices = [torch.device("cpu")]
        if needs is None:
            return devices
        return devices * needs

    devices = [torch.device(f"cuda:{index:d}") for index in range(num_gpus)]
    if needs is None:
        return devices
    return [device for _, device in zip(range(needs), itertools.cycle(devices))]
