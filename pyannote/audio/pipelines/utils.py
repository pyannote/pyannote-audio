# The MIT License (MIT)
#
# Copyright (c) 2017-2021 CNRS
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
from typing import Mapping, Text, Union

import numpy as np
import torch
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.config import from_dict as augmentation_from_dict

from pyannote.audio import Inference, Model
from pyannote.audio.core.io import AudioFile
from pyannote.core import Annotation, SlidingWindowFeature


def assert_string_labels(annotation: Annotation, name: str):
    """Check that annotation only contains string labels

    Parameters
    ----------
    annotation : Annotation
        Annotation.
    name : str
        Name of the annotation (used for user feedback in case of failure)
    """

    if any(not isinstance(label, str) for label in annotation.labels()):
        msg = f"{name} must contain `str` labels only."
        raise ValueError(msg)


def assert_int_labels(annotation: Annotation, name: str):
    """Check that annotation only contains integer labels

    Parameters
    ----------
    annotation : Annotation
        Annotation.
    name : str
        Name of the annotation (used for user feedback in case of failure)
    """

    if any(not isinstance(label, int) for label in annotation.labels()):
        msg = f"{name} must contain `int` labels only."
        raise ValueError(msg)


def gather_label_embeddings(
    annotation: Annotation,
    embeddings: Union[SlidingWindowFeature, Inference],
    file: AudioFile = None,
):
    """Extract one embedding per label

    Parameters
    ----------
    annotation : Annotation
        Annotation
    embeddings : SlidingWindowFeature or Inference
        Embeddings, either precomputed on a sliding window (SlidingWindowFeature)
        or to be computed on the fly (Inference).
    file : AudioFile, optional
        Needed when `embeddings` is an `Inference` instance

    Returns
    -------
    embeddings : ((len(embedded_labels), embedding_dimension) np.ndarray
        Embeddings.
    embedded_labels : list of labels
        Labels for which an embedding has been computed.
    skipped_labels : list of labels
        Labels for which no embedding could be computed.
    """

    X, embedded_labels, skipped_labels = [], [], []

    labels = annotation.labels()
    for label in labels:

        label_support = annotation.label_timeline(label, copy=False)

        if isinstance(embeddings, SlidingWindowFeature):

            # be more and more permissive until we have
            # at least one embedding for current speech turn
            for mode in ["strict", "center", "loose"]:
                x = embeddings.crop(label_support, mode=mode)
                if len(x) > 0:
                    break

            # skip labels so small we do not have any embedding for it
            if len(x) < 1:
                skipped_labels.append(label)
                continue

            embedded_labels.append(label)
            X.append(np.mean(x, axis=0))

        elif isinstance(embeddings, Inference):

            try:
                x = embeddings.crop(file, label_support)
            except RuntimeError:
                # skip labels so small that we cannot even extract embeddings
                skipped_labels.append(label)
                continue

            if embeddings.window == "sliding":
                X.append(np.mean(x, axis=0))

            elif embeddings.window == "whole":
                X.append(x)

            embedded_labels.append(label)

    return np.vstack(X), embedded_labels, skipped_labels


PipelineModel = Union[Model, Text, Mapping]


def get_model(model: PipelineModel) -> Model:
    """Load pretrained model and set it into `eval` mode.

    Parameter
    ---------
    model : Model, str, or dict
        When `Model`, returns `model` as is.
        When `str`, assumes that this is either the path to a checkpoint or the name of a
        pretrained model on Huggingface.co and loads with `Model.from_pretrained(model)`
        When `dict`, loads with `Model.from_pretrained(**model)`.

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

    elif isinstance(model, Text):
        model = Model.from_pretrained(model)

    elif isinstance(model, Mapping):
        model = Model.from_pretrained(**model)

    else:
        raise TypeError(
            f"Unsupported type ({type(model)}) for loading model: "
            f"expected `str` or `dict`."
        )

    model.eval()
    return model


PipelineInference = Union[Inference, Model, Text, Mapping]


def get_inference(inference: PipelineInference) -> Inference:
    """Load inference

    Parameter
    ---------
    inference : Inference, Model, str, or dict
        When `Inference`, returns `inference` as is.
        When `Model`, wraps it in `Inference(model)`.
        When `str`, assumes that this is either the path to a checkpoint or the name of a
        pretrained model on Huggingface.co and loads with `Inference(checkpoint)`.
        When `dict`, loads with `Inference(**inference)`.

    Returns
    -------
    inference : Inference
        Inference.

    Examples
    --------
    >>> inference = get_inference("hbredin/VoiceActivityDetection-PyanNet-DIHARD")
    >>> inference = get_inference("/path/to/checkpoint.ckpt")
    >>> inference = get_inference({"model": "hbredin/VoiceActivityDetection-PyanNet-DIHARD",
    ...                            "window": "sliding"})

    See also
    --------
    pyannote.audio.core.inference.Inference

    """

    if isinstance(inference, Inference):
        return inference

    if isinstance(inference, (Model, Text)):
        return Inference(inference)

    if isinstance(inference, Mapping):
        return Inference(**inference)

    raise TypeError(
        f"Unsupported type ({type(inference)}) for loading inference: "
        f"expected `Model`, `str` or `dict`."
    )


PipelineAugmentation = Union[BaseWaveformTransform, Mapping]


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


def get_devices(needs: int = None):
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
