#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2020 CNRS

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


import yaml
from pathlib import Path
from pyannote.core import Annotation
from pyannote.pipeline import Pipeline
from pyannote.core.utils.helper import get_class_by_name


def assert_string_labels(annotation: Annotation, name: str):
    """Check that annotation only contains string labels

    Parameters
    ----------
    annotation : `pyannote.core.Annotation`
        Annotation.
    name : `str`
        Name of the annotation (used for user feedback in case of failure)
    """
    if any(not isinstance(label, str) for label in annotation.labels()):
        msg = f"{name} must contain `str` labels only."
        raise ValueError(msg)


def assert_int_labels(annotation: Annotation, name: str):
    """Check that annotation only contains integer labels

    Parameters
    ----------
    annotation : `pyannote.core.Annotation`
        Annotation.
    name : `str`
        Name of the annotation (used for user feedback in case of failure)
    """
    if any(not isinstance(label, int) for label in annotation.labels()):
        msg = f"{name} must contain `int` labels only."
        raise ValueError(msg)


def load_pretrained_pipeline(train_dir: Path) -> Pipeline:
    """Load pretrained pipeline

    Parameters
    ----------
    train_dir : Path
        Path to training directory (i.e. the one that contains `params.yml`
        created by calling `pyannote-pipeline train ...`)

    Returns
    -------
    pipeline : Pipeline
        Pretrained pipeline
    """

    train_dir = Path(train_dir).expanduser().resolve(strict=True)

    config_yml = train_dir.parents[1] / "config.yml"
    with open(config_yml, "r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    pipeline_name = config["pipeline"]["name"]
    Klass = get_class_by_name(
        pipeline_name, default_module_name="pyannote.audio.pipeline"
    )
    pipeline = Klass(**config["pipeline"].get("params", {}))

    return pipeline.load_params(train_dir / "params.yml")
