#!/usr/bin/env python
# encoding:  utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2018 CNRS

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
# Marvin Lavechin - marvinlavechin@gmail.com

"""
Multi-class classifier BabyTrain

Usage: 
  pyannote-multilabel train [options] <experiment_dir> <database.task.protocol>
  pyannote-multilabel validate [options] [--every=<epoch> --chronological --precision=<precision> --use_der] <label> <train_dir> <database.task.protocol>
  pyannote-multilabel apply [options] [--step=<step>] <model.pt> <database.task.protocol> <output_dir>
  pyannote-multilabel -h | --help
  pyannote-multilabel --version

Common options: 
  <database.task.protocol>   Experimental protocol (e.g. "BabyTrain.SpeakerRole.JSALT")
  --database=<database.yml>        Path to database configuration file.
  --subset=<subset>          Set subset (train|developement|test).
                             Defaults to "train" in "train" mode. Defaults to
                             "development" in "validate" mode. Defaults to all subsets in
                             "apply" mode.
  --gpu                      Run on GPUs. Defaults to using CPUs.
  --batch=<size>             Set batch size. Has no effect in "train" mode.
                             [default:  32]
  --from=<epoch>             Start {train|validat}ing at epoch <epoch>. Has no
                             effect in "apply" mode. [default:  0]
  --to=<epochs>              End {train|validat}ing at epoch <epoch>.
                             Defaults to keep going forever.
"train" mode: 
  <experiment_dir>           Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See "Configuration file"
                             section below for more details.

"validation" mode: 
  --every=<epoch>            Validate model every <epoch> epochs [default:  1].
  --chronological            Force validation in chronological order.
  <label>                    Label to predict (KCHI, CHI, FEM, MAL or speech).
                             If options overlap and speech have been activated during training, one
                             can also to validate on the classes (SPEECH, OVERLAP).
  <train_dir>                Path to the directory containing pre-trained
                             models (i.e. the output of "train" mode).
  --precision=<precision>    Target detection precision [default:  0.8].
  --use_der                  Indicates if the DER should be used for validating the model.

"apply" mode: 
  <model.pt>                 Path to the pretrained model.
  --step=<step>              Sliding window step, in seconds.
                             Defaults to 25% of window duration.

Database configuration file <database.yml>: 
    The database configuration provides details as to where actual files are
    stored. See `pyannote.database.util.FileFinder` docstring for more
    information on the expected format.

Configuration file: 
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml, that describes the feature extraction process,
    the neural network architecture, and the task addressed.

    ................... <experiment_dir>/config.yml ...................
    # train the network for segmentation
    # see pyannote.audio.labeling.tasks for more details
    task: 
       name:  Segmentation
       params: 
          duration:  3.2     # sub-sequence duration
          per_epoch:  1      # 1 day of audio per epoch
          batch_size:  32    # number of sub-sequences per batch

    # use precomputed features (see feature extraction tutorial)
    feature_extraction: 
       name:  Precomputed
       params: 
          root_dir:  tutorials/feature-extraction

    # use the StackedRNN architecture.
    # see pyannote.audio.labeling.models for more details
    architecture: 
       name:  StackedRNN
       params: 
         rnn:  LSTM
         recurrent:  [32, 20]
         bidirectional:  True
         linear:  [40, 10]

    # use cyclic learning rate scheduler
    scheduler: 
       name:  CyclicScheduler
       params: 
           learning_rate:  auto
    ...................................................................

"train" mode: 
    This will create the following directory that contains the pre-trained
    neural network weights after each epoch: 

        <experiment_dir>/train/<database.task.protocol>.<subset>

    This means that the network was trained on the <subset> subset of the
    <database.task.protocol> protocol. By default, <subset> is "train".
    This directory is called <train_dir> in the subsequent "validate" mode.

    A bunch of values (loss, learning rate, ...) are sent to and can be
    visualized with tensorboard with the following command: 

        $ tensorboard --logdir=<experiment_dir>

"validate" mode: 
    Use the "validate" mode to run validation in parallel to training.
    "validate" mode will watch the <train_dir> directory, and run validation
    experiments every time a new epoch has ended. This will create the
    following directory that contains validation results: 

        <train_dir>/validate/<database.task.protocol>.<subset>

    You can run multiple "validate" in parallel (e.g. for every subset,
    protocol, task, or database).

    In practice, for each epoch, "validate" mode will look for the peak
    detection threshold that maximizes speech turn coverage, under the
    constraint that purity must be greater than the value provided by the
    "--purity" option. Both values (best threshold and corresponding coverage)
    are sent to tensorboard.

"apply" mode
    Use the "apply" mode to extract segmentation raw scores.
    Resulting files can then be used in the following way: 

    >>> from pyannote.audio.features import Precomputed
    >>> precomputed = Precomputed('<output_dir>')

    >>> from pyannote.database import get_protocol
    >>> protocol = get_protocol('<database.task.protocol>')
    >>> first_test_file = next(protocol.test())

    >>> from pyannote.audio.signal import Peak
    >>> peak_detection = Peak()

    >>> raw_scores = precomputed(first_test_file)
    >>> homogeneous_segments = peak_detection.apply(raw_scores, dimension=1)
"""

from functools import partial
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import scipy.optimize
from os.path import dirname, basename

import numpy as np
import torch
from docopt import docopt
from pyannote.database import get_annotated
from pyannote.database import get_protocol
from pyannote.database import FileFinder
from pyannote.database import get_unique_identifier

from pyannote.core import Timeline
from pyannote.core import SlidingWindowFeature
from pyannote.core.utils.helper import get_class_by_name


from .base import Application


from pyannote.audio.features import Precomputed
from pyannote.audio.labeling.extraction import SequenceLabeling
from pyannote.audio.features.utils import get_audio_duration

from pyannote.audio.pipeline.speech_activity_detection \
    import SpeechActivityDetection as SpeechActivityDetectionPipeline

from pyannote.audio.pipeline.speaker_activity \
    import SpeakerActivityDetection as SpeakerActivityDetectionPipeline

from pyannote.metrics.detection import DetectionErrorRate

from pyannote.metrics.detection import DetectionRecall
from pyannote.metrics.detection import DetectionPrecision


def validate_helper_func(current_file, pipeline=None, precision=None, recall=None, label=None, metric=None):
    reference = current_file[label]
    hypothesis = pipeline(current_file) # pipeline has been initialized with label, so that it can know which class needs to be assessed
    uem = get_annotated(current_file)

    if precision is not None:
        p = precision(reference, hypothesis, uem=uem)
        r = recall(reference, hypothesis, uem=uem)
        return p, r
    else:
        return metric(reference, hypothesis, uem=uem)


class Multilabel(Application):

    def __init__(self, protocol_name, experiment_dir, db_yml=None, training=False, use_der=False):

        super().__init__(experiment_dir, db_yml=db_yml, training=training)

        def collapse(item):
            dict_map = {
                "BRO": "CHI",
                "C1": "CHI",
                "C2": "CHI",
                "!CHI_0049": "KCHI",
                "!CHI_0396": "KCHI",
                "!CHI_0485": "KCHI",
                "!CHI_0643": "KCHI",
                "!CHI_0713": "KCHI",
                "!CHI_0790": "KCHI",
                "!CHI_1130": "KCHI",
                "!CHI_1156": "KCHI",
                "!CHI_1196": "KCHI",
                "!CHI_1299": "KCHI",
                "!CHI_1499": "KCHI",
                "!CHI_1618": "KCHI",
                "!CHI_1735": "KCHI",
                "!CHI_1768": "KCHI",
                "!CHI_1844": "KCHI",
                "!CHI_2109": "KCHI",
                "!CHI_2224": "KCHI",
                "!CHI_2337": "KCHI",
                "!CHI_2534": "KCHI",
                "!CHI_2535": "KCHI",
                "!CHI_2625": "KCHI",
                "!CHI_2745": "KCHI",
                "!CHI_2811": "KCHI",
                "!CHI_2927": "KCHI",
                "!CHI_3026": "KCHI",
                "!CHI_3090": "KCHI",
                "!CHI_3486": "KCHI",
                "!CHI_3510": "KCHI",
                "!CHI_3528": "KCHI",
                "!CHI_3542": "KCHI",
                "!CHI_3628": "KCHI",
                "!CHI_3634": "KCHI",
                "!CHI_3749": "KCHI",
                "!CHI_3895": "KCHI",
                "!CHI_4483": "KCHI",
                "!CHI_4736": "KCHI",
                "!CHI_4995": "KCHI",
                "!CHI_5223": "KCHI",
                "!CHI_5271": "KCHI",
                "!CHI_5613": "KCHI",
                "!CHI_5750": "KCHI",
                "!CHI_5959": "KCHI",
                "!CHI_6026": "KCHI",
                "!CHI_6035": "KCHI",
                "!CHI_6216": "KCHI",
                "!CHI_7176": "KCHI",
                "!CHI_7220": "KCHI",
                "!CHI_7326": "KCHI",
                "!CHI_7758": "KCHI",
                "!CHI_7798": "KCHI",
                "!CHI_7903": "KCHI",
                "!CHI_8179": "KCHI",
                "!CHI_8340": "KCHI",
                "!CHI_8357": "KCHI",
                "!CHI_8445": "KCHI",
                "!CHI_8560": "KCHI",
                "!CHI_8787": "KCHI",
                "!CHI_8788": "KCHI",
                "!CHI_8924": "KCHI",
                "!CHI_9051": "KCHI",
                "!CHI_9398": "KCHI",
                "!CHI_9408": "KCHI",
                "!CHI_9426": "KCHI",
                "!CHI_9427": "KCHI",
                "!CHI_9492": "KCHI",
                "!CHI_9527": "KCHI",
                "!CHI_9559": "KCHI",
                "!CHI_9733": "KCHI",
                "!CHI_9755": "KCHI",
                "!CHI_9801": "KCHI",
                "!CHI_9854": "KCHI",
                "!CHI_9858": "KCHI",
                "!CHI_9909": "KCHI",
                "!CHI_aiku": "KCHI",
                "!CHI_aoaa": "KCHI",
                "!CHI_C01": "KCHI",
                "!CHI_c01f2a": "KCHI",
                "!CHI_c01f3a": "KCHI",
                "!CHI_c01f4a": "KCHI",
                "!CHI_c01m5a": "KCHI",
                "!CHI_c02f2a": "KCHI",
                "!CHI_c02f3a": "KCHI",
                "!CHI_c02m4a": "KCHI",
                "!CHI_c02m5a": "KCHI",
                "!CHI_c03f2a": "KCHI",
                "!CHI_c03f3a": "KCHI",
                "!CHI_c03m4a": "KCHI",
                "!CHI_c03m5a": "KCHI",
                "!CHI_C04": "KCHI",
                "!CHI_c04f3a": "KCHI",
                "!CHI_c04f4a": "KCHI",
                "!CHI_c04m5a": "KCHI",
                "!CHI_c05f3a": "KCHI",
                "!CHI_c05f4a": "KCHI",
                "!CHI_c05m2a": "KCHI",
                "!CHI_c05m5a": "KCHI",
                "!CHI_c06f3a": "KCHI",
                "!CHI_c06m2a": "KCHI",
                "!CHI_c06m4a": "KCHI",
                "!CHI_c06m5a": "KCHI",
                "!CHI_c07f2a": "KCHI",
                "!CHI_c07f5a": "KCHI",
                "!CHI_c07m3a": "KCHI",
                "!CHI_c07m4a": "KCHI",
                "!CHI_c08m2a": "KCHI",
                "!CHI_c08m4a": "KCHI",
                "!CHI_c08m5a": "KCHI",
                "!CHI_C09": "KCHI",
                "!CHI_c09f2a": "KCHI",
                "!CHI_c09f4a": "KCHI",
                "!CHI_c09f5a": "KCHI",
                "!CHI_c09m3a": "KCHI",
                "!CHI_C10": "KCHI",
                "!CHI_c10f4a": "KCHI",
                "!CHI_c10f5a": "KCHI",
                "!CHI_c10m2a": "KCHI",
                "!CHI_c10m3a": "KCHI",
                "!CHI_C11": "KCHI",
                "!CHI_c11f5a": "KCHI",
                "!CHI_c11m2a": "KCHI",
                "!CHI_C12": "KCHI",
                "!CHI_c12f5a": "KCHI",
                "!CHI_c12m3a": "KCHI",
                "!CHI_c13f3b": "KCHI",
                "!CHI_c13f4b": "KCHI",
                "!CHI_c13f5a": "KCHI",
                "!CHI_c13m2b": "KCHI",
                "!CHI_c14f2b": "KCHI",
                "!CHI_c14f5a": "KCHI",
                "!CHI_c14m3b": "KCHI",
                "!CHI_C15": "KCHI",
                "!CHI_c15f2b": "KCHI",
                "!CHI_c15f3b": "KCHI",
                "!CHI_C16": "KCHI",
                "!CHI_c16f5a": "KCHI",
                "!CHI_c16m2b": "KCHI",
                "!CHI_c16m4b": "KCHI",
                "!CHI_c17f4b": "KCHI",
                "!CHI_c17m3b": "KCHI",
                "!CHI_c17m5b": "KCHI",
                "!CHI_C18": "KCHI",
                "!CHI_c18f2b": "KCHI",
                "!CHI_c18m3b": "KCHI",
                "!CHI_c18m4b": "KCHI",
                "!CHI_c19f3b": "KCHI",
                "!CHI_c19f4b": "KCHI",
                "!CHI_c19m2b": "KCHI",
                "!CHI_C20": "KCHI",
                "!CHI_c20f2b": "KCHI",
                "!CHI_c20f3b": "KCHI",
                "!CHI_c20m4b": "KCHI",
                "!CHI_c21f2b": "KCHI",
                "!CHI_c21m4b": "KCHI",
                "!CHI_C22": "KCHI",
                "!CHI_c22f3b": "KCHI",
                "!CHI_c22f5b": "KCHI",
                "!CHI_c22m4b": "KCHI",
                "!CHI_c23f4b": "KCHI",
                "!CHI_c23f5b": "KCHI",
                "!CHI_c23m2b": "KCHI",
                "!CHI_c23m3b": "KCHI",
                "!CHI_C24": "KCHI",
                "!CHI_c24m2b": "KCHI",
                "!CHI_c24m3b": "KCHI",
                "!CHI_c24m4b": "KCHI",
                "!CHI_c24m5b": "KCHI",
                "!CHI_C25": "KCHI",
                "!CHI_c25m5b": "KCHI",
                "!CHI_CEY": "KCHI",
                "!CHI_e01f2a": "KCHI",
                "!CHI_e01f3a": "KCHI",
                "!CHI_e01f5a": "KCHI",
                "!CHI_e01m4a": "KCHI",
                "!CHI_e02f5a": "KCHI",
                "!CHI_e02m3a": "KCHI",
                "!CHI_e02m4a": "KCHI",
                "!CHI_e03f2a": "KCHI",
                "!CHI_e03f3a": "KCHI",
                "!CHI_e03f4a": "KCHI",
                "!CHI_e03m5a": "KCHI",
                "!CHI_e04f3a": "KCHI",
                "!CHI_e04f5a": "KCHI",
                "!CHI_e04m2a": "KCHI",
                "!CHI_e04m4a": "KCHI",
                "!CHI_e05f4a": "KCHI",
                "!CHI_e05m2a": "KCHI",
                "!CHI_e05m3a": "KCHI",
                "!CHI_e05m5a": "KCHI",
                "!CHI_e06f5a": "KCHI",
                "!CHI_e06m2a": "KCHI",
                "!CHI_e06m4a": "KCHI",
                "!CHI_e07f4a": "KCHI",
                "!CHI_e07m2a": "KCHI",
                "!CHI_e07m3a": "KCHI",
                "!CHI_e07m5a": "KCHI",
                "!CHI_e08f3a": "KCHI",
                "!CHI_e08f5a": "KCHI",
                "!CHI_e08m4a": "KCHI",
                "!CHI_e09f2a": "KCHI",
                "!CHI_e09f5a": "KCHI",
                "!CHI_e09m3a": "KCHI",
                "!CHI_e09m4a": "KCHI",
                "!CHI_e10f4a": "KCHI",
                "!CHI_e10m2a": "KCHI",
                "!CHI_e11m2b": "KCHI",
                "!CHI_e11m3a": "KCHI",
                "!CHI_e11m4a": "KCHI",
                "!CHI_e11m5a": "KCHI",
                "!CHI_e12f2b": "KCHI",
                "!CHI_e12f4a": "KCHI",
                "!CHI_e13f3a": "KCHI",
                "!CHI_e13f4a": "KCHI",
                "!CHI_e13m2b": "KCHI",
                "!CHI_e13m5a": "KCHI",
                "!CHI_e14f2b": "KCHI",
                "!CHI_e14m4b": "KCHI",
                "!CHI_e14m5b": "KCHI",
                "!CHI_e15f5b": "KCHI",
                "!CHI_e16f3b": "KCHI",
                "!CHI_e16f4b": "KCHI",
                "!CHI_e16f5b": "KCHI",
                "!CHI_e16m2b": "KCHI",
                "!CHI_e17f5b": "KCHI",
                "!CHI_e17m3b": "KCHI",
                "!CHI_e18f3b": "KCHI",
                "!CHI_e18f4b": "KCHI",
                "!CHI_e18m2b": "KCHI",
                "!CHI_e19m2b": "KCHI",
                "!CHI_e19m3b": "KCHI",
                "!CHI_e19m5b": "KCHI",
                "!CHI_e20f2b": "KCHI",
                "!CHI_e20f4b": "KCHI",
                "!CHI_e20m3b": "KCHI",
                "!CHI_e21f2b": "KCHI",
                "!CHI_e21m3b": "KCHI",
                "!CHI_e21m4b": "KCHI",
                "!CHI_e21m5b": "KCHI",
                "!CHI_e22f3b": "KCHI",
                "!CHI_e22m2b": "KCHI",
                "!CHI_e22m4b": "KCHI",
                "!CHI_e22m5b": "KCHI",
                "!CHI_e23f2b": "KCHI",
                "!CHI_e23f3b": "KCHI",
                "!CHI_e23f5b": "KCHI",
                "!CHI_e24f2b": "KCHI",
                "!CHI_e24f3b": "KCHI",
                "!CHI_e24f4b": "KCHI",
                "!CHI_e25f5b": "KCHI",
                "!CHI_e25m3b": "KCHI",
                "!CHI_e25m4b": "KCHI",
                "!CHI_eiun": "KCHI",
                "!CHI_eoak": "KCHI",
                "!CHI_eoxk": "KCHI",
                "!CHI_ern": "KCHI",
                "!CHI_fhugo": "KCHI",
                "!CHI_flore": "KCHI",
                "!CHI_g01f2a": "KCHI",
                "!CHI_g01f5a": "KCHI",
                "!CHI_g01m3a": "KCHI",
                "!CHI_g01m4a": "KCHI",
                "!CHI_g02f2a": "KCHI",
                "!CHI_g02f4a": "KCHI",
                "!CHI_g02m3a": "KCHI",
                "!CHI_g02m5a": "KCHI",
                "!CHI_g03f2a": "KCHI",
                "!CHI_g03f5a": "KCHI",
                "!CHI_g04f5a": "KCHI",
                "!CHI_g04m2a": "KCHI",
                "!CHI_g04m3a": "KCHI",
                "!CHI_g04m4a": "KCHI",
                "!CHI_g05f2a": "KCHI",
                "!CHI_g05f3a": "KCHI",
                "!CHI_g05f4a": "KCHI",
                "!CHI_g05m5a": "KCHI",
                "!CHI_g06f4a": "KCHI",
                "!CHI_g06f5a": "KCHI",
                "!CHI_g07f3a": "KCHI",
                "!CHI_g07m2a": "KCHI",
                "!CHI_g07m4a": "KCHI",
                "!CHI_g08f2a": "KCHI",
                "!CHI_g08f5a": "KCHI",
                "!CHI_g08m3a": "KCHI",
                "!CHI_g09f5a": "KCHI",
                "!CHI_g09m2a": "KCHI",
                "!CHI_g09m4a": "KCHI",
                "!CHI_g10f3a": "KCHI",
                "!CHI_g10m2a": "KCHI",
                "!CHI_g10m4a": "KCHI",
                "!CHI_g10m5a": "KCHI",
                "!CHI_g11f2a": "KCHI",
                "!CHI_g11f3a": "KCHI",
                "!CHI_g11f4b": "KCHI",
                "!CHI_g12f3a": "KCHI",
                "!CHI_g12m5a": "KCHI",
                "!CHI_g13f2b": "KCHI",
                "!CHI_g13f4b": "KCHI",
                "!CHI_g13m3a": "KCHI",
                "!CHI_g14f4b": "KCHI",
                "!CHI_g14m2b": "KCHI",
                "!CHI_g14m3b": "KCHI",
                "!CHI_g15f3b": "KCHI",
                "!CHI_g15f4b": "KCHI",
                "!CHI_g16f5b": "KCHI",
                "!CHI_g16m2b": "KCHI",
                "!CHI_g16m4b": "KCHI",
                "!CHI_g17f2b": "KCHI",
                "!CHI_g17m3b": "KCHI",
                "!CHI_g17m5b": "KCHI",
                "!CHI_g18f2b": "KCHI",
                "!CHI_g18f3b": "KCHI",
                "!CHI_g18f5b": "KCHI",
                "!CHI_g18m4b": "KCHI",
                "!CHI_g19f3b": "KCHI",
                "!CHI_g19m2b": "KCHI",
                "!CHI_g19m4b": "KCHI",
                "!CHI_g19m5b": "KCHI",
                "!CHI_g20m3b": "KCHI",
                "!CHI_g20m5b": "KCHI",
                "!CHI_g21f2b": "KCHI",
                "!CHI_g21f5b": "KCHI",
                "!CHI_g21m4b": "KCHI",
                "!CHI_g22f3b": "KCHI",
                "!CHI_g22m4b": "KCHI",
                "!CHI_g22m5b": "KCHI",
                "!CHI_g23f4b": "KCHI",
                "!CHI_g23m2b": "KCHI",
                "!CHI_g23m3b": "KCHI",
                "!CHI_g23m5b": "KCHI",
                "!CHI_g24f4b": "KCHI",
                "!CHI_g24f5b": "KCHI",
                "!CHI_g24m3b": "KCHI",
                "!CHI_g25f3b": "KCHI",
                "!CHI_g25f4b": "KCHI",
                "!CHI_g25m2b": "KCHI",
                "!CHI_g25m5b": "KCHI",
                "!CHI_g26m2b": "KCHI",
                "!CHI_gust": "KCHI",
                "!CHI_HBL": "KCHI",
                "!CHI_HYS": "KCHI",
                "!CHI_j01f2a": "KCHI",
                "!CHI_j01f3a": "KCHI",
                "!CHI_j01f4a": "KCHI",
                "!CHI_j01m5a": "KCHI",
                "!CHI_j02f3a": "KCHI",
                "!CHI_j02f5a": "KCHI",
                "!CHI_j02m2a": "KCHI",
                "!CHI_j02m4a": "KCHI",
                "!CHI_j03f3a": "KCHI",
                "!CHI_j03f4a": "KCHI",
                "!CHI_j03f5a": "KCHI",
                "!CHI_j04f2a": "KCHI",
                "!CHI_j04f3a": "KCHI",
                "!CHI_j04f4a": "KCHI",
                "!CHI_j04f5a": "KCHI",
                "!CHI_j05f2a": "KCHI",
                "!CHI_j05f4a": "KCHI",
                "!CHI_j05f5a": "KCHI",
                "!CHI_j05m3a": "KCHI",
                "!CHI_j06f2a": "KCHI",
                "!CHI_j06f3a": "KCHI",
                "!CHI_j06f5a": "KCHI",
                "!CHI_j06m4a": "KCHI",
                "!CHI_j07f3a": "KCHI",
                "!CHI_j07f5a": "KCHI",
                "!CHI_j07m2a": "KCHI",
                "!CHI_j08f4a": "KCHI",
                "!CHI_j08m2a": "KCHI",
                "!CHI_j08m5a": "KCHI",
                "!CHI_j09m4a": "KCHI",
                "!CHI_j09m5a": "KCHI",
                "!CHI_j10m3a": "KCHI",
                "!CHI_j10m5a": "KCHI",
                "!CHI_j11f3a": "KCHI",
                "!CHI_j11m2a": "KCHI",
                "!CHI_j11m4a": "KCHI",
                "!CHI_j12m2a": "KCHI",
                "!CHI_j12m3a": "KCHI",
                "!CHI_j12m4a": "KCHI",
                "!CHI_j13f2a": "KCHI",
                "!CHI_j13m3b": "KCHI",
                "!CHI_j13m4a": "KCHI",
                "!CHI_j14m2a": "KCHI",
                "!CHI_j14m4a": "KCHI",
                "!CHI_j15f2b": "KCHI",
                "!CHI_j15f4a": "KCHI",
                "!CHI_j15f5a": "KCHI",
                "!CHI_j15m3b": "KCHI",
                "!CHI_j16f5b": "KCHI",
                "!CHI_j16m2b": "KCHI",
                "!CHI_j16m3b": "KCHI",
                "!CHI_j17f3b": "KCHI",
                "!CHI_j17m4b": "KCHI",
                "!CHI_j17m5b": "KCHI",
                "!CHI_j18m2b": "KCHI",
                "!CHI_j18m3b": "KCHI",
                "!CHI_j19f2b": "KCHI",
                "!CHI_j19f3b": "KCHI",
                "!CHI_j19m4b": "KCHI",
                "!CHI_j19m5b": "KCHI",
                "!CHI_j20f2b": "KCHI",
                "!CHI_j20f3b": "KCHI",
                "!CHI_j20f4b": "KCHI",
                "!CHI_j20m5b": "KCHI",
                "!CHI_j21f2b": "KCHI",
                "!CHI_j21f4b": "KCHI",
                "!CHI_j21m3b": "KCHI",
                "!CHI_j21m5b": "KCHI",
                "!CHI_j22f3b": "KCHI",
                "!CHI_j22f5b": "KCHI",
                "!CHI_j22m4b": "KCHI",
                "!CHI_j23f2b": "KCHI",
                "!CHI_j23f4b": "KCHI",
                "!CHI_j23m3b": "KCHI",
                "!CHI_j24f3b": "KCHI",
                "!CHI_j24f4b": "KCHI",
                "!CHI_j24m2b": "KCHI",
                "!CHI_j24m5b": "KCHI",
                "!CHI_j25m2b": "KCHI",
                "!CHI_j25m3b": "KCHI",
                "!CHI_j25m5b": "KCHI",
                "!CHI_j26m4b": "KCHI",
                "!CHI_j27f4b": "KCHI",
                "!CHI_j28f4b": "KCHI",
                "!CHI_j29m4b": "KCHI",
                "!CHI_leon": "KCHI",
                "!CHI_LMC": "KCHI",
                "!CHI_LWJ": "KCHI",
                "!CHI_LYC": "KCHI",
                "!CHI_marin": "KCHI",
                "!CHI_nath": "KCHI",
                "!CHI_nohlan": "KCHI",
                "!CHI_noxk": "KCHI",
                "!CHI_noxt": "KCHI",
                "!CHI_oegd": "KCHI",
                "!CHI_oekd": "KCHI",
                "!CHI_outg": "KCHI",
                "!CHI_sacha": "KCHI",
                "!CHI_TWX": "KCHI",
                "!CHI_uebn": "KCHI",
                "!CHI_uoga": "KCHI",
                "!CHI_va10": "KCHI",
                "!CHI_va12": "KCHI",
                "!CHI_va14": "KCHI",
                "!CHI_va18": "KCHI",
                "!CHI_van4": "KCHI",
                "!CHI_van5": "KCHI",
                "!CHI_van6": "KCHI",
                "!CHI_van7": "KCHI",
                "!CHI_van8": "KCHI",
                "!CHI_WW04": "KCHI",
                "!CHI_WW15": "KCHI",
                "!CHI_WW18": "KCHI",
                "!CHI_WW25": "KCHI",
                "EE1": "SIL",
                "FA1": "FEM",
                "!FA1_0643": "FEM",
                "!FA1_2109": "FEM",
                "!FA1_2625": "FEM",
                "!FA1_3026": "FEM",
                "!FA1_6216": "FEM",
                "!FA1_7176": "FEM",
                "!FA1_7220": "FEM",
                "!FA1_7326": "FEM",
                "!FA1_8179": "FEM",
                "!FA1_8787": "FEM",
                "FA2": "FEM",
                "!FA2_0643": "FEM",
                "!FA2_2625": "FEM",
                "!FA2_6216": "FEM",
                "!FA2_7176": "FEM",
                "!FA2_7220": "FEM",
                "!FA2_7326": "FEM",
                "!FA2_8179": "FEM",
                "!FA2_8787": "FEM",
                "FA3": "FEM",
                "!FA3_2625": "FEM",
                "!FA3_6216": "FEM",
                "!FA3_7176": "FEM",
                "!FA3_7220": "FEM",
                "!FA3_7326": "FEM",
                "!FA3_8179": "FEM",
                "!FA3_8787": "FEM",
                "FA4": "FEM",
                "!FA4_2625": "FEM",
                "!FA4_6216": "FEM",
                "!FA4_7176": "FEM",
                "!FA4_7220": "FEM",
                "!FA4_7326": "FEM",
                "!FA4_8179": "FEM",
                "FA5": "FEM",
                "!FA5_7326": "FEM",
                "!FA5_8179": "FEM",
                "FA6": "FEM",
                "!FA6_7326": "FEM",
                "FA7": "FEM",
                "FA8": "FEM",
                "FAE": "SIL",
                "!FAT_ern": "MAL",
                "!FAT_fhugo": "MAL",
                "!FAT_flore": "MAL",
                "!FAT_gust": "MAL",
                "!FAT_HBL": "MAL",
                "!FAT_leon": "MAL",
                "!FAT_marin": "MAL",
                "!FAT_nath": "MAL",
                "!FAT_nohlan": "MAL",
                "!FAT_sacha": "MAL",
                "!FAT_TWX": "MAL",
                "FC1": "CHI",
                "!FC1_0643": "CHI",
                "!FC1_2625": "CHI",
                "!FC1_3026": "CHI",
                "!FC1_7176": "CHI",
                "!FC1_7220": "CHI",
                "!FC1_7326": "CHI",
                "!FC1_8179": "CHI",
                "!FC1_8787": "CHI",
                "FC2": "CHI",
                "!FC2_0643": "CHI",
                "!FC2_2625": "CHI",
                "!FC2_3026": "CHI",
                "!FC2_7176": "CHI",
                "!FC2_7326": "CHI",
                "!FC2_8179": "CHI",
                "FC3": "CHI",
                "!FC3_0643": "CHI",
                "!FC3_3026": "CHI",
                "!FC3_7176": "CHI",
                "!FC3_7326": "CHI",
                "!FC3_8179": "CHI",
                "!FC4_3026": "CHI",
                "!FC4_8179": "CHI",
                "FEM": "FEM",
                "GRF": "MAL",
                "GRM": "FEM",
                "!INV_Joyce": "FEM",
                "!INV_Kay": "FEM",
                "!INV_Rose": "FEM",
                "MA1": "MAL",
                "!MA1_0643": "MAL",
                "!MA1_2109": "MAL",
                "!MA1_2625": "MAL",
                "!MA1_3026": "MAL",
                "!MA1_6216": "MAL",
                "!MA1_7176": "MAL",
                "!MA1_7220": "MAL",
                "!MA1_7326": "MAL",
                "!MA1_8179": "MAL",
                "!MA1_8787": "MAL",
                "MA2": "MAL",
                "!MA2_2625": "MAL",
                "!MA2_6216": "MAL",
                "!MA2_7220": "MAL",
                "!MA2_7326": "MAL",
                "!MA2_8179": "MAL",
                "!MA2_8787": "MAL",
                "MA3": "MAL",
                "!MA3_7326": "MAL",
                "!MA3_8179": "MAL",
                "MA4": "MAL",
                "!MA4_7326": "MAL",
                "!MA4_8179": "MAL",
                "MA5": "MAL",
                "!MA5_7326": "MAL",
                "!MA5_8179": "MAL",
                "!MA6_8179": "MAL",
                "MAE": "SIL",
                "MAL": "MAL",
                "MC1": "CHI",
                "!MC1_0643": "CHI",
                "!MC1_2625": "CHI",
                "!MC1_6216": "CHI",
                "!MC1_7220": "CHI",
                "!MC1_8179": "CHI",
                "!MC1_8787": "CHI",
                "MC2": "CHI",
                "!MC2_2625": "CHI",
                "!MC2_7220": "CHI",
                "!MC2_8179": "CHI",
                "!MC2_8787": "CHI",
                "MC3": "CHI",
                "!MC3_7220": "CHI",
                "!MC4_7220": "CHI",
                "!MC5_7220": "CHI",
                "!MC6_7220": "CHI",
                "MI1": "MAL",
                "MOT*": "FEM",
                "!MOT_0049": "FEM",
                "!MOT_0396": "FEM",
                "!MOT_0485": "FEM",
                "!MOT_0643": "FEM",
                "!MOT_0790": "FEM",
                "!MOT_1130": "FEM",
                "!MOT_1156": "FEM",
                "!MOT_1196": "FEM",
                "!MOT_1299": "FEM",
                "!MOT_1499": "FEM",
                "!MOT_1618": "FEM",
                "!MOT_1735": "FEM",
                "!MOT_1768": "FEM",
                "!MOT_1844": "FEM",
                "!MOT_2224": "FEM",
                "!MOT_2337": "FEM",
                "!MOT_2534": "FEM",
                "!MOT_2535": "FEM",
                "!MOT_2625": "FEM",
                "!MOT_2745": "FEM",
                "!MOT_2811": "FEM",
                "!MOT_2927": "FEM",
                "!MOT_3090": "FEM",
                "!MOT_3486": "FEM",
                "!MOT_3510": "FEM",
                "!MOT_3542": "FEM",
                "!MOT_3634": "FEM",
                "!MOT_3749": "FEM",
                "!MOT_3895": "FEM",
                "!MOT_4483": "FEM",
                "!MOT_4736": "FEM",
                "!MOT_4995": "FEM",
                "!MOT_5223": "FEM",
                "!MOT_5271": "FEM",
                "!MOT_5613": "FEM",
                "!MOT_5750": "FEM",
                "!MOT_5959": "FEM",
                "!MOT_6026": "FEM",
                "!MOT_6035": "FEM",
                "!MOT_7758": "FEM",
                "!MOT_7798": "FEM",
                "!MOT_7903": "FEM",
                "!MOT_8340": "FEM",
                "!MOT_8357": "FEM",
                "!MOT_8445": "FEM",
                "!MOT_8560": "FEM",
                "!MOT_8787": "FEM",
                "!MOT_8788": "FEM",
                "!MOT_8924": "FEM",
                "!MOT_9051": "FEM",
                "!MOT_9398": "FEM",
                "!MOT_9408": "FEM",
                "!MOT_9426": "FEM",
                "!MOT_9427": "FEM",
                "!MOT_9492": "FEM",
                "!MOT_9527": "FEM",
                "!MOT_9559": "FEM",
                "!MOT_9733": "FEM",
                "!MOT_9755": "FEM",
                "!MOT_9801": "FEM",
                "!MOT_9854": "FEM",
                "!MOT_9858": "FEM",
                "!MOT_9909": "FEM",
                "!MOT_ern": "FEM",
                "!MOT_fhugo": "FEM",
                "!MOT_flore": "FEM",
                "!MOT_gust": "FEM",
                "!MOT_HBL": "FEM",
                "!MOT_HYS": "FEM",
                "!MOT_leon": "FEM",
                "!MOT_LWJ": "FEM",
                "!MOT_LYC": "FEM",
                "!MOT_marin": "FEM",
                "!MOT_nath": "FEM",
                "!MOT_nohlan": "FEM",
                "!MOT_sacha": "FEM",
                "!MOT_TWX": "FEM",
                "OCH": "SPEECH",
                "OTH": "SPEECH",
                "SIS":  "CHI",
                "UA1": "SPEECH",
                "UC1": "CHI",
                "!UC1_0643": "CHI",
                "!UC1_2625": "CHI",
                "!UC1_6216": "CHI",
                "!UC1_7220": "CHI",
                "!UC1_7326": "CHI",
                "!UC1_8179": "CHI",
                "UC2": "CHI",
                "!UC2_2625": "CHI",
                "!UC2_7326": "CHI",
                "!UC2_8179": "CHI",
                "UC3": "CHI",
                "!UC3_2625": "CHI",
                "!UC3_8179": "CHI",
                "UC4": "CHI",
                "!UC4_8179": "CHI",
                "UC5": "CHI",
                "!UC5_8179": "CHI",
                "UC6": "CHI",
                "UU": "SPEECH",
                "!UU1_2625": "SPEECH",
                "!UU1_6216": "SPEECH",
                "FEE005": "FEM",
                "FEE013": "FEM",
                "FEE016": "FEM",
                "FEE019": "FEM",
                "FEE021": "FEM",
                "FEE024": "FEM",
                "FEE028": "FEM",
                "FEE029": "FEM",
                "FEE030": "FEM",
                "FEE032": "FEM",
                "FEE036": "FEM",
                "FEE037": "FEM",
                "FEE038": "FEM",
                "FEE039": "FEM",
                "FEE040": "FEM",
                "FEE041": "FEM",
                "FEE042": "FEM",
                "FEE043": "FEM",
                "FEE044": "FEM",
                "FEE046": "FEM",
                "FEE047": "FEM",
                "FEE049": "FEM",
                "FEE050": "FEM",
                "FEE051": "FEM",
                "FEE052": "FEM",
                "FEE055": "FEM",
                "FEE057": "FEM",
                "FEE058": "FEM",
                "FEE059": "FEM",
                "FEE060": "FEM",
                "FEE064": "FEM",
                "FEE078": "FEM",
                "FEE080": "FEM",
                "FEE081": "FEM",
                "FEE083": "FEM",
                "FEE085": "FEM",
                "FEE087": "FEM",
                "FEE088": "FEM",
                "FEE096": "FEM",
                "FEO023": "FEM",
                "FEO026": "FEM",
                "FEO065": "FEM",
                "FEO066": "FEM",
                "FEO070": "FEM",
                "FEO072": "FEM",
                "FEO079": "FEM",
                "FEO084": "FEM",
                "FIE037": "FEM",
                "FIE038": "FEM",
                "FIE073": "FEM",
                "FIE081": "FEM",
                "FIE088": "FEM",
                "FIO017": "FEM",
                "FIO041": "FEM",
                "FIO074": "FEM",
                "FIO084": "FEM",
                "FIO087": "FEM",
                "FIO089": "FEM",
                "FIO093": "FEM",
                "FTD019UID": "FEM",
                "MEE006": "MAL",
                "MEE007": "MAL",
                "MEE008": "MAL",
                "MEE009": "MAL",
                "MEE010": "MAL",
                "MEE011": "MAL",
                "MEE012": "MAL",
                "MEE014": "MAL",
                "MEE017": "MAL",
                "MEE018": "MAL",
                "MEE025": "MAL",
                "MEE027": "MAL",
                "MEE031": "MAL",
                "MEE033": "MAL",
                "MEE034": "MAL",
                "MEE035": "MAL",
                "MEE045": "MAL",
                "MEE048": "MAL",
                "MEE053": "MAL",
                "MEE054": "MAL",
                "MEE056": "MAL",
                "MEE061": "MAL",
                "MEE063": "MAL",
                "MEE067": "MAL",
                "MEE068": "MAL",
                "MEE071": "MAL",
                "MEE073": "MAL",
                "MEE075": "MAL",
                "MEE076": "MAL",
                "MEE089": "MAL",
                "MEE094": "MAL",
                "MEE095": "MAL",
                "MEO015": "MAL",
                "MEO020": "MAL",
                "MEO022": "MAL",
                "MEO062": "MAL",
                "MEO069": "MAL",
                "MEO074": "MAL",
                "MEO082": "MAL",
                "MEO086": "MAL",
                "MIE002": "MAL",
                "MIE029": "MAL",
                "MIE034": "MAL",
                "MIE080": "MAL",
                "MIE083": "MAL",
                "MIE085": "MAL",
                "MIE090": "MAL",
                "MIO005": "MAL",
                "MIO008": "MAL",
                "MIO012": "MAL",
                "MIO016": "MAL",
                "MIO018": "MAL",
                "MIO019": "MAL",
                "MIO020": "MAL",
                "MIO022": "MAL",
                "MIO023": "MAL",
                "MIO024": "MAL",
                "MIO025": "MAL",
                "MIO026": "MAL",
                "MIO031": "MAL",
                "MIO035": "MAL",
                "MIO036": "MAL",
                "MIO039": "MAL",
                "MIO040": "MAL",
                "MIO043": "MAL",
                "MIO046": "MAL",
                "MIO047": "MAL",
                "MIO049": "MAL",
                "MIO050": "MAL",
                "MIO055": "MAL",
                "MIO066": "MAL",
                "MIO072": "MAL",
                "MIO075": "MAL",
                "MIO076": "MAL",
                "MIO077": "MAL",
                "MIO078": "MAL",
                "MIO082": "MAL",
                "MIO086": "MAL",
                "MIO091": "MAL",
                "MIO092": "MAL",
                "MIO094": "MAL",
                "MIO095": "MAL",
                "MIO097": "MAL",
                "MIO098": "MAL",
                "MIO099": "MAL",
                "MIO100": "MAL",
                "MIO101": "MAL",
                "MIO104": "MAL",
                "MIO105": "MAL",
                "MIO106": "MAL",
                "MTD0010ID": "MAL",
                "MTD009PM": "MAL",
                "MTD011UID": "MAL",
                "MTD012ME": "MAL",
                "MTD013PM": "MAL",
                "MTD014ID": "MAL",
                "MTD015UID": "MAL",
                "MTD016ME": "MAL",
                "MTD017PM": "MAL",
                "MTD018ID": "MAL",
                "MTD020ME": "MAL",
                "MTD021PM": "MAL",
                "MTD022ID": "MAL",
                "MTD023UID": "MAL",
                "MTD024ME": "MAL",
                "MTD025PM": "MAL",
                "MTD026UID": "MAL",
                "MTD027ID": "MAL",
                "MTD028ME": "MAL",
                "MTD029PM": "MAL",
                "MTD030ID": "MAL",
                "MTD031UID": "MAL",
                "MTD032ME": "MAL",
                "MTD033PM": "MAL",
                "MTD034ID": "MAL",
                "MTD035UID": "MAL",
                "MTD036ME": "MAL",
                "MTD037PM": "MAL",
                "MTD038ID": "MAL",
                "MTD039UID": "MAL",
                "MTD040ME": "MAL",
                "MTD041PM": "MAL",
                "MTD042ID": "MAL",
                "MTD043UID": "MAL",
                "MTD044ME": "MAL",
                "MTD045PM": "MAL",
                "MTD046ID": "MAL",
                "MTD047UID": "MAL",
                "MTD048ME": "MAL",
                "P01": "FEM",
                "P02": "FEM",
                "P03": "MAL",
                "P04": "FEM",
                "P05": "FEM",
                "P06": "MAL",
                "P07": "MAL",
                "P08": "FEM",
                "P09": "MAL",
                "P10": "MAL",
                "P11": "MAL",
                "P12": "MAL",
                "P13": "MAL",
                "P14": "FEM",
                "P15": "FEM",
                "P16": "MAL",
                "P17": "FEM",
                "P18": "MAL",
                "P19": "FEM",
                "P20": "MAL",
                "P21": "MAL",
                "P22": "MAL",
                "P23": "MAL",
                "P24": "MAL",
                "P25": "FEM",
                "P26": "FEM",
                "P27": "FEM",
                "P28": "FEM",
                "P33": "MAL",
                "P34": "MAL",
                "P35": "MAL",
                "P36": "FEM",
                "P41": "FEM",
                "P42": "MAL",
                "P43": "FEM",
                "P44": "FEM",
                "P45": "MAL",
                "P46": "FEM",
                "P47": "MAL",
                "P48": "FEM",
                "P49": "FEM",
                "P50": "MAL",
                "P51": "MAL",
                "P52": "FEM",
                "P53": "FEM",
                "P54": "MAL",
                "P55": "MAL",
                "P56": "FEM"
            }

            # If model must predict "SPEECH", then all the speakers who are not
            # classified as being [KCHI,CHI,FEM,MAL] are classified as being "SPEECH".
            # If not, these classes are classified as being "SIL".
            if not ("speech" in self.config_['task'].get('params', {}).keys() and
                    self.config_['task'].get('params', {})["speech"]):
                for key, value in dict_map.items():
                    if value == "SPEECH":
                        dict_map[key] = "SIL"

            for segment, track, label in item["annotation"].itertracks(yield_label=True):
                if label in dict_map.keys():
                    item["annotation"][segment, track] = dict_map[label]
                else:
                    raise ValueError("No mapping found for %s" % label)

            # Extract all but SIL class
            item["annotation"] = item["annotation"].subset(["SIL"], invert=True)
            return item["annotation"]

        # Careful because protocol.trn_iter() won't apply this preprocessor
        # However, protocol.train() will
        self.preprocessors_['annotation'] = collapse

        self.use_der = use_der

        # task
        Task = get_class_by_name(
            self.config_['task']['name'],
            default_module_name='pyannote.audio.labeling.tasks')

        self.task_ = Task(
            protocol_name,
            preprocessors=self.preprocessors_,
            **self.config_['task'].get('params', {}))

        n_features = int(self.feature_extraction_.dimension)
        n_classes = self.task_.n_classes
        task_type = self.task_.task_type

        # architecture
        Architecture = get_class_by_name(
            self.config_['architecture']['name'],
            default_module_name='pyannote.audio.labeling.models')
        self.model_ = Architecture(
            n_features, n_classes, task_type,
            **self.config_['architecture'].get('params', {}))


    @classmethod
    def from_train_dir(cls, protocol_name, train_dir, db_yml=None, training=False, use_der=False):
        experiment_dir = dirname(dirname(train_dir))
        app = cls(protocol_name, experiment_dir, db_yml=db_yml, training=training, use_der=use_der)
        app.train_dir_ = train_dir
        return app

    @classmethod
    def from_model_pt(cls, protocol_name, model_pt, db_yml=None, training=False, use_der=False):
        train_dir = dirname(dirname(model_pt))
        app = cls.from_train_dir(protocol_name, train_dir, db_yml=db_yml, training=training, use_der=use_der)
        app.model_pt_ = model_pt
        epoch = int(basename(app.model_pt_)[:-3])
        app.model_ = app.load_model(epoch, train_dir=train_dir)
        return app

    def validate_init(self, protocol_name, subset='development'): 

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)
        files = getattr(protocol, subset)()

        self.pool_ = mp.Pool(mp.cpu_count())

        # pre-compute features for each validation files
        validation_data = []
        for current_file in tqdm(files, desc='Feature extraction'): 

            # precompute features
            if not isinstance(self.feature_extraction_, Precomputed): 
                current_file['features'] = self.feature_extraction_(
                    current_file)

            if self.label == "SPEECH":
                # all the speakers
                current_file[self.label] = current_file['annotation']
            elif self.label in ["CHI", "FEM", "KCHI", "MAL"]:
                reference = current_file['annotation']
                label_speech = reference.subset([self.label])
                current_file[self.label] = label_speech
            elif self.label == "OVERLAP":
                # build overlap reference
                uri = current_file['uri']
                overlap = Timeline(uri=uri)
                turns = current_file['annotation']
                for track1, track2 in turns.co_iter(turns):
                    if track1 == track2:
                        continue
                    overlap.add(track1[0] & track2[0])
                current_file["OVERLAP"] = overlap.support().to_annotation()
            validation_data.append(current_file)
        return validation_data

    def validate_epoch(self, epoch, protocol_name, subset='development',
                       validation_data=None): 
        func = getattr(self, f'validate_epoch_class')
        return func(epoch, protocol_name, subset=subset,
                    validation_data=validation_data)

    def validate_epoch_class(self, epoch, protocol_name, subset='development',
                              validation_data=None): 
        """
        Validate function given a class which must belongs to ["CHI", "FEM", "KCHI", "MAL", "SPEECH", "OVERLAP"]
        """
        # Name of the class that needs to be validated
        class_name = self.label

        # load model for current epoch
        model = self.load_model(epoch).to(self.device)
        model.eval()

        # compute (and store) SAD scores
        duration = self.task_.duration
        step = .25 * duration

        sequence_labeling = SequenceLabeling(
            model=model, feature_extraction=self.feature_extraction_,
            duration=duration, step=step, batch_size=self.batch_size,
            device=self.device)

        for current_file in validation_data:
            scores = sequence_labeling(current_file)
            if class_name == "SPEECH" and "SPEECH" not in self.task_.labels:
                # We sum up all the scores of every speakers
                scores_data = np.sum(scores.data, axis=1).reshape(-1, 1)
            else: 
                # We extract the score of interest
                dimension = self.task_.labels.index(class_name)
                scores_data = scores.data[:, dimension].reshape(-1, 1)

            current_file[class_name+'_scores'] = SlidingWindowFeature(
                scores_data,
                scores.sliding_window)

        # pipeline
        pipeline = SpeakerActivityDetectionPipeline(label=self.label, use_der=self.use_der)

        lower_alpha = 0.
        upper_alpha = 1.
        best_alpha = .5 * (lower_alpha + upper_alpha)
        best_recall = 0.

        if not self.use_der:
            for _ in range(10):

                current_alpha = .5 * (lower_alpha + upper_alpha)
                pipeline.instantiate({'onset':  current_alpha,
                                      'offset':  current_alpha,
                                      'min_duration_on':  0.,
                                      'min_duration_off':  0.,
                                      'pad_onset':  0.,
                                      'pad_offset':  0.})

                precision = DetectionPrecision(parallel=True)
                recall = DetectionRecall(parallel=True)

                validate = partial(validate_helper_func,
                                   pipeline=pipeline,
                                   precision=precision,
                                   recall=recall,
                                   label=self.label)
                _ = self.pool_.map(validate, validation_data)

                precision = abs(precision)
                recall = abs(recall)

                if precision < self.precision:
                    # precision is not high enough:  try higher thresholds
                    lower_alpha = current_alpha

                else:
                    upper_alpha = current_alpha
                    if recall > best_recall:
                        best_recall = recall
                        best_alpha = current_alpha

            return {'metric':  f'recall@{self.precision: .2f}precision',
                    'minimize':  False,
                    'value':  best_recall,
                    'pipeline':  pipeline.instantiate({'onset':  best_alpha,
                                                      'offset':  best_alpha,
                                                      'min_duration_on':  0.,
                                                      'min_duration_off':  0.,
                                                      'pad_onset':  0.,
                                                      'pad_offset':  0.})}
        else:
            def fun(threshold):
                pipeline.instantiate({'onset': threshold,
                                      'offset': threshold,
                                      'min_duration_on': 0.,
                                      'min_duration_off': 0.,
                                      'pad_onset': 0.,
                                      'pad_offset': 0.})
                metric = DetectionErrorRate(parallel=True)
                validate = partial(validate_helper_func,
                                   pipeline=pipeline,
                                   label=self.label,
                                   metric=metric)
                _ = self.pool_.map(validate, validation_data)

                return abs(metric)

            res = scipy.optimize.minimize_scalar(
                fun, bounds=(0., 1.), method='bounded', options={'maxiter': 10})

            threshold = res.x.item()

            return {'metric': 'detection_error_rate',
                    'minimize': True,
                    'value': res.fun,
                    'pipeline': pipeline.instantiate({'onset': threshold,
                                                      'offset': threshold,
                                                      'min_duration_on': 0.,
                                                      'min_duration_off': 0.,
                                                      'pad_onset': 0.,
                                                      'pad_offset': 0.})}

    def apply(self, protocol_name, output_dir, step=None, subset=None): 

        model = self.model_.to(self.device)
        model.eval()

        duration = self.task_.duration
        if step is None: 
            step = 0.25 * duration

        # do not use memmap as this would lead to too many open files
        if isinstance(self.feature_extraction_, Precomputed): 
            self.feature_extraction_.use_memmap = False

        # initialize embedding extraction
        sequence_labeling = SequenceLabeling(
            model=model, feature_extraction=self.feature_extraction_,
            duration=duration, step=step, batch_size=self.batch_size,
            device=self.device)

        sliding_window = sequence_labeling.sliding_window
        n_classes = self.task_.n_classes
        labels = self.task_.labels

        # create metadata file at root that contains
        # sliding window and dimension information
        precomputed = Precomputed(
            root_dir=output_dir,
            sliding_window=sliding_window,
            dimension=n_classes,
            labels=labels)

        # file generator
        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)

        if subset is None: 
            files = FileFinder.protocol_file_iter(protocol,
                                                  extra_keys=['audio'])
        else: 
            files = getattr(protocol, subset)()

        for current_file in files: 
            fX = sequence_labeling(current_file)
            precomputed.dump(current_file, fX)


def main(): 
    arguments = docopt(__doc__, version='Multilabel')
    db_yml = arguments['--database']
    protocol_name = arguments['<database.task.protocol>']
    subset = arguments['--subset']
    gpu = arguments['--gpu']
    device = torch.device('cuda') if gpu else torch.device('cpu')
    use_der = arguments['--use_der']

    if arguments['train']:
        experiment_dir = Path(arguments['<experiment_dir>'])
        experiment_dir = experiment_dir.expanduser().resolve(strict=True)

        if subset is None: 
            subset = 'train'

        # start training at this epoch (defaults to 0)
        restart = int(arguments['--from'])

        # stop training at this epoch (defaults to never stop)
        epochs = arguments['--to']
        if epochs is None: 
            epochs = np.inf
        else: 
            epochs = int(epochs)

        application = Multilabel(protocol_name, experiment_dir, db_yml=db_yml,
                                             training=True)
        application.device = device
        application.train(protocol_name, subset=subset,
                          restart=restart, epochs=epochs)

    if arguments['validate']: 
        label = arguments['<label>']
        precision = float(arguments['--precision'])
        train_dir = Path(arguments['<train_dir>'])
        train_dir = train_dir.expanduser().resolve(strict=True)

        if subset is None: 
            subset = 'development'

        # start validating at this epoch (defaults to 0)
        start = int(arguments['--from'])

        # stop validating at this epoch (defaults to np.inf)
        end = arguments['--to']
        if end is None: 
            end = np.inf
        else: 
            end = int(end)

        # validate every that many epochs (defaults to 1)
        every = int(arguments['--every'])

        # validate epochs in chronological order
        in_order = arguments['--chronological']

        # batch size
        batch_size = int(arguments['--batch'])

        application = Multilabel.from_train_dir(protocol_name, train_dir, db_yml=db_yml, training=False, use_der=use_der)

        application.device = device
        application.batch_size = batch_size
        application.label = label
        application.precision = precision

        application.validate(protocol_name, subset=subset,
                             start=start, end=end, every=every,
                             in_order=in_order, task=label)

    if arguments['apply']: 

        if subset is None: 
            subset = 'test'

        model_pt = Path(arguments['<model.pt>'])
        model_pt = model_pt.expanduser().resolve(strict=True)

        output_dir = Path(arguments['<output_dir>'])
        output_dir = output_dir.expanduser().resolve(strict=False)

        # TODO. create README file in <output_dir>

        step = arguments['--step']
        if step is not None: 
            step = float(step)

        batch_size = int(arguments['--batch'])

        application = Multilabel.from_model_pt(
            protocol_name, model_pt, db_yml=db_yml, training=False)
        application.device = device
        application.batch_size = batch_size
        application.apply(protocol_name, output_dir, step=step, subset=subset)