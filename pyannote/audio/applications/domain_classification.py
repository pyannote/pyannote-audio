#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019 CNRS

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
Domain classification

Usage:
  pyannote-domain-classification train [options] <experiment_dir> <database.task.protocol>
  pyannote-domain-classification validate [options] [--every=<epoch> --chronological] <train_dir> <database.task.protocol>
  pyannote-domain-classification apply [options] [--step=<step>] <validate_dir> <database.task.protocol>
  pyannote-domain-classification -h | --help
  pyannote-domain-classification --version

Common options:
  <database.task.protocol>   Experimental protocol (e.g. "AMI.SpeakerDiarization.MixHeadset")
  --database=<database.yml>  Path to pyannote.database configuration file.
  --subset=<subset>          Set subset (train|developement|test).
                             Defaults to "train" in "train" mode. Defaults to
                             "development" in "validate" mode. Defaults to
                             "test" in "apply" mode.
  --gpu                      Run on GPUs. Defaults to using CPUs.
  --batch=<size>             Set batch size. Has no effect in "train" mode.
                             [default: 32]
  --from=<epoch>             Start {train|validat}ing at epoch <epoch>. Has no
                             effect in "apply" mode. [default: 0]
  --to=<epochs>              End {train|validat}ing at epoch <epoch>.
                             Defaults to keep going forever.

"train" mode:
  <experiment_dir>           Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See "Configuration file"
                             section below for more details.

"validation" mode:
  --every=<epoch>            Validate model every <epoch> epochs [default: 1].
  --chronological            Force validation in chronological order.
  <train_dir>                Path to the directory containing pre-trained
                             models (i.e. the output of "train" mode).

"apply" mode:
  <validate_dir>             Path to the directory containing validation
                             results (i.e. the output of "validate" mode).
  --step=<step>              Sliding window step, in seconds.
                             Defaults to 25% of window duration.

Configuration file:
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml, that describes the feature extraction process,
    the neural network architecture, and the task addressed.

    ................... <experiment_dir>/config.yml ...................
    # train the network for domain classification
    # see pyannote.audio.labeling.tasks for more details
    task:
       name: DomainClassification
       params:
          duration: 3.2     # sub-sequence duration
          per_epoch: 1      # 1 day of audio per epoch
          batch_size: 32    # number of sub-sequences per batch

    # use the PyanNet architecture
    # see pyannote.audio.models for more details
    architecture:
       name: PyanNet
       params:
         rnn:
            pool: max

    # use cyclic learning rate scheduler
    scheduler:
       name: ConstantScheduler
       params:
           learning_rate: 0.1
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

    In practice, for each epoch, "validate" mode will compute the
    classification accuracy.

"apply" mode:
    Use the "apply" mode to extract domain classification raw scores and and
    results. This will create the following directory that contains domain
    classification results:

        <validate_dir>/apply/<epoch>
"""

from functools import partial
from pathlib import Path
import torch
import numpy as np
import scipy.optimize
from docopt import docopt
from .base_labeling import BaseLabeling
from pyannote.database import get_annotated
from pyannote.audio.labeling.extraction import SequenceLabeling
from collections import Counter
from sklearn.metrics import confusion_matrix


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0.:
                continue
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    size=6)
    fig.tight_layout()
    return ax



class DomainClassification(BaseLabeling):

    def validate_epoch(self, epoch, protocol_name, subset='development',
                       validation_data=None):

        # load model for current epoch
        model = self.load_model(epoch).to(self.device)
        model.eval()

        domain = self.task_.domain
        domains = model.specifications['y']['classes']

        duration = self.task_.duration
        sequence_labeling = SequenceLabeling(
            model=model, feature_extraction=self.feature_extraction_,
            duration=duration, step=.25 * duration, batch_size=self.batch_size,
            device=self.device)

        y_true_file, y_pred_file = [], []

        for current_file in validation_data:

            y_pred = sequence_labeling(current_file).data.argmax(axis=1)
            y_pred_file.append(Counter(y_pred).most_common(1)[0][0])

            y_true = domains.index(current_file[domain])
            y_true_file.append(y_true)

        accuracy = np.mean(np.array(y_true_file) == np.array(y_pred_file))
        return {'metric': 'accuracy',
                'minimize': False,
                'value': float(accuracy)}

def main():
    arguments = docopt(__doc__, version='Domain classification')

    db_yml = arguments['--database']
    protocol_name = arguments['<database.task.protocol>']
    subset = arguments['--subset']

    gpu = arguments['--gpu']
    device = torch.device('cuda') if gpu else torch.device('cpu')

    # HACK to "book" GPU as soon as possible
    _ = torch.Tensor([0]).to(device)

    if arguments['train']:
        experiment_dir = Path(arguments['<experiment_dir>'])
        experiment_dir = experiment_dir.expanduser().resolve(strict=True)

        if subset is None:
            subset = 'train'

        # start training at this epoch (defaults to 0)
        warm_start = int(arguments['--from'])

        # stop training at this epoch (defaults to never stop)
        epochs = arguments['--to']
        if epochs is None:
            epochs = np.inf
        else:
            epochs = int(epochs)

        application = DomainClassification(experiment_dir, db_yml=db_yml,
                                           training=True)
        application.device = device
        application.train(protocol_name, subset=subset,
                          warm_start=warm_start, epochs=epochs)

    if arguments['validate']:

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

        application = DomainClassification.from_train_dir(
            train_dir, db_yml=db_yml, training=False)
        application.device = device
        application.batch_size = batch_size
        application.validate(protocol_name, subset=subset,
                             start=start, end=end, every=every,
                             in_order=in_order)

    if arguments['apply']:

        validate_dir = Path(arguments['<validate_dir>'])
        validate_dir = validate_dir.expanduser().resolve(strict=True)

        if subset is None:
            subset = 'test'

        step = arguments['--step']
        if step is not None:
            step = float(step)

        batch_size = int(arguments['--batch'])

        application = DomainClassification.from_validate_dir(
            validate_dir, db_yml=db_yml, training=False)
        application.device = device
        application.batch_size = batch_size
        application.apply(protocol_name, step=step, subset=subset)
