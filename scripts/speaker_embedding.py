#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2017 CNRS

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
Speaker embedding

Usage:
  speaker_embedding train [--cache=<cache.h5> --robust --parallel --database=<db.yml> --subset=<subset> --duration=<duration> --min-duration=<duration> --step=<step> --validation=<subset>...] <experiment_dir> <database.task.protocol>
  speaker_embedding validation [--database=<db.yml> --subset=<subset>] <train_dir> <database.task.protocol>
  speaker_embedding tune [--database=<db.yml> --subset=<subset> --false-alarm=<beta>] <train_dir> <database.task.protocol>
  speaker_embedding test [--database=<db.yml> --subset=<subset> --false-alarm=<beta>] <tune_dir> <database.task.protocol>
  speaker_embedding apply [--database=<db.yml> --subset=<subset> --step=<step> --internal=<index>] <tune_dir> <database.task.protocol>
  speaker_embedding -h | --help
  speaker_embedding --version

Options:
  <experiment_dir>           Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See "Configuration file"
                             section below for more details.
  <database.task.protocol>   Set evaluation protocol (e.g. "Etape.SpeakerDiarization.TV")
  --cache=<cache.h5>         When provided, cache and/or use cached sequences.
  --robust                   When provided, skip files for which feature extraction fails.
  --parallel                 Run batch generator in parallel (faster, but possibly less robust).
  --database=<db.yml>        Path to database configuration file.
                             [default: ~/.pyannote/db.yml]
  --subset=<subset>          Set subset (train|developement|test).
                             In "train" mode, default is "train".
                             In "validation" mode, default is "development".
                             In "tune" mode, default is "development".
                             In "apply" mode, default is "test".
  --duration=<D>             Set duration of embedded sequences [default: 5.0]
  --min-duration=<d>         Use sequences with duration in range [<d>, <D>].
                             Defaults to sequences with fixed duration D.
  --step=<step>              Set step between sequences, in seconds.
                             Defaults to half duration.
  --validation=<subset>      Set validation subset (train|development|test).
                             May be repeated.
  --false-alarm=<beta>       Set importance of false alarm with respect to
                             false rejection [default: 1.0]
  --internal=<index>         Index of layer for which to return the activation.
                             Defaults (-1) to returning the activation of the
                             final layer.
  -h --help                  Show this screen.
  --version                  Show version.

Database configuration file:
    The database configuration provides details as to where actual files are
    stored. See `pyannote.audio.util.FileFinder` docstring for more information
    on the expected format.

Configuration file:
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml, that describes the architecture of the neural
    network used for sequence embedding, the feature extraction process
    (e.g. MFCCs) and the training strategy.

    ................... <experiment_dir>/config.yml ...................
    feature_extraction:
       name: YaafeMFCC
       params:
          e: False                   # this experiments relies
          De: True                   # on 11 MFCC coefficients
          DDe: True                  # with 1st and 2nd derivatives
          D: True                    # without energy, but with
          DD: True                   # energy derivatives

    preprocessors:
       annotation:
           name: GregoryGellySAD
           params:
              sad_yml: /vol/work1/bredin/sre10/sad.yml

    architecture:
       name: TristouNet
       params:                        # this experiments relies
          lstm: [16]                  # on one LSTM layer (16 outputs)
          bidirectional: 'concat'     # which is bidirectional
          mlp: [16, 16]               # and two dense MLP layers

    glue:
       name: LegacyTripletLoss
       params:
          distance: sqeuclidean
          margin: 0.2
          per_label: 40
    ...................................................................

"train" mode:
    First, one should train the raw sequence embedding neural network using
    "train" mode. This will create the following directory that contains
    the pre-trained neural network weights after each epoch:

        <experiment_dir>/train/<database.task.protocol>.<subset>/<duration>

    This means that the network was trained on the <subset> subset of the
    <database.task.protocol> protocol. By default, <subset> is "train".
    This directory is called <train_dir> in the subsequent "tune" mode.

"validation" mode:
    When "train" mode is launched without --validation (recommended), use the
    "validation" mode to run validation in parallel. "validation" mode will
    watch the <train_dir> directory, and run validation experiments every time
    a new epoch has ended. This will create the following directory that
    contains validation results:

        <train_dir>/validation/<database.task.protocol>.<subset>

    You can run multiple "validation" in parallel (e.g. for every subset,
    protocol, task, or database).

"tune" mode:
    Then, one should tune the hyper-parameters using "tune" mode.
    This will create the following directory that contains a file called
    "tune.yml" describing the best hyper-parameters to use:

        <train_dir>/tune/<database.task.protocol>.<subset>

    This means that hyper-parameters were tuned on the <subset> subset of the
    <database.task.protocol> protocol. By default, <subset> is "development".
    This directory is called <tune_dir> in the subsequence "apply" mode.

"test" mode:
    ...

"apply" mode
    Finally, one can apply the embedding using "apply" mode.
    This will create the following files that contains the hard and soft
    outputs of speech activity detection.

        <tune_dir>/apply/<database.task.protocol>.<subset>/{uri}.pkl

    This means that file whose unique resource identifier is {uri} has been
    processed.

"""

import time
import yaml
import h5py
import pickle
import os.path
import datetime
import functools
import itertools
import numpy as np
np.random.seed(1337)

from docopt import docopt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyannote.core

# file management
from pyannote.database import get_database
from pyannote.database.util import FileFinder
from pyannote.database.util import get_unique_identifier

from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.database.protocol import SpeakerRecognitionProtocol

from pyannote.audio.util import mkdir_p

from pyannote.audio.optimizers import SSMORMS3
from pyannote.audio.embedding.base import SequenceEmbedding
from pyannote.audio.generators.labels import FixedDurationSequences
from pyannote.audio.generators.labels import VariableDurationSequences

# evaluation
from pyannote.audio.embedding.utils import pdist, cdist
from pyannote.audio.embedding.utils import get_range, l2_normalize
from pyannote.metrics.binary_classification import det_curve

from pyannote.metrics.plot.binary_classification import plot_distributions
from pyannote.metrics.plot.binary_classification import plot_det_curve
from pyannote.metrics.plot.binary_classification import plot_precision_recall_curve

# embedding extraction
from pyannote.audio.embedding.extraction import Extraction
from pyannote.audio.embedding.aggregation import SequenceEmbeddingAggregation
from pyannote.audio.features.utils import Precomputed

# needed for register_custom_object to be called
import pyannote.audio.embedding.models

import skopt
import skopt.utils
import skopt.space
import skopt.plots
import sklearn.metrics
from pyannote.metrics import f_measure


def train(protocol, duration, experiment_dir, train_dir, subset='train',
          min_duration=None, step=None, validation=[],
          cache=False, robust=False, parallel=False):

    # -- TRAINING --
    nb_epoch = 1000
    optimizer = SSMORMS3()
    batch_size = 8192

    # load configuration file
    config_yml = experiment_dir + '/config.yml'
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    # -- PREPROCESSORS --
    for key, preprocessor in config.get('preprocessors', {}).items():
        preprocessor_name = preprocessor['name']
        preprocessor_params = preprocessor.get('params', {})
        preprocessors = __import__('pyannote.audio.preprocessors',
                                   fromlist=[preprocessor_name])
        Preprocessor = getattr(preprocessors, preprocessor_name)
        protocol.preprocessors[key] = Preprocessor(**preprocessor_params)

    # -- FEATURE EXTRACTION --
    feature_extraction_name = config['feature_extraction']['name']
    features = __import__('pyannote.audio.features',
                          fromlist=[feature_extraction_name])
    FeatureExtraction = getattr(features, feature_extraction_name)
    feature_extraction = FeatureExtraction(
        **config['feature_extraction'].get('params', {}))

    # -- ARCHITECTURE --
    architecture_name = config['architecture']['name']
    models = __import__('pyannote.audio.embedding.models',
                        fromlist=[architecture_name])
    Architecture = getattr(models, architecture_name)
    architecture = Architecture(
        **config['architecture'].get('params', {}))

    # -- GLUE --
    glue_name = config['glue']['name']
    glues = __import__('pyannote.audio.embedding',
                        fromlist=[glue_name])
    Glue = getattr(glues, glue_name)
    glue = Glue(feature_extraction,
                duration=duration,
                step=step,
                min_duration=min_duration,
                cache=cache,
                robust=robust,
                **config['glue'].get('params', {}))

    # actual training
    embedding = SequenceEmbedding(glue=glue)
    embedding.fit(architecture, protocol, nb_epoch, train=subset,
                  optimizer=optimizer, batch_size=batch_size,
                  log_dir=train_dir, validation=validation,
                  max_q_size=1 if parallel else 0)


def speaker_recognition_xp(aggregation, protocol, subset='development',
                           distance='angular', threads=None):

    method = '{subset}_enroll'.format(subset=subset)
    enroll = getattr(protocol, method)(yield_name=True)

    method = '{subset}_test'.format(subset=subset)
    test = getattr(protocol, method)(yield_name=True)

    # TODO parallelize using multiprocessing
    fX = {}
    for name, item in itertools.chain(enroll, test):
        if name in fX:
            continue
        embeddings = aggregation.apply(item)
        fX[name] = np.sum(embeddings.data, axis=0)

    method = '{subset}_keys'.format(subset=subset)
    keys = getattr(protocol, method)()

    enroll_fX = l2_normalize(np.vstack([fX[name] for name in keys.index]))
    test_fX = l2_normalize(np.vstack([fX[name] for name in keys]))

    # compare all possible (enroll, test) pairs at once
    D = cdist(enroll_fX, test_fX, metric=distance)

    positive = D[np.where(keys == 1)]
    negative = D[np.where(keys == -1)]
    # untested = D[np.where(keys == 0)]
    y_pred = np.hstack([positive, negative])

    n_positive = positive.shape[0]
    n_negative = negative.shape[0]
    # n_untested = untested.shape[0]
    y_true = np.hstack([np.ones(n_positive,), np.zeros(n_negative)])

    return det_curve(y_true, y_pred, distances=True)


def speaker_diarization_xp(sequence_embedding, X, y, distance='angular'):

    fX = sequence_embedding.transform(X)

    # compute distance between every pair of sequences
    y_pred = pdist(fX, metric=distance)

    # compute same/different groundtruth
    y_true = pdist(y, metric='chebyshev') < 1

    # return DET curve
    return det_curve(y_true, y_pred, distances=True)


def validate(protocol, train_dir, validation_dir, subset='development'):

    mkdir_p(validation_dir)

    # -- DURATIONS --
    duration, min_duration, step = path_to_duration(os.path.basename(train_dir))

    # -- CONFIGURATION --
    config_dir = os.path.dirname(os.path.dirname(os.path.dirname(train_dir)))
    config_yml = config_dir + '/config.yml'
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    # -- DISTANCE --
    distance = config['glue'].get('params', {}).get('distance', 'sqeuclidean')

    # -- PREPROCESSORS --
    for key, preprocessor in config.get('preprocessors', {}).items():
        preprocessor_name = preprocessor['name']
        preprocessor_params = preprocessor.get('params', {})
        preprocessors = __import__('pyannote.audio.preprocessors',
                                   fromlist=[preprocessor_name])
        Preprocessor = getattr(preprocessors, preprocessor_name)
        protocol.preprocessors[key] = Preprocessor(**preprocessor_params)

    # -- FEATURE EXTRACTION --
    feature_extraction_name = config['feature_extraction']['name']
    features = __import__('pyannote.audio.features',
                          fromlist=[feature_extraction_name])
    FeatureExtraction = getattr(features, feature_extraction_name)
    feature_extraction = FeatureExtraction(
        **config['feature_extraction'].get('params', {}))

    architecture_yml = train_dir + '/architecture.yml'
    WEIGHTS_H5 = train_dir + '/weights/{epoch:04d}.h5'

    EER_TEMPLATE = '{epoch:04d} {now} {eer:5f}\n'
    eers = []

    path = validation_dir + '/{subset}.eer.txt'.format(subset=subset)
    with open(path, mode='w') as fp:

        epoch = 0
        while True:

            # wait until weight file is available
            weights_h5 = WEIGHTS_H5.format(epoch=epoch)
            if not os.path.isfile(weights_h5):
                time.sleep(60)
                continue

            now = datetime.datetime.now().isoformat()

            # load current model
            sequence_embedding = SequenceEmbedding.from_disk(
                architecture_yml, weights_h5)

            # if speaker recognition protocol
            if isinstance(protocol, SpeakerRecognitionProtocol):

                aggregation = SequenceEmbeddingAggregation(
                    sequence_embedding, feature_extraction,
                    duration=duration, min_duration=min_duration,
                    step=step, internal=-2, batch_size=8192)
                aggregation.cache_preprocessed_ = False

                # compute equal error rate
                _, _, _, eer = speaker_recognition_xp(
                    aggregation, protocol, subset=subset, distance=distance)

            elif isinstance(protocol, SpeakerDiarizationProtocol):

                if epoch == 0:
                    X, y = generate_test(
                        protocol, subset, feature_extraction,
                        duration, min_duration=min_duration, step=step)

                _, _, _, eer = speaker_diarization_xp(
                    sequence_embedding, X, y, distance=distance)

            fp.write(EER_TEMPLATE.format(epoch=epoch, eer=eer, now=now))
            fp.flush()

            eers.append(eer)
            best_epoch = np.argmin(eers)
            best_value = np.min(eers)
            fig = plt.figure()
            plt.plot(eers, 'b')
            plt.plot([best_epoch], [best_value], 'bo')
            plt.plot([0, epoch], [best_value, best_value], 'k--')
            plt.grid(True)
            plt.xlabel('epoch')
            plt.ylabel('EER on {subset}'.format(subset=subset))
            TITLE = 'EER = {best_value:.5g} on {subset} @ epoch #{best_epoch:d}'
            title = TITLE.format(best_value=best_value,
                                 best_epoch=best_epoch,
                                 subset=subset)
            plt.title(title)
            plt.tight_layout()
            path = validation_dir + '/{subset}.eer.png'.format(subset=subset)
            plt.savefig(path, dpi=150)
            plt.close(fig)

            # skip to next epoch
            epoch += 1


def generate_test(protocol, subset, feature_extraction,
                  duration, min_duration=None, step=None):

    np.random.seed(1337)

    generator = FixedDurationSequences(
        feature_extraction,
        duration=duration, min_duration=min_duration, step=step,
        batch_size=-1)

    X, y = zip(*generator(getattr(protocol, subset)()))
    X, y = np.vstack(X), np.hstack(y)

    # randomly select (at most) 100 sequences from each speaker to ensure
    # all speakers have the same importance in the evaluation
    unique, y, counts = np.unique(y, return_inverse=True, return_counts=True)
    n_speakers = len(unique)
    indices = []
    for speaker in range(n_speakers):
        i = np.random.choice(np.where(y == speaker)[0], size=min(100, counts[speaker]), replace=False)
        indices.append(i)
    indices = np.hstack(indices)
    X, y = X[indices], y[indices, np.newaxis]

    return X, y


def tune(protocol, train_dir, tune_dir, beta=1.0, subset='development'):

    batch_size = 32
    os.makedirs(tune_dir)

    architecture_yml = train_dir + '/architecture.yml'
    WEIGHTS_H5 = train_dir + '/weights/{epoch:04d}.h5'

    nb_epoch = 0
    while True:
        weights_h5 = WEIGHTS_H5.format(epoch=nb_epoch)
        if not os.path.isfile(weights_h5):
            break
        nb_epoch += 1

    # -- DURATIONS --
    duration, min_duration, step = path_to_duration(os.path.basename(train_dir))

    config_dir = os.path.dirname(os.path.dirname(os.path.dirname(train_dir)))
    config_yml = config_dir + '/config.yml'
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    # -- PREPROCESSORS --
    for key, preprocessor in config.get('preprocessors', {}).items():
        preprocessor_name = preprocessor['name']
        preprocessor_params = preprocessor.get('params', {})
        preprocessors = __import__('pyannote.audio.preprocessors',
                                   fromlist=[preprocessor_name])
        Preprocessor = getattr(preprocessors, preprocessor_name)
        protocol.preprocessors[key] = Preprocessor(**preprocessor_params)

    # -- FEATURE EXTRACTION --
    feature_extraction_name = config['feature_extraction']['name']
    features = __import__('pyannote.audio.features',
                          fromlist=[feature_extraction_name])
    FeatureExtraction = getattr(features, feature_extraction_name)
    feature_extraction = FeatureExtraction(
        **config['feature_extraction'].get('params', {}))

    distance = config['glue'].get('params', {}).get('distance', 'sqeuclidean')

    X, y = generate_test(protocol, subset, feature_extraction,
                         duration, min_duration=min_duration, step=step)

    alphas = {}

    def objective_function(parameters, beta=1.0):

        epoch = parameters[0]

        weights_h5 = WEIGHTS_H5.format(epoch=epoch)
        sequence_embedding = SequenceEmbedding.from_disk(
            architecture_yml, weights_h5)

        fX = sequence_embedding.transform(X, batch_size=batch_size)

        # compute distance between every pair of sequences
        y_distance = pdist(fX, metric=distance)

        # compute same/different groundtruth
        y_true = pdist(y, metric='chebyshev') < 1

        # false positive / true positive
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(
            y_true, -y_distance, pos_label=True, drop_intermediate=True)

        fnr = 1. - tpr
        far = fpr

        thresholds = -thresholds
        fscore = 1. - f_measure(1. - fnr, 1. - far, beta=beta)

        i = np.nanargmin(fscore)
        alphas[epoch] = float(thresholds[i])
        return fscore[i]

    def callback(res):

        n_trials = len(res.func_vals)

        # save best parameters so far
        epoch = int(res.x[0])
        alpha = alphas[epoch]

        params = {'status': {'nb_epoch': nb_epoch,
                             'false_alarm': beta},
                  'epoch': int(epoch),
                  'alpha': float(alpha)}

        with open(tune_dir + '/tune.yml', 'w') as fp:
            yaml.dump(params, fp, default_flow_style=False)

        # plot convergence
        _ = skopt.plots.plot_convergence(res)
        plt.savefig(tune_dir + '/convergence.png', dpi=150)
        plt.close()

        if n_trials % 10 > 0:
            return

        # save results so far
        func = res['specs']['args']['func']
        callback = res['specs']['args']['callback']
        del res['specs']['args']['func']
        del res['specs']['args']['callback']
        skopt.utils.dump(res, tune_dir + '/tune.gz', store_objective=True)
        res['specs']['args']['func'] = func
        res['specs']['args']['callback'] = callback

    epoch = skopt.space.Integer(0, nb_epoch - 1)

    res = skopt.gp_minimize(
        functools.partial(objective_function, beta=beta),
        [epoch, ], callback=callback,
        n_calls=1000, n_random_starts=10,
        x0=[nb_epoch - 1, 0.1],
        random_state=1337, verbose=True)

    return res


def test(protocol, tune_dir, test_dir, subset, beta=1.0):

    batch_size = 32

    try:
        os.makedirs(test_dir)
    except Exception as e:
        pass

    train_dir = os.path.dirname(os.path.dirname(tune_dir))

    # -- DURATIONS --
    duration, min_duration, step = path_to_duration(os.path.basename(train_dir))

    config_dir = os.path.dirname(os.path.dirname(os.path.dirname(train_dir)))
    config_yml = config_dir + '/config.yml'
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    # -- PREPROCESSORS --
    for key, preprocessor in config.get('preprocessors', {}).items():
        preprocessor_name = preprocessor['name']
        preprocessor_params = preprocessor.get('params', {})
        preprocessors = __import__('pyannote.audio.preprocessors',
                                   fromlist=[preprocessor_name])
        Preprocessor = getattr(preprocessors, preprocessor_name)
        protocol.preprocessors[key] = Preprocessor(**preprocessor_params)

    # -- FEATURE EXTRACTION --
    feature_extraction_name = config['feature_extraction']['name']
    features = __import__('pyannote.audio.features',
                          fromlist=[feature_extraction_name])
    FeatureExtraction = getattr(features, feature_extraction_name)
    feature_extraction = FeatureExtraction(
        **config['feature_extraction'].get('params', {}))

    distance = config['glue'].get('params', {}).get('distance', 'sqeuclidean')

    # -- HYPER-PARAMETERS --
    tune_yml = tune_dir + '/tune.yml'
    with open(tune_yml, 'r') as fp:
        tune = yaml.load(fp)

    architecture_yml = train_dir + '/architecture.yml'
    WEIGHTS_H5 = train_dir + '/weights/{epoch:04d}.h5'
    weights_h5 = WEIGHTS_H5.format(epoch=tune['epoch'])

    sequence_embedding = SequenceEmbedding.from_disk(
        architecture_yml, weights_h5)

    X, y = generate_test(protocol, subset, feature_extraction,
                         duration, min_duration=min_duration, step=step)
    fX = sequence_embedding.transform(X, batch_size=batch_size)
    y_distance = pdist(fX, metric=distance)
    y_true = pdist(y, metric='chebyshev') < 1

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        y_true, -y_distance, pos_label=True, drop_intermediate=True)

    frr = 1. - tpr
    far = fpr
    thresholds = -thresholds

    eer_index = np.where(far > frr)[0][0]
    eer = .25 * (far[eer_index-1] + far[eer_index] +
                 frr[eer_index-1] + frr[eer_index])

    fscore = 1. - f_measure(1. - frr, 1. - far, beta=beta)

    opt_i = np.nanargmin(fscore)
    opt_alpha = float(thresholds[opt_i])
    opt_far = far[opt_i]
    opt_frr = frr[opt_i]
    opt_fscore = fscore[opt_i]

    alpha = tune['alpha']
    actual_i = np.searchsorted(thresholds, alpha)
    actual_far = far[actual_i]
    actual_frr = frr[actual_i]
    actual_fscore = fscore[actual_i]

    save_to = test_dir + '/' + subset
    plot_distributions(y_true, y_distance, save_to)
    eer = plot_det_curve(y_true, -y_distance, save_to)
    plot_precision_recall_curve(y_true, -y_distance, save_to)

    with open(save_to + '.txt', 'w') as fp:
        fp.write('# cond. thresh  far     frr     fscore  eer\n')
        TEMPLATE = '{condition} {alpha:.5f} {far:.5f} {frr:.5f} {fscore:.5f} {eer:.5f}\n'
        fp.write(TEMPLATE.format(condition='optimal',
                                 alpha=opt_alpha,
                                 far=opt_far,
                                 frr=opt_frr,
                                 fscore=opt_fscore,
                                 eer=eer))
        fp.write(TEMPLATE.format(condition='actual ',
                                 alpha=alpha,
                                 far=actual_far,
                                 frr=actual_frr,
                                 fscore=actual_fscore,
                                 eer=eer))


def embed(protocol, tune_dir, apply_dir, subset='test', step=None, internal=None):

    os.makedirs(apply_dir)

    train_dir = os.path.dirname(os.path.dirname(tune_dir))

    duration, _, _ = path_to_duration(os.path.basename(train_dir))

    config_dir = os.path.dirname(os.path.dirname(os.path.dirname(train_dir)))
    config_yml = config_dir + '/config.yml'
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    # -- FEATURE EXTRACTION --
    feature_extraction_name = config['feature_extraction']['name']
    features = __import__('pyannote.audio.features',
                          fromlist=[feature_extraction_name])
    FeatureExtraction = getattr(features, feature_extraction_name)
    feature_extraction = FeatureExtraction(
        **config['feature_extraction'].get('params', {}))

    # -- HYPER-PARAMETERS --
    tune_yml = tune_dir + '/tune.yml'
    with open(tune_yml, 'r') as fp:
        tune = yaml.load(fp)

    architecture_yml = train_dir + '/architecture.yml'
    WEIGHTS_H5 = train_dir + '/weights/{epoch:04d}.h5'
    weights_h5 = WEIGHTS_H5.format(epoch=tune['epoch'])

    sequence_embedding = SequenceEmbedding.from_disk(
        architecture_yml, weights_h5)

    dimension = sequence_embedding.embedding_.output_shape[-1]

    extraction = Extraction(sequence_embedding, feature_extraction,
                            duration=duration, step=step, internal=internal)

    dimension = extraction.dimension
    sliding_window = extraction.sliding_window

    # create metadata file at root that contains
    # sliding window and dimension information
    path = Precomputed.get_config_path(apply_dir)
    f = h5py.File(path)
    f.attrs['start'] = sliding_window.start
    f.attrs['duration'] = sliding_window.duration
    f.attrs['step'] = sliding_window.step
    f.attrs['dimension'] = dimension
    f.close()

    for item in getattr(protocol, subset)():

        uri = get_unique_identifier(item)
        path = Precomputed.get_path(apply_dir, item)

        extracted = extraction.apply(item)

        # create parent directory
        mkdir_p(os.path.dirname(path))

        f = h5py.File(path)
        f.attrs['start'] = sliding_window.start
        f.attrs['duration'] = sliding_window.duration
        f.attrs['step'] = sliding_window.step
        f.attrs['dimension'] = dimension
        f.create_dataset('features', data=extracted.data)
        f.close()

# (5, None, None) ==> '5'
# (5, 1, None) ==> '1-5'
# (5, None, 2) ==> '5+2'
# (5, 1, 2) ==> '1-5+2'
def duration_to_path(duration=5.0, min_duration=None, step=None):
    PATH = '' if min_duration is None else '{min_duration:g}-'
    PATH += '{duration:g}'
    if step is not None:
        PATH += '+{step:g}'
    return PATH.format(duration=duration, min_duration=min_duration, step=step)

# (5, None, None) <== '5'
# (5, 1, None) <== '1-5'
# (5, None, 2) <== '5+2'
# (5, 1, 2) <== '1-5+2'
def path_to_duration(path):
    tokens = path.split('+')
    step = float(tokens[1]) if len(tokens) == 2 else None
    tokens = tokens[0].split('-')
    min_duration = float(tokens[0]) if len(tokens) == 2 else None
    duration = float(tokens[0]) if len(tokens) == 1 else float(tokens[1])
    return duration, min_duration, step


if __name__ == '__main__':

    arguments = docopt(__doc__, version='Speaker embedding')

    db_yml = os.path.expanduser(arguments['--database'])
    preprocessors = {'wav': FileFinder(db_yml)}

    if '<database.task.protocol>' in arguments:
        protocol = arguments['<database.task.protocol>']
        database_name, task_name, protocol_name = protocol.split('.')
        database = get_database(database_name, preprocessors=preprocessors)
        protocol = database.get_protocol(task_name, protocol_name, progress=True)

    subset = arguments['--subset']

    if arguments['train']:
        experiment_dir = arguments['<experiment_dir>']

        if subset is None:
            subset = 'train'

        duration = float(arguments['--duration'])

        min_duration = arguments['--min-duration']
        if min_duration is not None:
            min_duration = float(min_duration)

        step = arguments['--step']
        if step is not None:
            step = float(step)

        TRAIN_DIR = '{experiment_dir}/train/{protocol}.{subset}/{path}'
        # e.g. '1-5+2' or '5'
        path = duration_to_path(duration=duration,
                                min_duration=min_duration,
                                step=step)
        train_dir = TRAIN_DIR.format(
            experiment_dir=experiment_dir,
            protocol=arguments['<database.task.protocol>'],
            subset=subset, path=path)

        validation = arguments['--validation']

        cache = arguments['--cache']
        robust = arguments['--robust']
        parallel = arguments['--parallel']

        train(protocol, duration, experiment_dir, train_dir, subset=subset,
              min_duration=min_duration, step=step, validation=validation,
              cache=cache, robust=robust, parallel=parallel)

    if arguments['validation']:
        train_dir = arguments['<train_dir>']
        if subset is None:
            subset = 'development'
        validation_dir = train_dir + '/validation/' + arguments['<database.task.protocol>']
        res = validate(protocol, train_dir, validation_dir, subset=subset)

    if arguments['tune']:
        train_dir = arguments['<train_dir>']
        if subset is None:
            subset = 'development'
        beta = float(arguments['--false-alarm'])
        tune_dir = train_dir + '/tune/' + arguments['<database.task.protocol>'] + '.' + subset
        res = tune(protocol, train_dir, tune_dir, beta=beta, subset=subset)

    if arguments['test']:
        tune_dir = arguments['<tune_dir>']
        if subset is None:
            subset = 'test'
        beta = float(arguments['--false-alarm'])
        test_dir = tune_dir + '/test/' + arguments['<database.task.protocol>']
        test(protocol, tune_dir, test_dir, subset, beta=beta)

    if arguments['apply']:
        tune_dir = arguments['<tune_dir>']

        apply_dir = tune_dir + '/apply'

        step = arguments['--step']
        if step is not None:
            step = float(step)
            apply_dir += '/step_{step:g}'.format(step=step)

        internal = arguments['--internal']
        if internal is not None:
            internal = int(internal)
            apply_dir += '/internal_{internal:d}'.format(internal=internal)

        if subset is None:
            subset = 'test'

        embed(protocol, tune_dir, apply_dir,
              subset=subset, step=step,
              internal=internal)
