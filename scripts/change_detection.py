#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

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
(Speaker) change detection

Usage:
  change_detection tune  [--database=<db.yml> --subset=<subset> --purity=<beta>] <train_dir> <database.task.protocol>
  change_detection apply [--database=<db.yml> --subset=<subset> --purity=<beta>] <tune_dir> <database.task.protocol>
  change_detection -h | --help
  change_detection --version

Options:
  <train_dir>                Set path to the directory containing pre-trained
                             models (i.e. speaker embeddings).
  <database.task.protocol>   Set evaluation protocol (e.g. "Etape.SpeakerDiarization.TV")
  <tune_dir>                 Set path to the directory containing optimal
                             hyper-parameters (i.e. the output of "tune" mode).
  --database=<db.yml>        Path to database configuration file.
                             [default: ~/.pyannote/db.yml]
  --subset=<subset>          Set subset (developement|test).
                             In "tune" mode, default subset is "development".
                             In "apply" mode, default subset is "test".
  --purity=<beta>            Set importance of purity with respect to coverage.
                             [default: 1.0]
                             Use higher values if you want to improve purity.
  -h --help                  Show this screen.
  --version                  Show version.

Database configuration file:
    The database configuration provides details as to where actual files are
    stored. See `pyannote.audio.util.FileFinder` docstring for more information
    on the expected format.

"tune" mode:
    Then, one should tune the hyper-parameters using "tune" mode.
    This will create the following directory that contains a file called
    "tune.yml" describing the best hyper-parameters to use:

        <train_dir>/tune/<database.task.protocol>.<subset>

    This means that hyper-parameters were tuned on the <subset> subset of the
    <database.task.protocol> protocol. By default, <subset> is "development".
    This directory is called <tune_dir> in the subsequence "apply" mode.

"apply" mode
    Finally, one can apply (speaker) change detection using "apply" mode.
    This will create the following files that contains the hard and soft
    outputs of (speaker) change detection.

        <tune_dir>/apply/<database.task.protocol>.<subset>/{uri}.hard
                                                          /{uri}.soft
                                                          /eval.txt

    This means that file whose unique resource identifier is {uri} has been
    processed.

"""

import yaml
import pickle
import os.path
import functools
import numpy as np

from docopt import docopt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyannote.core
import pyannote.core.json

from pyannote.audio.util import mkdir_p

from pyannote.audio.embedding.base import SequenceEmbedding
from pyannote.audio.embedding.segmentation import Segmentation

from pyannote.audio.signal import Peak

from pyannote.database import get_database
from pyannote.database.util import get_unique_identifier
from pyannote.database.util import FileFileFinder
from pyannote.audio.optimizers import SSMORMS3

import skopt
import skopt.utils
import skopt.space
import skopt.plots
from pyannote.metrics.segmentation import SegmentationPurity
from pyannote.metrics.segmentation import SegmentationCoverage
from pyannote.metrics import f_measure

# needed for register_custom_object to be called
import pyannote.audio.embedding.models


def tune(protocol, train_dir, tune_dir, beta=1.0, subset='development'):

    # train_dir = experiment_dir/train/Etape.SpeakerDiarization.TV.train/2.0/...

    np.random.seed(1337)
    os.makedirs(tune_dir)

    architecture_yml = train_dir + '/architecture.yml'
    WEIGHTS_H5 = train_dir + '/weights/{epoch:04d}.h5'

    nb_epoch = 0
    while True:
        weights_h5 = WEIGHTS_H5.format(epoch=nb_epoch)
        if not os.path.isfile(weights_h5):
            break
        nb_epoch += 1

    duration = float(os.path.basename(train_dir))
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

    predictions = {}

    def objective_function(parameters, beta=1.0):

        epoch, alpha = parameters

        weights_h5 = WEIGHTS_H5.format(epoch=epoch)
        sequence_embedding = SequenceEmbedding.from_disk(
            architecture_yml, weights_h5)

        segmentation = Segmentation(
            sequence_embedding, feature_extraction,
            duration=duration, step=0.100)

        if epoch not in predictions:
            predictions[epoch] = {}

        purity = SegmentationPurity()
        coverage = SegmentationCoverage()

        f, n = 0., 0
        for dev_file in getattr(protocol, subset)():

            uri = get_unique_identifier(dev_file)
            reference = dev_file['annotation']
            n += 1

            if uri in predictions[epoch]:
                prediction = predictions[epoch][uri]
            else:
                prediction = segmentation.apply(dev_file)
                predictions[epoch][uri] = prediction

            peak = Peak(alpha=alpha)
            hypothesis = peak.apply(prediction)

            p = purity(reference, hypothesis)
            c = coverage(reference, hypothesis)
            f += f_measure(c, p, beta=beta)

        return 1 - (f / n)

    def callback(res):

        n_trials = len(res.func_vals)

        # save best parameters so far
        epoch, alpha = res.x
        params = {'status': {'nb_epoch': nb_epoch,
                             'purity': beta},
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

        # plot evaluations
        _ = skopt.plots.plot_evaluations(res)
        plt.savefig(tune_dir + '/evaluation.png', dpi=150)
        plt.close()

        try:
            # plot objective function
            _ = skopt.plots.plot_objective(res)
            plt.savefig(tune_dir + '/objective.png', dpi=150)
            plt.close()
        except Exception as e:
            pass

        # save results so far
        func = res['specs']['args']['func']
        callback = res['specs']['args']['callback']
        del res['specs']['args']['func']
        del res['specs']['args']['callback']
        skopt.utils.dump(res, tune_dir + '/tune.gz', store_objective=True)
        res['specs']['args']['func'] = func
        res['specs']['args']['callback'] = callback

    epoch = skopt.space.Integer(0, nb_epoch - 1)
    alpha = skopt.space.Real(0., 1., prior='uniform')

    res = skopt.gp_minimize(
        functools.partial(objective_function, beta=beta),
        [epoch, alpha], callback=callback,
        n_calls=1000, n_random_starts=10,
        x0=[nb_epoch - 1, 0.1],
        random_state=1337, verbose=True)

    return res


def test(protocol, tune_dir, apply_dir, subset='test', beta=1.0):

    os.makedirs(apply_dir)

    train_dir = os.path.dirname(os.path.dirname(os.path.dirname(tune_dir)))

    duration = float(os.path.basename(train_dir))
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

    segmentation = Segmentation(
        sequence_embedding, feature_extraction,
        duration=duration, step=0.100)

    peak = Peak(alpha=tune['alpha'])

    HARD_JSON = apply_dir + '/{uri}.hard.json'
    SOFT_PKL = apply_dir + '/{uri}.soft.pkl'

    eval_txt = apply_dir + '/eval.txt'
    TEMPLATE = '{uri} {purity:.5f} {coverage:.5f} {f_measure:.5f}\n'
    purity = SegmentationPurity()
    coverage = SegmentationCoverage()
    fscore = []

    for test_file in getattr(protocol, subset)():

        soft = segmentation.apply(test_file)
        hard = peak.apply(soft)

        uri = get_unique_identifier(test_file)

        path = SOFT_PKL.format(uri=uri)
        mkdir_p(os.path.dirname(path))
        with open(path, 'w') as fp:
            pickle.dump(soft, fp)

        path = HARD_JSON.format(uri=uri)
        mkdir_p(os.path.dirname(path))
        with open(path, 'w') as fp:
            pyannote.core.json.dump(hard, fp)

        try:
            reference = test_file['annotation']
            uem = test_file['annotated']
        except KeyError as e:
            continue

        p = purity(reference, hard)
        c = coverage(reference, hard)
        f = f_measure(c, p, beta=beta)
        fscore.append(f)

        line = TEMPLATE.format(
            uri=uri, purity=p, coverage=c, f_measure=f)
        with open(eval_txt, 'a') as fp:
            fp.write(line)

    p = abs(purity)
    c = abs(coverage)
    f = np.mean(fscore)
    line = TEMPLATE.format(
        uri='ALL', purity=p, coverage=c, f_measure=f)
    with open(eval_txt, 'a') as fp:
        fp.write(line)


if __name__ == '__main__':

    arguments = docopt(__doc__, version='(Speaker) change detection')

    db_yml = os.path.expanduser(arguments['--database'])
    preprocessors = {'wav': FileFinder(db_yml)}

    if '<database.task.protocol>' in arguments:
        protocol = arguments['<database.task.protocol>']
        database_name, task_name, protocol_name = protocol.split('.')
        database = get_database(database_name, preprocessors=preprocessors)
        protocol = database.get_protocol(task_name, protocol_name,
                                         progress=True)

    subset = arguments['--subset']

    if arguments['tune']:
        train_dir = arguments['<train_dir>']
        if subset is None:
            subset = 'development'
        beta = float(arguments.get('--purity'))
        tune_dir = train_dir + '/change_detection/tune/' + arguments['<database.task.protocol>'] + '.' + subset
        res = tune(protocol, train_dir, tune_dir, beta=beta, subset=subset)

    if arguments['apply']:
        tune_dir = arguments['<tune_dir>']
        if subset is None:
            subset = 'test'
        beta = float(arguments.get('--purity'))
        apply_dir = tune_dir + '/apply/' + arguments['<database.task.protocol>'] + '.' + subset
        res = test(protocol, tune_dir, apply_dir, beta=beta, subset=subset)
