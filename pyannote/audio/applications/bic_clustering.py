#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017 CNRS

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
BIC clustering

Usage:
  pyannote-bic-clustering tune  [--database=<db.yml> --subset=<subset>] <experiment_dir> <database.task.protocol>
  pyannote-bic-clustering apply [--database=<db.yml> --subset=<subset>] <tune_dir> <database.task.protocol>
  pyannote-bic-clustering -h | --help
  pyannote-bic-clustering --version

Options:
  <experiment_dir>           Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See "Configuration file"
                             section below for more details.
  <database.task.protocol>   Set evaluation protocol (e.g. "Etape.SpeakerDiarization.TV")
  <tune_dir>                 Set path to the directory containing optimal
                             hyper-parameters (i.e. the output of "tune" mode).
  --database=<db.yml>        Path to database configuration file.
                             [default: ~/.pyannote/db.yml]
  --subset=<subset>          Set subset (train|developement|test).
                             In "tune" mode, default subset is "development".
                             In "apply" mode, default subset is "test".
  -h --help                  Show this screen.
  --version                  Show version.


Database configuration file:
    The database configuration provides details as to where actual files are
    stored. See `pyannote.audio.util.FileFinder` docstring for more information
    on the expected format.

Configuration file:
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml, that describes the architecture of the neural
    network used for sequence labeling (0 vs. 1, non-speech vs. speech), the
    feature extraction process (e.g. MFCCs) and the sequence generator used for
    both training and testing.

    ................... <experiment_dir>/config.yml ...................
    feature_extraction:
       name: Precomputed
       params:
          root_dir: /.../.../...

    segmentation: /.../...{protocol}.{subset}.mdtm
    ...................................................................

"tune" mode:
    One should tune the hyper-parameters using "tune" mode.
    This will create the following files describing the best hyper-parameters
    to use:

        <experiment_dir>/tune/<database.task.protocol>.<subset>/tune.yml
        <experiment_dir>/tune/<database.task.protocol>.<subset>/tune.png

    This means that hyper-parameters were tuned on the <subset> subset of the
    <database.task.protocol> protocol. By default, <subset> is "development".
    This directory is called <tune_dir> in the subsequent "apply" mode.

"apply" mode
    Finally, one can apply BIC clustering using "apply" mode.
    This will create the following files that contains the outputs of BIC
    clustering

        <tune_dir>/apply/<database.task.protocol>.<subset>.mdtm

"""

import io
import yaml
import time
import warnings
import functools
from os.path import dirname, isfile, expanduser
import numpy as np

from multiprocessing import cpu_count, Pool

from docopt import docopt

from pyannote.database.util import get_unique_identifier
from pyannote.database.util import get_annotated
from pyannote.database import get_protocol

from pyannote.parser import MDTMParser

from pyannote.audio.util import mkdir_p

from pyannote.audio.features.utils import Precomputed
import pyannote.algorithms.clustering.bic
import h5py

from pyannote.metrics.diarization import GreedyDiarizationErrorRate

from .base import Application

import skopt
import skopt.space


def helper_cluster_tune(item_segmentation_features, metric=None,
                        covariance_type='full', penalty_coef=3.5, **kwargs):
    """Apply clustering on prediction and evaluate the result

    Parameters
    ----------
    item : dict
        Protocol item.
    segmentation : Annotation
        Initial segmentation
    features : SlidingWindowFeature
        Precomputed features
    metric : BaseMetric, optional
    covariance_type : {'diag', 'full'}, optional
    penalty_coef : float, optional
    **kwargs : dict
        Passed to clustering.apply()

    Returns
    -------
    value : float
        Metric value
    """

    item, segmentation, features = item_segmentation_features

    clustering = pyannote.algorithms.clustering.bic.BICClustering(
        covariance_type=covariance_type, penalty_coef=penalty_coef)

    uem = get_annotated(item)
    reference = item['annotation']
    hypothesis = clustering(segmentation, features=features)
    result = metric(reference, hypothesis, uem=uem)

    return result


class BICClustering(Application):

    TUNE_DIR = '{experiment_dir}/tune/{protocol}.{subset}'
    APPLY_DIR = '{tune_dir}/apply'

    TUNE_YML = '{tune_dir}/tune.yml'
    TUNE_PNG = '{tune_dir}/tune.png'

    SEGMENTATION_MDTM = '{segmentation_dir}/{protocol}.{subset}.mdtm'
    HARD_MDTM = '{apply_dir}/{protocol}.{subset}.mdtm'

    @classmethod
    def from_tune_dir(cls, tune_dir, db_yml=None):
        experiment_dir = dirname(dirname(tune_dir))
        bic_clustering = cls(experiment_dir, db_yml=db_yml)
        bic_clustering.tune_dir_ = tune_dir
        return bic_clustering

    def __init__(self, experiment_dir, db_yml=None):

        super(BICClustering, self).__init__(experiment_dir, db_yml=None)

        # segmentation
        self.segmentation_dir_ = self.config_['segmentation']

    def tune(self, protocol_name, subset='development'):

        tune_dir = self.TUNE_DIR.format(
            experiment_dir=self.experiment_dir,
            protocol=protocol_name,
            subset=subset)

        mkdir_p(tune_dir)

        tune_yml = self.TUNE_YML.format(tune_dir=tune_dir)
        tune_png = self.TUNE_PNG.format(tune_dir=tune_dir)

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)

        items = list(getattr(protocol, subset)())

        # segmentation
        segmentation_mdtm = self.SEGMENTATION_MDTM.format(
            segmentation_dir=self.segmentation_dir_,
            protocol=protocol_name, subset=subset)
        parser = MDTMParser().read(segmentation_mdtm)
        segmentations = [parser(item['uri']) for item in items]

        # features
        features = [self.feature_extraction_(item) for item in items]

        n_jobs = min(cpu_count(), len(items))
        pool = Pool(n_jobs)

        print(n_jobs, 'jobs')

        def callback(res):

            # plot convergence
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import skopt.plots
            _ = skopt.plots.plot_convergence(res)
            plt.savefig(tune_png, dpi=75)
            plt.close()

            # save state
            params = {'status': {'objective': float(res.fun)},
                      'covariance_type': str(res.x[0]),
                      'penalty_coef': float(res.x[1])}

            with io.open(tune_yml, 'w') as fp:
                yaml.dump(params, fp, default_flow_style=False)

        def objective_function(params):

            metric = GreedyDiarizationErrorRate()

            covariance_type, penalty_coef, = params
            process_one_file = functools.partial(
                helper_cluster_tune, metric=metric,
                covariance_type=covariance_type, penalty_coef=penalty_coef)

            if n_jobs > 1:
                results = list(pool.map(process_one_file,
                                        zip(items, segmentations, features)))
            else:
                results = [process_one_file(isf)
                           for isf in zip(items, segmentations, features)]

            return abs(metric)

        space = [skopt.space.Categorical(['full', 'diag']),
                 skopt.space.Real(0., 5., prior='uniform')]

        res = skopt.gp_minimize(
            objective_function, space, random_state=1337,
            n_calls=20, n_random_starts=10,
            verbose=True, callback=callback)

        return {'covariance_type': str(res.x[0])}, res.fun

    # def apply(self, protocol_name, subset='test'):
    #
    #     apply_dir = self.APPLY_DIR.format(tune_dir=self.tune_dir_)
    #
    #     mkdir_p(apply_dir)
    #
    #     # load tuning results
    #     tune_yml = self.TUNE_YML.format(tune_dir=self.tune_dir_)
    #     with io.open(tune_yml, 'r') as fp:
    #         self.tune_ = yaml.load(fp)
    #     covariance_type = self.tune_['covariance_type']
    #     penalty_coef = self.tune_['penalty_coef']
    #
    #     clustering = pyannote.algorithms.clustering.bic.BICClustering(
    #         covariance_type=covariance_type, penalty_coef=penalty_coef)
    #
    #     # segmentation
    #     segmentation_mdtm = self.SEGMENTATION_MDTM.format(
    #         segmentation_dir=self.segmentation_dir_,
    #         protocol=protocol_name, subset=subset)
    #     segmentations = MagicParser().read(segmentation_mdtm)
    #
    #     # initialize protocol
    #     protocol = get_protocol(protocol_name, progress=True,
    #                             preprocessors=self.preprocessors_)
    #
    #     for i, item in enumerate(getattr(protocol, subset)()):
    #
    #         features = self.feature_extraction_(item)
    #         segmentation = segmentations
    #
    #
    #         prediction = aggregation.apply(item)
    #
    #         if i == 0:
    #             # create metadata file at root that contains
    #             # sliding window and dimension information
    #             path = Precomputed.get_config_path(apply_dir)
    #             f = h5py.File(path)
    #             f.attrs['start'] = prediction.sliding_window.start
    #             f.attrs['duration'] = prediction.sliding_window.duration
    #             f.attrs['step'] = prediction.sliding_window.step
    #             f.attrs['dimension'] = 2
    #             f.close()
    #
    #         path = Precomputed.get_path(apply_dir, item)
    #
    #         # create parent directory
    #         mkdir_p(dirname(path))
    #
    #         f = h5py.File(path)
    #         f.attrs['start'] = prediction.sliding_window.start
    #         f.attrs['duration'] = prediction.sliding_window.duration
    #         f.attrs['step'] = prediction.sliding_window.step
    #         f.attrs['dimension'] = 2
    #         f.create_dataset('features', data=prediction.data)
    #         f.close()
    #
    #     # initialize binarizer
    #     onset = self.tune_['onset']
    #     offset = self.tune_['offset']
    #     binarize = Binarize(onset=onset, offset=offset)
    #
    #     precomputed = Precomputed(root_dir=apply_dir)
    #
    #     writer = MDTMParser()
    #     path = self.HARD_MDTM.format(apply_dir=apply_dir, protocol=protocol_name,
    #                             subset=subset)
    #     with io.open(path, mode='w') as gp:
    #         for item in getattr(protocol, subset)():
    #             prediction = precomputed(item)
    #             segmentation = binarize.apply(prediction, dimension=1)
    #             writer.write(segmentation.to_annotation(),
    #                          f=gp, uri=item['uri'], modality='speaker')


def main():

    arguments = docopt(__doc__, version='BIC clustering')

    db_yml = expanduser(arguments['--database'])
    protocol_name = arguments['<database.task.protocol>']
    subset = arguments['--subset']

    if arguments['tune']:
        experiment_dir = arguments['<experiment_dir>']
        if subset is None:
            subset = 'development'
        application = BICClustering(experiment_dir, db_yml=db_yml)
        application.tune(protocol_name, subset=subset)

    # if arguments['apply']:
    #     tune_dir = arguments['<tune_dir>']
    #     if subset is None:
    #         subset = 'test'
    #     application = BICClustering.from_tune_dir(
    #         tune_dir, db_yml=db_yml)
    #     application.apply(protocol_name, subset=subset)
