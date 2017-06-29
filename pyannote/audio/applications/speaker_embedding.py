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
Speaker embedding

Usage:
  pyannote-speaker-embedding data [--database=<db.yml> --duration=<duration> --step=<step> --heterogeneous] <root_dir> <database.task.protocol>
  pyannote-speaker-embedding train [--subset=<subset> --start=<epoch> --end=<epoch>] <experiment_dir> <database.task.protocol>
  pyannote-speaker-embedding validate [--subset=<subset> --aggregate --every=<epoch> --from=<epoch>] <train_dir> <database.task.protocol>
  pyannote-speaker-embedding apply [--database=<db.yml> --step=<step> --internal] <validate.txt> <database.task.protocol> <output_dir>
  pyannote-speaker-embedding -h | --help
  pyannote-speaker-embedding --version

Options:
  <root_dir>                 Set root directory. This script expects a
                             configuration file called "config.yml" to live in
                             this directory. See '"data" mode' section below
                             for more details.
  <database.task.protocol>   Set evaluation protocol (e.g. "Etape.SpeakerDiarization.TV")
  --database=<db.yml>        Path to database configuration file.
                             [default: ~/.pyannote/db.yml]
  --duration=<duration>      Set duration of embedded sequences [default: 3.2]
  --step=<step>              Set step between sequences, in seconds.
                             Defaults to 0.5 x <duration>.
  --heterogeneous            Allow heterogeneous sequences. In this case, the
                             label given to heterogeneous sequences is the most
                             overlapping one.
  --start=<epoch>            Restart training after that many epochs.
  --end=<epoch>              Stop training after than many epochs [default: 1000]
  <experiment_dir>           Set experiment directory. This script expects a
                             configuration file called "config.yml" to live
                             in this directory. See '"train" mode' section
                             for more details.
  --subset=<subset>          Set subset (train|developement|test).
                             In "train" mode, defaults subset is "train".
                             In "validate" mode, defaults to "development".
  --aggregate                Aggregate
  --every=<epoch>            Defaults to every epoch [default: 1].
  --from=<epoch>             Start at this epoch [default: 0].
  <train_dir>                Path to directory created by "train" mode.
  --internal                 Extract internal representation.
  -h --help                  Show this screen.
  --version                  Show version.


Database configuration file:
    The database configuration provides details as to where actual files are
    stored. See `pyannote.audio.util.FileFinder` docstring for more information
    on the expected format.

"data" mode:

    A file called <root_dir>/config.yml should exist, that describes the
    feature extraction process (e.g. MFCCs):

    ................... <root_dir>/config.yml .........................
    feature_extraction:
       name: YaafeMFCC
       params:
          e: False                   # this experiments relies
          De: True                   # on 11 MFCC coefficients
          DDe: True                  # with 1st and 2nd derivatives
          D: True                    # without energy, but with
          DD: True                   # energy derivatives
    ...................................................................

    Using "data" mode will create the following directory that contains
    the pre-computed sequences for train, development, and test subsets:

        <root_dir>/<duration>+<step>/sequences/<database.task.protocol>.{train|development|test}.h5

    This means that <duration>-long sequences were generated with a step of
    <step> seconds, from the <database.task.protocol> protocol. This directory
    is called <data_dir> in the subsequent modes.

"train" mode:

    The configuration of each experiment is described in a file called
    <data_dir>/<xp_id>/config.yml, that describes the architecture of the
    neural network, and the approach (e.g. triplet loss) used for training the
    network:

    ................... <train_dir>/config.yml ...................
    architecture:
       name: TristouNet
       params:
         lstm: [16]
         mlp: [16, 16]
         bidirectional: concat

    approach:
       name: TripletLoss
       params:
         per_label: 2
         per_fold: 10
    ...................................................................

    Using "train" mode will create the following directory that contains a
    bunch of files including the pre-trained neural network weights after each
    epoch:

        <data_dir>/<xp_id>/train/<database.task.protocol>.<subset>

    This means that the network was trained using the <subset> subset of the
    <database.task.protocol> protocol, using the configuration described in
    <data_dir>/<xp_id>/config.yml. This directory  is called <train_dir> in the
    subsequent modes.

"validate" mode:
    Use the "validate" mode to run validation in parallel to training.
    "validate" mode will watch the <train_dir> directory, and run validation
    experiments every time a new epoch has ended. This will create the
    following directory that contains validation results:

        <train_dir>/validate/<database.task.protocol>

    You can run multiple "validate" in parallel (e.g. for every subset,
    protocol, task, or database).
"""

from os.path import dirname, basename, isfile, expanduser
import numpy as np
import pandas as pd
import time
import yaml
from tqdm import tqdm

from docopt import docopt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyannote.database import get_protocol
from pyannote.audio.util import mkdir_p
import h5py

from .base import Application

from pyannote.generators.fragment import SlidingLabeledSegments
from pyannote.audio.optimizers import SSMORMS3

import keras.models
from pyannote.audio.keras_utils import CUSTOM_OBJECTS
from pyannote.audio.embedding.utils import pdist, cdist, l2_normalize
from pyannote.metrics.binary_classification import det_curve

from pyannote.audio.callback import LoggingCallback

from pyannote.core.util import pairwise
import keras.backend as K

from sortedcontainers import SortedDict
from pyannote.audio.features import Precomputed
from pyannote.audio.embedding.base_autograd import SequenceEmbeddingAutograd
from pyannote.audio.embedding.extraction import Extraction

class SpeakerEmbedding(Application):

    # created by "data" mode
    DATA_DIR = '{root_dir}/{params}'
    DATA_H5 = '{data_dir}/sequences/{protocol}.{subset}.h5'

    # created by "train" mode
    TRAIN_DIR = '{experiment_dir}/train/{protocol}.{subset}'

    # created by "validate" mode
    VALIDATE_DIR = '{train_dir}/validate/{protocol}'
    VALIDATE_TXT = '{validate_dir}/{subset}.{aggregate}eer.txt'
    VALIDATE_TXT_TEMPLATE = '{epoch:04d} {eer:5f}\n'

    VALIDATE_PNG = '{validate_dir}/{subset}.{aggregate}eer.png'
    VALIDATE_EPS = '{validate_dir}/{subset}.{aggregate}eer.eps'

    @classmethod
    def from_root_dir(cls, root_dir, db_yml=None):
        speaker_embedding = cls(root_dir, db_yml=db_yml)
        speaker_embedding.root_dir_ = root_dir
        return speaker_embedding

    @classmethod
    def from_train_dir(cls, train_dir, db_yml=None):
        """Initialize application from <train_dir>"""
        experiment_dir = dirname(dirname(train_dir))
        speaker_embedding = cls(experiment_dir, db_yml=db_yml)
        speaker_embedding.train_dir_ = train_dir
        return speaker_embedding

    @classmethod
    def from_validate_txt(cls, validate_txt, db_yml=None):
        train_dir = dirname(dirname(dirname(validate_txt)))
        speaker_embedding = cls.from_train_dir(train_dir, db_yml=db_yml)
        speaker_embedding.validate_txt_ = validate_txt

        root_dir = dirname(dirname(dirname(dirname(train_dir))))
        with open(root_dir + '/config.yml', 'r') as fp:
            config = yaml.load(fp)
            extraction_name = config['feature_extraction']['name']
            features = __import__('pyannote.audio.features',
                                  fromlist=[extraction_name])
            FeatureExtraction = getattr(features, extraction_name)
            speaker_embedding.feature_extraction_ = FeatureExtraction(
                **config['feature_extraction'].get('params', {}))

            # do not cache features in memory when they are precomputed on disk
            # as this does not bring any significant speed-up
            # but does consume (potentially) a LOT of memory
            speaker_embedding.cache_preprocessed_ = 'Precomputed' not in extraction_name

        return speaker_embedding

    def __init__(self, experiment_dir, db_yml=None):

        super(SpeakerEmbedding, self).__init__(
            experiment_dir, db_yml=db_yml)

        # architecture
        if 'architecture' in self.config_:
            architecture_name = self.config_['architecture']['name']
            models = __import__('pyannote.audio.embedding.models',
                                fromlist=[architecture_name])
            Architecture = getattr(models, architecture_name)
            self.architecture_ = Architecture(
                **self.config_['architecture'].get('params', {}))

        # approach
        if 'approach' in self.config_:
            approach_name = self.config_['approach']['name']
            approaches = __import__('pyannote.audio.embedding.approaches',
                                    fromlist=[approach_name])
            Approach = getattr(approaches, approach_name)
            self.approach_ = Approach(
                **self.config_['approach'].get('params', {}))

    # (5, None, None, False) ==> '5'
    # (5, 1, None, False) ==> '1-5'
    # (5, None, 2, False) ==> '5+2'
    # (5, 1, 2, False) ==> '1-5+2'
    # (5, None, None, True) ==> '5x'
    @staticmethod
    def _params_to_directory(duration=5.0, min_duration=None, step=None,
                            heterogeneous=False, skip_unlabeled=True,
                            **kwargs):
        if not skip_unlabeled:
            raise NotImplementedError('skip_unlabeled not supported yet.')

        DIRECTORY = '' if min_duration is None else '{min_duration:g}-'
        DIRECTORY += '{duration:g}'
        if step is not None:
            DIRECTORY += '+{step:g}'
        if heterogeneous:
            DIRECTORY += 'x'
        return DIRECTORY.format(duration=duration,
                                min_duration=min_duration,
                                step=step)

    # (5, None, None, False) <== '5'
    # (5, 1, None, False) <== '1-5'
    # (5, None, 2, False) <== '5+2'
    # (5, 1, 2, False) <== '1-5+2'
    @staticmethod
    def _directory_to_params(directory):
        heterogeneous = False
        if directory[-1] == 'x':
            heterogeneous = True
            directory = directory[:-1]
        tokens = directory.split('+')
        step = float(tokens[1]) if len(tokens) == 2 else None
        tokens = tokens[0].split('-')
        min_duration = float(tokens[0]) if len(tokens) == 2 else None
        duration = float(tokens[0]) if len(tokens) == 1 else float(tokens[1])
        return duration, min_duration, step, heterogeneous

    def data(self, protocol_name, duration=3.2, min_duration=None, step=None,
             heterogeneous=False):

        # labeled segment generator
        generator = SlidingLabeledSegments(duration=duration,
                                           min_duration=min_duration,
                                           step=step,
                                           heterogeneous=heterogeneous,
                                           source='annotated')

        data_dir = self.DATA_DIR.format(
            root_dir=self.root_dir_,
            params=self._params_to_directory(duration=duration,
                                            min_duration=min_duration,
                                            step=step,
                                            heterogeneous=heterogeneous))

        # file generator
        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)

        for subset in ['train', 'development', 'test']:

            try:
                file_generator = getattr(protocol, subset)()
                first_item = next(file_generator)
            except NotImplementedError as e:
                continue

            file_generator = getattr(protocol, subset)()

            data_h5 = self.DATA_H5.format(data_dir=data_dir,
                                          protocol=protocol_name,
                                          subset=subset)
            mkdir_p(dirname(data_h5))

            with h5py.File(data_h5, mode='w') as fp:

                # initialize with a fixed number of sequences
                n_sequences = 1000

                # dataset meant to store the speaker identifier
                Y = fp.create_dataset(
                    'y', shape=(n_sequences, ),
                    dtype=h5py.special_dtype(vlen=bytes),
                    maxshape=(None, ))

                # dataset meant to store the speech turn unique ID
                Z = fp.create_dataset(
                    'z', shape=(n_sequences, ),
                    dtype=np.int64,
                    maxshape=(None, ))

                i = 0  # number of sequences
                z = 0  # speech turn identifier

                for item in file_generator:

                    # feature extraction
                    features = self.feature_extraction_(item)

                    for segment, y in generator.from_file(item):

                        # extract feature sequence
                        x = features.crop(segment,
                                          mode='center',
                                          fixed=duration)

                        # create X dataset to store feature sequences
                        # this cannot be done before because we need
                        # the number of samples per sequence and the
                        # dimension of feature vectors.
                        if i == 0:
                            # get number of samples and feature dimension
                            # from the first sequence...
                            n_samples, n_features = x.shape

                            # create X dataset accordingly
                            X = fp.create_dataset(
                                'X', dtype=x.dtype, compression='gzip',
                                shape=(n_sequences, n_samples, n_features),
                                chunks=(1, n_samples, n_features),
                                maxshape=(None, n_samples, n_features))

                            # make sure the speech turn identifier
                            # will not be erroneously incremented
                            prev_y = y

                        # increase the size of the datasets when full
                        if i == n_sequences:
                            n_sequences = int(n_sequences * 1.1)
                            X.resize(n_sequences, axis=0)
                            Y.resize(n_sequences, axis=0)
                            Z.resize(n_sequences, axis=0)

                        # save current feature sequence and its label
                        X[i] = x
                        Y[i] = y

                        # a change of label indicates that a new speech turn has began.
                        # increment speech turn identifier (z) accordingly
                        if y != prev_y:
                            prev_y = y
                            z += 1

                        # save speech turn identifier
                        Z[i] = z

                        # increment number of sequences
                        i += 1

                X.resize(i-1, axis=0)
                Y.resize(i-1, axis=0)
                Z.resize(i-1, axis=0)

    def train(self, protocol_name, subset='train', start=None, end=1000):

        data_dir = dirname(self.experiment_dir)
        data_h5 = self.DATA_H5.format(data_dir=data_dir,
                                      protocol=protocol_name,
                                      subset=subset)

        got = self.approach_.get_batch_generator(data_h5)
        batch_generator = got['batch_generator']
        batches_per_epoch = got['batches_per_epoch']
        n_classes = got.get('n_classes', None)
        classes = got.get('classes', None)

        train_dir = self.TRAIN_DIR.format(experiment_dir=self.experiment_dir,
                                          protocol=protocol_name,
                                          subset=subset)

        if start is None:
            init_embedding = self.architecture_
        else:
            init_embedding = self.approach_.load(train_dir, start)

        self.approach_.fit(init_embedding, batch_generator,
                           batches_per_epoch=batches_per_epoch,
                           n_classes=n_classes, classes=classes,
                           epochs=end, log_dir=train_dir,
                           optimizer=SSMORMS3())

    def _validation_set_z(self, protocol_name, subset='development'):

        # reproducibility
        np.random.seed(1337)

        data_dir = dirname(dirname(dirname(self.train_dir_)))
        data_h5 = self.DATA_H5.format(data_dir=data_dir,
                                      protocol=protocol_name,
                                      subset=subset)

        with h5py.File(data_h5, mode='r') as fp:

            h5_X = fp['X']
            h5_y = fp['y']
            h5_z = fp['z']

            # group sequences by z
            df = pd.DataFrame({'y': h5_y, 'z': h5_z})
            z_groups = df.groupby('z')

            # label of each group
            y_groups = [group.y.iloc[0] for _, group in z_groups]

            # randomly select (at most) 10 groups from each speaker to ensure
            # all speakers have the same importance in the evaluation
            unique, y, counts = np.unique(y_groups, return_inverse=True,
                                          return_counts=True)
            n_speakers = len(unique)
            X, N, Y = [], [], []
            for speaker in range(n_speakers):
                I = np.random.choice(np.where(y == speaker)[0],
                                     size=min(10, counts[speaker]),
                                     replace=False)
                for i in I:
                    selector = z_groups.get_group(i).index
                    x = h5_X[selector]
                    X.append(x)
                    N.append(x.shape[0])
                    Y.append(y[i])

        return np.vstack(X), np.array(N),  np.array(Y)[:, np.newaxis]

    def _validation_set_y(self, protocol_name, subset='development'):

        # reproducibility
        np.random.seed(1337)

        data_dir = dirname(dirname(dirname(self.train_dir_)))
        data_h5 = self.DATA_H5.format(data_dir=data_dir,
                                      protocol=protocol_name,
                                      subset=subset)

        with h5py.File(data_h5, mode='r') as fp:

            X = fp['X']
            y = fp['y']

            # randomly select (at most) 100 sequences from each speaker to ensure
            # all speakers have the same importance in the evaluation
            unique, y, counts = np.unique(y, return_inverse=True,
                                          return_counts=True)
            n_speakers = len(unique)
            indices = []
            for speaker in range(n_speakers):
                i = np.random.choice(np.where(y == speaker)[0],
                                     size=min(100, counts[speaker]),
                                     replace=False)
                indices.append(i)
            # indices have to be sorted because of h5py indexing limitations
            indices = sorted(np.hstack(indices))

            X = np.array(X[indices])
            y = np.array(y[indices, np.newaxis])

        return X, y

    def validate(self, protocol_name, subset='development',
                 aggregate=False, every=1, start=0):

        # prepare paths
        validate_dir = self.VALIDATE_DIR.format(train_dir=self.train_dir_,
                                                protocol=protocol_name)
        validate_txt = self.VALIDATE_TXT.format(
            validate_dir=validate_dir, subset=subset,
            aggregate='aggregate.' if aggregate else '')
        validate_png = self.VALIDATE_PNG.format(
            validate_dir=validate_dir, subset=subset,
            aggregate='aggregate.' if aggregate else '')
        validate_eps = self.VALIDATE_EPS.format(
            validate_dir=validate_dir, subset=subset,
            aggregate='aggregate.' if aggregate else '')

        # create validation directory
        mkdir_p(validate_dir)

        # Build validation set
        if aggregate:
            X, n, y = self._validation_set_z(protocol_name, subset=subset)
        else:
            X, y = self._validation_set_y(protocol_name, subset=subset)

        # list of equal error rates, and epoch to process
        eers, epoch = SortedDict(), start

        desc_format = ('Best EER = {best_eer:.2f}% @ epoch #{best_epoch:d} ::'
                       ' EER = {eer:.2f}% @ epoch #{epoch:d} :')

        progress_bar = tqdm(unit='epoch')

        with open(validate_txt, mode='w') as fp:

            # watch and evaluate forever
            while True:

                # last completed epochs
                completed_epochs = self.get_epochs(self.train_dir_) - 1

                if completed_epochs < epoch:
                    time.sleep(60)
                    continue

                # if last completed epoch has already been processed
                # go back to first epoch that hasn't been processed yet
                process_epoch = epoch if completed_epochs in eers \
                                      else completed_epochs

                weights_h5 = LoggingCallback.WEIGHTS_H5.format(
                    log_dir=self.train_dir_, epoch=process_epoch)

                # this is needed for corner case when training is started from
                # an epoch > 0
                if not isfile(weights_h5):
                    time.sleep(60)
                    continue

                # sleep 5 seconds to let the checkpoint callback finish
                time.sleep(5)

                # TODO update this code once keras > 2.0.4 is released
                try:
                    embedding = keras.models.load_model(
                        weights_h5, custom_objects=CUSTOM_OBJECTS,
                        compile=False)
                except TypeError as e:
                    embedding = keras.models.load_model(
                        weights_h5, custom_objects=CUSTOM_OBJECTS)

                if aggregate:
                    def embed(X):
                        func = K.function(
                            [embedding.get_layer(name='input').input, K.learning_phase()],
                            [embedding.get_layer(name='internal').output])
                        return func([X, 0])[0]
                else:
                    embed = embedding.predict

                # embed all validation sequences
                fX = embed(X)

                if aggregate:
                    indices = np.hstack([[0], np.cumsum(n)])
                    fX = np.stack([np.sum(np.sum(fX[i:j], axis=0), axis=0)
                                   for i, j in pairwise(indices)])
                    fX = l2_normalize(fX)

                # compute pairwise distances
                y_pred = pdist(fX, metric=self.approach_.metric)
                # compute pairwise groundtruth
                y_true = pdist(y, metric='chebyshev') < 1
                # estimate equal error rate
                _, _, _, eer = det_curve(y_true, y_pred, distances=True)
                eers[process_epoch] = eer

                # save equal error rate to file
                fp.write(self.VALIDATE_TXT_TEMPLATE.format(
                    epoch=process_epoch, eer=eer))
                fp.flush()

                # keep track of best epoch so far
                best_epoch = eers.iloc[np.argmin(eers.values())]
                best_eer = eers[best_epoch]

                progress_bar.set_description(
                    desc_format.format(epoch=process_epoch, eer=100*eer,
                                       best_epoch=best_epoch,
                                       best_eer=100*best_eer))

                # plot
                fig = plt.figure()
                plt.plot(eers.keys(), eers.values(), 'b')
                plt.plot([best_epoch], [best_eer], 'bo')
                plt.plot([eers.iloc[0], eers.iloc[-1]], [best_eer, best_eer], 'k--')
                plt.grid(True)
                plt.xlabel('epoch')
                plt.ylabel('EER on {subset}'.format(subset=subset))
                TITLE = '{best_eer:.5g} @ epoch #{best_epoch:d}'
                title = TITLE.format(best_eer=best_eer,
                                     best_epoch=best_epoch,
                                     subset=subset)
                plt.title(title)
                plt.tight_layout()
                plt.savefig(validate_png, dpi=150)
                plt.savefig(validate_eps)
                plt.close(fig)

                # go to next epoch
                if epoch == process_epoch:
                    epoch += every
                    progress_bar.update(every)
                else:
                    progress_bar.update(0)

        progress_bar.close()

    def apply(self, protocol_name, output_dir, step=None, internal=False):

        # load best performing model
        with open(self.validate_txt_, 'r') as fp:
            eers = SortedDict(np.loadtxt(fp))
        best_epoch = int(eers.iloc[np.argmin(eers.values())])
        embedding = SequenceEmbeddingAutograd.load(
            self.train_dir_, best_epoch)

        # guess sequence duration from path (.../3.2+0.8/...)
        directory = basename(dirname(self.experiment_dir))
        duration, _, _, _ = self._directory_to_params(directory)
        if step is None:
            step = 0.5 * duration

        # initialize embedding extraction
        batch_size = self.approach_.batch_size
        extraction = Extraction(embedding, self.feature_extraction_, duration,
                                step=step, batch_size=batch_size,
                                internal=internal)
        sliding_window = extraction.sliding_window
        dimension = extraction.dimension

        # create metadata file at root that contains
        # sliding window and dimension information
        path = Precomputed.get_config_path(output_dir)
        mkdir_p(dirname(path))
        f = h5py.File(path)
        f.attrs['start'] = sliding_window.start
        f.attrs['duration'] = sliding_window.duration
        f.attrs['step'] = sliding_window.step
        f.attrs['dimension'] = dimension
        f.close()

        # file generator
        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)

        for subset in ['development', 'test', 'train']:

            try:
                file_generator = getattr(protocol, subset)()
                first_item = next(file_generator)
            except NotImplementedError as e:
                continue

            file_generator = getattr(protocol, subset)()

            for current_file in file_generator:

                fX = extraction.apply(current_file)

                path = Precomputed.get_path(output_dir, current_file)
                mkdir_p(dirname(path))

                f = h5py.File(path)
                f.attrs['start'] = sliding_window.start
                f.attrs['duration'] = sliding_window.duration
                f.attrs['step'] = sliding_window.step
                f.attrs['dimension'] = dimension
                f.create_dataset('features', data=fX.data)
                f.close()

def main():

    arguments = docopt(__doc__, version='Speaker embedding')

    db_yml = expanduser(arguments['--database'])
    protocol_name = arguments['<database.task.protocol>']
    subset = arguments['--subset']

    if arguments['data']:

        duration = float(arguments['--duration'])

        step = arguments['--step']
        if step is not None:
            step = float(step)

        heterogeneous = arguments['--heterogeneous']

        root_dir = arguments['<root_dir>']
        if subset is None:
            subset = 'train'

        application = SpeakerEmbedding.from_root_dir(root_dir, db_yml=db_yml)
        application.data(protocol_name, duration=duration, step=step,
                         heterogeneous=heterogeneous)

    if arguments['train']:
        experiment_dir = arguments['<experiment_dir>']

        if subset is None:
            subset = 'train'

        start = arguments['--start']
        if start is not None:
            start = int(start)

        end = int(arguments['--end'])

        application = SpeakerEmbedding(experiment_dir)
        application.train(protocol_name, subset=subset, start=start, end=end)

    if arguments['validate']:
        train_dir = arguments['<train_dir>']

        if subset is None:
            subset = 'development'

        aggregate = arguments['--aggregate']
        every = int(arguments['--every'])
        start = int(arguments['--from'])

        application = SpeakerEmbedding.from_train_dir(train_dir)
        application.validate(protocol_name, subset=subset,
                             aggregate=aggregate, every=every,
                             start=start)

    if arguments['apply']:
        validate_txt = arguments['<validate.txt>']
        output_dir = arguments['<output_dir>']
        if subset is None:
            subset = 'test'

        step = arguments['--step']
        if step is not None:
            step = float(step)

        internal = arguments['--internal']

        application = SpeakerEmbedding.from_validate_txt(validate_txt)
        application.apply(protocol_name, output_dir, step=step,
                          internal=internal)
