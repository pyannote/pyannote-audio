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

import os.path

import keras.backend as K
from pyannote.audio.callback import LoggingCallback
from keras.models import model_from_yaml
from pyannote.audio.keras_utils import CUSTOM_OBJECTS

import time
import numpy as np
import warnings
import os.path
import datetime
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.spatial.distance import pdist
from pyannote.metrics.plot.binary_classification import plot_det_curve, plot_distributions


class SequenceEmbedding(object):
    """Sequence embedding

    Parameters
    ----------
    glue : pyannote.audio.embedding.glue.Glue
        `Glue` instance. It is expected to implement the following methods:
        loss, build_model, and extract_embedding

    See also
    --------
    pyannote.audio.embedding.glue.Glue for more details on `glue` parameter
    """
    def __init__(self, glue=None):
        super(SequenceEmbedding, self).__init__()
        self.glue = glue

    @classmethod
    def from_disk(cls, architecture, weights):
        """Load pre-trained sequence embedding from disk

        Parameters
        ----------
        architecture : str
            Path to architecture file (e.g. created by `to_disk` method)
        weights : str
            Path to pre-trained weight file (e.g. created by `to_disk` method)

        Returns
        -------
        sequence_embedding : SequenceEmbedding
            Pre-trained sequence embedding model.
        """
        self = SequenceEmbedding()

        with open(architecture, 'r') as fp:
            yaml_string = fp.read()
        self.embedding_ = model_from_yaml(
            yaml_string, custom_objects=CUSTOM_OBJECTS)
        self.embedding_.load_weights(weights)
        return self

    def to_disk(self, architecture=None, weights=None, overwrite=False):
        """Save trained sequence embedding to disk

        Parameters
        ----------
        architecture : str, optional
            When provided, path where to save architecture.
        weights : str, optional
            When provided, path where to save weights
        overwrite : boolean, optional
            Overwrite (architecture or weights) file in case they exist.
        """

        if not hasattr(self, 'model_'):
            raise AttributeError('Model must be trained first.')

        if architecture and os.path.isfile(architecture) and not overwrite:
            raise ValueError("File '{architecture}' already exists.".format(architecture=architecture))

        if weights and os.path.isfile(weights) and not overwrite:
            raise ValueError("File '{weights}' already exists.".format(weights=weights))

        embedding = self.glue.extract_embedding(self.model_)

        if architecture:
            yaml_string = embedding.to_yaml()
            with open(architecture, 'w') as fp:
                fp.write(yaml_string)

        if weights:
            embedding.save_weights(weights, overwrite=overwrite)

    def fit(self, design_embedding,
            protocol, nb_epoch, subset='train',
            optimizer='rmsprop', batch_size=None,
            log_dir=None):

        """Train the embedding

        Parameters
        ----------
        design_embedding : function or callable
            This function should take input_shape as input and return a Keras
            model that takes a sequence as input, and returns the embedding as
            output.
        protocol : pyannote.database.Protocol

        nb_epoch : int
            Total number of iterations on the data
        subset : {'train', 'development', 'test'}, optional
            Defaults to 'train'.
        optimizer: str, optional
            Keras optimizer. Defaults to 'rmsprop'.
        batch_size : int, optional
            Batch size
        log_dir: str, optional
            When provided, log status after each epoch into this directory.
            This will create several files, including loss plots and weights
            files.

        See also
        --------
        keras.engine.training.Model.fit_generator
        """

        callbacks = []

        extract_embedding = self.glue.extract_embedding

        if log_dir is not None:
            callback = LoggingCallback(
                log_dir, extract_embedding=extract_embedding)
            callbacks.append(callback)

        file_generator = getattr(protocol, subset)()
        generator = self.glue.get_generator(
            file_generator, batch_size=batch_size)

        # in case the {generator | optimizer | glue} define their own
        # callbacks, append them as well. this might be useful.
        for stuff in [generator, optimizer, self.glue]:
            if hasattr(stuff, 'callbacks'):
                callbacks.extend(stuff.callbacks(
                    extract_embedding=extract_embedding))

        # if generator has n_labels attribute, pass it to build_model
        nlabels = getattr(generator, 'n_labels', None)
        self.model_ = self.glue.build_model(
            generator.shape, design_embedding, n_labels=n_labels)
        self.model_.compile(optimizer=optimizer, loss=self.glue.loss)

        samples_per_epoch = generator.get_samples_per_epoch(
            protocol, subset=subset)

        return self.model_.fit_generator(
            generator, samples_per_epoch, nb_epoch,
            verbose=1, callbacks=callbacks)

    def fastfit(self, design_embedding, generator_train, nb_epoch,
            nb_batches_per_epoch, batch_size, per_label, nb_of_threads=12,
            generator_test=None, optimizer='rmsprop', LOG_DIR=None):
        """Train the embedding faster

        Parameters
        ----------
        input_shape : (n_frames, n_features) tuple
            Shape of input sequence
        design_embedding : function or callable
            This function should take input_shape as input and return a Keras
            model that takes a sequence as input, and returns the embedding as
            output.
        generator : iterable
            The output of the generator must be a tuple (inputs, targets) or a
            tuple (inputs, targets, sample_weights). All arrays should contain
            the same number of samples. The generator is expected to loop over
            its data indefinitely. An epoch finishes when `samples_per_epoch`
            samples have been seen by the model.
        samples_per_epoch : int
            Number of samples to process before going to the next epoch.
        nb_epoch : int
            Total number of iterations on the data
        optimizer: str, optional
            Keras optimizer. Defaults to 'SSMORMS3'.
        log_dir: str, optional
            When provided, log status after each epoch into this directory.
            This will create several files, including loss plots and weights
            files.

        See also
        --------
        keras.engine.training.Model.fit_generator
        """
        # create log_dir directory (and subdirectory)
        os.makedirs(LOG_DIR)
        os.makedirs(LOG_DIR + '/weights')

        # write value to file
        LOGPATH = LOG_DIR + '/{name}.{subset}.log'

        start_time = time.time()
        # generate set of labeled sequences
        X, y = zip(*generator_test)
        X, y = np.vstack(X), np.hstack(y)
        # X = np.reshape(X, (X.shape[0], X.shape[1]/5, X.shape[2]*5))
        print("Test database Loading --- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        # randomly select (at most) 100 sequences from each speaker to ensure
        # all speakers have the same importance in the evaluation
        unique, y, counts = np.unique(y, return_inverse=True, return_counts=True)
        n_speakers = len(unique)
        indices = []
        for speaker in range(n_speakers):
            i = np.random.choice(np.where(y == speaker)[0], size=min(100, counts[speaker]), replace=False)
            indices.append(i)
        indices = np.hstack(indices)
        Xtest, ytest = X[indices], y[indices, np.newaxis]
        print("Test database random selection --- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        # generate training set of labeled sequences
        X, y = zip(*generator_train)
        X, y2 = np.vstack(X), np.hstack(y)
        # X = np.reshape(X, (X.shape[0], X.shape[1]/5, X.shape[2]*5))
        n_seqs = y2.size
        print("Train database with %d files loading --- %s seconds ---" % (n_seqs, time.time() - start_time))

        # unique labels
        unique, y, counts = np.unique(y2, return_inverse=True, return_counts=True)
        n_labels = len(unique)

        # shuffle labels
        shuffled_labels = np.random.choice(n_labels,
                                           size=n_labels,
                                           replace=False)
        
        random_indices = {}
        for ii in range(n_labels):
            label = shuffled_labels[ii]
            tmp = np.where(y == label)[0]
            ind = np.random.choice(tmp,
                                size=len(tmp),
                                replace=False)
            if (len(ind) > 0):
                random_indices[label] = [ind, 0, len(ind)]
            else:
                print 'Problem in paradise'

        # warn that some labels have very few training samples
        too_few_samples = np.sum(counts < 10)
        if too_few_samples > 0:
            msg = '{n} labels (out of {N}) have less than {per_label} training samples.'
            warnings.warn(msg.format(n=too_few_samples, N=10, per_label=per_label))

        # Model Building
        self.model_ = self.glue.build_model(X[0].shape, design_embedding, n_labels)
        self.model_.compile(optimizer=optimizer, loss=self.glue.loss)
        self.model_.summary()

        # writing model to disk
        architecture = LOG_DIR + '/architecture.yml'
        self.to_disk(architecture=architecture)

        loss_train = []
        eer_test = []
        label_pos = 0
        pool = mp.Pool(nb_of_threads)
        for epoch in range(nb_epoch):
            start_time_epoch = time.time()
            for ii in range(nb_batches_per_epoch):
                start_time_batch = time.time()
                indices = []
                count_per_label = 0
                while (len(indices) < nb_of_threads*batch_size):
                    indices.append(random_indices[shuffled_labels[label_pos]][0][random_indices[shuffled_labels[label_pos]][1]])
                    
                    random_indices[shuffled_labels[label_pos]][1] += 1
                    if (random_indices[shuffled_labels[label_pos]][1] >= random_indices[shuffled_labels[label_pos]][2]):
                        random_indices[shuffled_labels[label_pos]][1] = 0
                    
                    count_per_label += 1
                    if (count_per_label >= per_label):
                        count_per_label = 0
                        label_pos += 1
                        if (label_pos >= shuffled_labels.size):
                            label_pos = 0
                            shuffled_labels = np.random.choice(n_labels,
                                                       size=n_labels,
                                                       replace=False)
                indices = np.hstack(indices)

                # selected sequences
                sequences = X[indices]
                batch_labels = y[indices]

                # their embeddings (using current state of embedding network)
                start_time = time.time()
                embeddings = self.model_.predict(sequences, batch_size=nb_of_threads*batch_size)
                pred_dur = (time.time() - start_time)

                start_time = time.time()
                [costs, derivatives] = self.glue.compute_cost_and_derivatives(embeddings, batch_labels, batch_size, nb_of_threads, pool)
                deriv_dur = (time.time() - start_time)

                start_time = time.time()
                loss_value = self.model_.train_on_batch([sequences], derivatives)
                batch_dur = (time.time() - start_time)

                overall_batch_dur = (time.time() - start_time_batch)
                batch_cost = np.sum(costs)

                TXT_TEMPLATE = 'Predict --- {pred_dur:.3f} seconds ---\n'
                TXT_TEMPLATE += 'Triplet derivation --- {deriv_dur:.3f} seconds ---\n'
                TXT_TEMPLATE += 'Train on batch --- {batch_dur:.3f} seconds ---\n'
                TXT_TEMPLATE += 'Epoch {epoch:04d}, batch {ii:04d} / {nb_of_batch_in_epoch:04d} processed in {overall_batch_dur:.3f}'
                TXT_TEMPLATE += ' seconds with cost value of {batch_cost:.5g}\n\n'

                log_string = TXT_TEMPLATE.format(pred_dur=pred_dur, deriv_dur=deriv_dur,\
                    batch_dur=batch_dur, epoch=epoch, ii=ii+1, nb_of_batch_in_epoch=nb_batches_per_epoch,\
                    overall_batch_dur=overall_batch_dur, batch_cost=batch_cost)
                print log_string

                mode = 'w' if ((epoch == 0)and(ii == 0)) else 'a'
                try:
                    log_filename = LOGPATH.format(subset='train', name='details')
                    with open(log_filename, mode) as fp:
                        fp.write(log_string)
                        fp.flush()
                except Exception as e:
                    pass
                try:
                    log_filename = LOGPATH.format(subset='train', name='loss')
                    now = datetime.datetime.now().isoformat()
                    TXT_TEMPLATE = '{epoch:d} {ii:d} ' + now + ' {value:.5g}\n'
                    log_string = TXT_TEMPLATE.format(epoch=epoch, ii=ii, value=batch_cost)
                    with open(log_filename, mode) as fp:
                        fp.write(log_string)
                        fp.flush()
                except Exception as e:
                    print e
                    pass

                # keep track of cost after last batch
                loss_train.append(batch_cost)
                best_batch = np.argmin(loss_train)
                best_value = np.min(loss_train)

                # plot values to file and mark best value so far
                fig = plt.figure()
                plt.semilogy(loss_train, 'b')
                plt.semilogy([best_batch], [best_value], 'bo')
                plt.grid(True)

                plt.xlabel('batch')
                plt.ylabel('loss on train')

                TITLE = 'loss = {best_value:.5g} on train @ batch #{best_batch:d}'
                title = TITLE.format(best_value=best_value, best_batch=best_batch)
                plt.title(title)

                plt.tight_layout()

                # save plot as PNG
                try:
                    plt.savefig(LOG_DIR + '/train.loss.png', dpi=150)
                except Exception as e:
                    pass
                plt.close(fig)

            epoch_dur = (time.time() - start_time_epoch)

            PATH = LOG_DIR+'/weights/weights-{epoch:02d}.hdf5'
            PATH = PATH.format(epoch=epoch)
            self.to_disk(weights=PATH)

            # testing
            fX = self.model_.predict([Xtest])

            # def eval_eer(fX, ytest, LOG_DIR, LOGPATH, epoch, epoch_dur):
            start_time = time.time()
            # compute euclidean distance between every pair of sequences
            distances = np.arccos(np.clip(1.0-pdist(fX, metric='cosine'), -1.0, 1.0))
            # distances = pdist(fX, metric='euclidean')

            # compute same/different groundtruth
            y_true = pdist(ytest, metric='chebyshev') < 1

            # plot positive/negative scores distribution
            # plot DET curve and return equal error rate
            prefix = LOG_DIR + '/plot.{epoch:04d}'.format(epoch=epoch)
            plot_distributions(y_true, distances, prefix, xlim=(0, np.pi), ymax=3, nbins=100)
            eer = plot_det_curve(y_true, -distances, prefix)
            eer *= 100.0
            eer_dur = (time.time() - start_time)

            TXT_TEMPLATE = 'Epoch #{epoch:05d} processed in {epoch_dur:.3f}'
            TXT_TEMPLATE += ' seconds with EER={eer:.5g} on test dataset ---\n'
            TXT_TEMPLATE += 'EER evaluation --- {eer_dur:.3f} seconds ---\n\n'

            log_string = TXT_TEMPLATE.format(epoch=epoch, epoch_dur=epoch_dur,\
                eer=eer, eer_dur=eer_dur)
            print log_string

            mode = 'w' if (epoch == 0) else 'a'
            try:
                log_filename = LOGPATH.format(subset='train', name='details')
                with open(log_filename, mode) as fp:
                    fp.write(log_string)
                    fp.flush()
            except Exception as e:
                pass
            try:
                log_filename = LOGPATH.format(subset='test', name='eer')
                now = datetime.datetime.now().isoformat()
                TXT_TEMPLATE = '{epoch:d} ' + now + ' {value:.5g}\n'
                log_string = TXT_TEMPLATE.format(epoch=epoch, value=eer)
                with open(log_filename, mode) as fp:
                    fp.write(log_string)
                    fp.flush()
            except Exception as e:
                print e
                pass

            # keep track of cost after last batch
            eer_test.append(eer)
            best_epoch = np.argmin(eer_test)
            best_value = np.min(eer_test)

            # plot values to file and mark best value so far
            fig = plt.figure()
            plt.plot(eer_test, 'b')
            plt.plot([best_epoch], [best_value], 'bo')
            plt.grid(True)

            plt.xlabel('epoch')
            plt.ylabel('EER on test')

            TITLE = 'EER = {best_value:.5g} on test @ epoch #{best_epoch:d}'
            title = TITLE.format(best_value=best_value, best_epoch=best_epoch)
            plt.title(title)

            plt.tight_layout()

            # save plot as PNG
            try:
                plt.savefig(LOG_DIR + '/test.eer.png', dpi=150)
            except Exception as e:
                pass
            plt.close(fig)


    def transform(self, sequences, layer_index=None, batch_size=32):
        """Apply pre-trained embedding to sequences

        Parameters
        ----------
        sequences : (n_samples, n_frames, n_features) array
            Array of sequences
        layer_index : int, optional
            Index of layer for which to return the activation.
            Defaults to returning the activation of the final layer.
        batch_size : int, optional
            Number of samples per batch
        verbose : int, optional

        Returns
        -------
        embeddings : (n_samples, n_dimensions)
        """

        if not hasattr(self, 'embedding_'):
            self.embedding_ = self.glue.extract_embedding(self.model_)

        if not hasattr(self, 'activation_'):
            self.activation_ = []
            for layer in self.embedding_.layers:
                func = K.function(
                    [self.embedding_.layers[0].input, K.learning_phase()],
                    layer.output)
                self.activation_.append(func)

        if layer_index is not None:
            return self.activation_[layer_index]([sequences, 0])

        return self.embedding_.predict(sequences, batch_size=batch_size)
