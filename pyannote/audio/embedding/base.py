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

    def fit(self, input_shape, design_embedding, generator,
            samples_per_epoch, nb_epoch, optimizer='rmsprop', log_dir=None):
        """Train the embedding

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
            Keras optimizer. Defaults to 'rmsprop'.
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

        # in case the {generator | optimizer | glue} define their own
        # callbacks, append them as well. this might be useful.
        for stuff in [generator, optimizer, self.glue]:
            if hasattr(stuff, 'callbacks'):
                callbacks.extend(stuff.callbacks(
                    extract_embedding=extract_embedding))

        self.model_ = self.glue.build_model(input_shape, design_embedding)
        self.model_.compile(optimizer=optimizer, loss=self.glue.loss)

        return self.model_.fit_generator(
            generator, samples_per_epoch, nb_epoch,
            verbose=1, callbacks=callbacks)

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
