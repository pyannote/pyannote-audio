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
import warnings

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

    def fit(self, design_embedding,
            protocol, nb_epoch, train='train',
            optimizer='rmsprop', batch_size=None,
            log_dir=None, validation=['development'],
            max_q_size=1):

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
        train : {'train', 'development', 'test'}, optional
            Defaults to 'train'.
        validation: list, optional
            List of validation subsets among {'train', 'development', 'test'}.
            Defaults to ['development'].
        optimizer: str, optional
            Keras optimizer. Defaults to 'rmsprop'.
        batch_size : int, optional
            Batch size
        log_dir: str, optional
            When provided, log status after each epoch into this directory.
            This will create several files, including loss plots and weights
            files.
        max_q_size: int, optional
            Maximum size for the generator queue. In case the generator depends
            on the current state of the model and/or you cannot run the
            generator in parallel to the model, set max_q_size to 0.

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

        file_generator = getattr(protocol, train)()
        generator = self.glue.get_generator(
            file_generator, batch_size=batch_size)

        # in case the {generator | optimizer | glue} define their own
        # callbacks, append them as well. this might be useful.
        for stuff in [generator, optimizer, self.glue]:
            if hasattr(stuff, 'callbacks'):
                callbacks.extend(stuff.callbacks(
                    extract_embedding=extract_embedding))

        if validation:

            from pyannote.database.protocol.speaker_diarization import \
                SpeakerDiarizationProtocol
            from pyannote.database.protocol.speaker_recognition import \
                SpeakerRecognitionProtocol

            # speaker diarization
            if isinstance(protocol, SpeakerDiarizationProtocol):
                from pyannote.audio.embedding.callbacks import \
                    SpeakerDiarizationValidation
                ValidationCallback = SpeakerDiarizationValidation

            # speaker recognition
            elif isinstance(protocol, SpeakerRecognitionProtocol):
                from pyannote.audio.embedding.callbacks import \
                    SpeakerRecognitionValidation
                ValidationCallback = SpeakerRecognitionValidation

            else:
                warnings.warn(
                    'No validation callback available for this protocol.')

            for subset in validation:
                callback = ValidationCallback(
                    self.glue, protocol, subset, log_dir)
                callbacks.append(callback)

        # if generator has n_labels attribute, pass it to build_model
        n_labels = getattr(generator, 'n_labels', None)
        self.model_ = self.glue.build_model(
            generator.shape, design_embedding, n_labels=n_labels)
        self.model_.compile(optimizer=optimizer, loss=self.glue.loss)

        samples_per_epoch = generator.get_samples_per_epoch(
            protocol, subset=train)

        return self.model_.fit_generator(
            generator, samples_per_epoch, nb_epoch,
            verbose=1, callbacks=callbacks, max_q_size=max_q_size)

    def transform(self, sequences, internal=None, batch_size=32):
        """Apply pre-trained embedding to sequences

        Parameters
        ----------
        sequences : (n_samples, n_frames, n_features) array
            Array of sequences
        internal : int, optional
            Index of layer for which to return the activation.
            Defaults to returning the activation of the final layer (-1).
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

        if internal is not None:
            return self.activation_[internal]([sequences, 0])

        return self.embedding_.predict(sequences, batch_size=batch_size)
