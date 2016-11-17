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


class Glue(object):
    """Glue for sequence embedding training

    Parameters
    ----------
    feature_extractor
    duration : float, optional
    min_duration : float, optional
    distance : {'sqeuclidean', 'cosine', 'angular'}, optional
    """

    def __init__(self, feature_extractor,
                 duration=5.0, min_duration=None,
                 distance='sqeuclidean', **kwargs):
        super(Glue, self).__init__()
        self.feature_extractor = feature_extractor
        self.duration = duration
        self.min_duration = min_duration
        self.distance = distance

    def loss(self, y_true, y_pred):
        raise NotImplementedError('')

    def get_generator(self, file_generator, batch_size=None, **kwargs):
        """Get batch generator

        * `batch_generator` must define a `shape` property that returns the
          shape of generated sequences as a (n_samples, n_features) tuple.

        * `batch_generator` must define a method called `get_samples_per_epoch`
          with the signature `def get_samples_per_epoch(self, protocol, subset)`
          that returns the number of samples to generate before ending an epoch.

        * `batch_generator` may optionally define a method called `callbacks`
          with the signature `def callbacks(self, extract_embedding=None)` that
          is expected to return a list of Keras callbacks that will be added to
          the list of callbacks during training. This might come in handy in
          case the `batch_generator` depends on the internal state of the model
          currently being trained.

        Parameters
        ----------
        file_generator : iterable
            e.g. protocol.train()
        batch_size : int, optional
            Batch size

        Returns
        -------
        batch_generator : iterable
            The output of this generator must be a tuple (inputs, targets) or a
            tuple (inputs, targets, sample_weights). All arrays should contain
            the same number of samples. The generator is expected to loop over
            its data indefinitely.
        """
        raise NotImplementedError('')

    def build_model(self, input_shape, design_embedding):
        """Design the model for which the loss is optimized

        Parameters
        ----------
        input_shape: (n_samples, n_features) tuple
            Shape of input sequences.
        design_embedding : function or callable
            This function should take input_shape as input and return a Keras
            model that takes a sequence as input, and returns the embedding as
            output.

        Returns
        -------
        model : Keras model

        See also
        --------
        An example of such a method can be found in `TripletLoss` class
        """
        return design_embedding(input_shape)

    def extract_embedding(self, from_model):
        """Extract embedding from internal Keras model

        Parameters
        ----------
        from_model : Keras model
            Current state of the model

        Returns
        -------
        embedding : Keras model
        """
        return from_model
