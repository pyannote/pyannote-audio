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


from keras.callbacks import Callback
from pyannote.audio.embedding.base import SequenceEmbedding
from keras.models import model_from_yaml
from pyannote.audio.keras_utils import CUSTOM_OBJECTS


class UpdateGeneratorEmbedding(Callback):

    def __init__(self, generator, extract_embedding, name='embedding'):
        super(UpdateGeneratorEmbedding, self).__init__()
        self.generator = generator
        self.extract_embedding = extract_embedding
        self.name = name

    def _copy_embedding(self, current_model):

        # make a copy of current embedding
        embedding = self.extract_embedding(current_model)
        embedding_copy = model_from_yaml(
            embedding.to_yaml(), custom_objects=CUSTOM_OBJECTS)
        embedding_copy.set_weights(embedding.get_weights())

        # encapsulate it in a SequenceEmbedding instance
        sequence_embedding = SequenceEmbedding()
        sequence_embedding.embedding_ = embedding_copy
        return sequence_embedding

    def on_train_begin(self, logs={}):
        embedding = self._copy_embedding(self.model)
        setattr(self.generator, self.name, embedding)

    def on_batch_begin(self, batch, logs={}):
        embedding = self._copy_embedding(self.model)
        setattr(self.generator, self.name, embedding)
