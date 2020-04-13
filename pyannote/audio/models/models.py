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
# Juan Manuel Coria

from typing import Optional
from typing import Text

import torch
import torch.nn as nn

from .sincnet import SincNet
from .tdnn import XVectorNet
from .pooling import TemporalPooling


from .convolutional import Convolutional
from .recurrent import Recurrent
from .linear import Linear
from .pooling import Pooling
from .scaling import Scaling

from pyannote.audio.train.model import Model
from pyannote.audio.train.model import Resolution
from pyannote.audio.train.model import RESOLUTION_CHUNK
from pyannote.audio.train.model import RESOLUTION_FRAME
from pyannote.audio.train.task import Task


class RNN(nn.Module):
    """Recurrent layers

    Parameters
    ----------
    n_features : `int`
        Input feature shape.
    unit : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    hidden_size : `int`, optional
        Number of features in the hidden state h. Defaults to 16.
    num_layers : `int`, optional
        Number of recurrent layers. Defaults to 1.
    bias : `boolean`, optional
        If False, then the layer does not use bias weights. Defaults to True.
    dropout : `float`, optional
        If non-zero, introduces a Dropout layer on the outputs of each layer
        except the last layer, with dropout probability equal to dropout.
        Defaults to 0.
    bidirectional : `boolean`, optional
        If True, becomes a bidirectional RNN. Defaults to False.
    concatenate : `boolean`, optional
        Concatenate output of each layer instead of using only the last one
        (which is the default behavior).
    pool : {'sum', 'max', 'last', 'x-vector'}, optional
        Temporal pooling strategy. Defaults to no pooling.
    """

    def __init__(
        self,
        n_features,
        unit="LSTM",
        hidden_size=16,
        num_layers=1,
        bias=True,
        dropout=0,
        bidirectional=False,
        concatenate=False,
        pool=None,
    ):
        super().__init__()

        self.n_features = n_features

        self.unit = unit
        Klass = getattr(nn, self.unit)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.concatenate = concatenate
        self.pool = pool
        self.pool_ = TemporalPooling.create(pool) if pool is not None else None

        if num_layers < 1:
            msg = '"bidirectional" must be set to False when num_layers < 1'
            if bidirectional:
                raise ValueError(msg)
            msg = '"concatenate" must be set to False when num_layers < 1'
            if concatenate:
                raise ValueError(msg)
            return

        if self.concatenate:

            self.rnn_ = nn.ModuleList([])
            for i in range(self.num_layers):

                if i > 0:
                    input_dim = self.hidden_size
                    if self.bidirectional:
                        input_dim *= 2
                else:
                    input_dim = self.n_features

                if i + 1 == self.num_layers:
                    dropout = 0
                else:
                    dropout = self.dropout

                rnn = Klass(
                    input_dim,
                    self.hidden_size,
                    num_layers=1,
                    bias=self.bias,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=self.bidirectional,
                )

                self.rnn_.append(rnn)

        else:
            self.rnn_ = Klass(
                self.n_features,
                self.hidden_size,
                num_layers=self.num_layers,
                bias=self.bias,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            )

    def forward(self, features, return_intermediate=False):
        """Apply recurrent layer (and optional temporal pooling)

        Parameters
        ----------
        features : `torch.Tensor`
            Features shaped as (batch_size, n_frames, n_features)
        return_intermediate : `boolean`, optional
            Return intermediate RNN hidden state.

        Returns
        -------
        output : `torch.Tensor`
            TODO. Shape depends on parameters...
        intermediate : `torch.Tensor`
            (num_layers, batch_size, hidden_size * num_directions)
        """

        if self.num_layers < 1:

            if return_intermediate:
                msg = (
                    '"return_intermediate" must be set to False ' "when num_layers < 1"
                )
                raise ValueError(msg)

            output = features

        else:

            if return_intermediate:
                num_directions = 2 if self.bidirectional else 1

            if self.concatenate:

                if return_intermediate:
                    msg = (
                        '"return_intermediate" is not supported '
                        'when "concatenate" is True'
                    )
                    raise NotADirectoryError(msg)

                outputs = []

                hidden = None
                output = None
                # apply each layer separately...
                for i, rnn in enumerate(self.rnn_):
                    if i > 0:
                        output, hidden = rnn(output, hidden)
                    else:
                        output, hidden = rnn(features)
                    outputs.append(output)

                # ... and concatenate their output
                output = torch.cat(outputs, dim=2)

            else:
                output, hidden = self.rnn_(features)

                if return_intermediate:
                    if self.unit == "LSTM":
                        h = hidden[0]
                    elif self.unit == "GRU":
                        h = hidden

                    # to (num_layers, batch_size, num_directions * hidden_size)
                    h = h.view(self.num_layers, num_directions, -1, self.hidden_size)
                    intermediate = (
                        h.transpose(2, 1)
                        .contiguous()
                        .view(self.num_layers, -1, num_directions * self.hidden_size)
                    )

        if self.pool_ is not None:
            output = self.pool_(output)

        if return_intermediate:
            return output, intermediate

        return output

    def dimension():
        doc = "Output features dimension."

        def fget(self):
            if self.num_layers < 1:
                dimension = self.n_features
            else:
                dimension = self.hidden_size

            if self.bidirectional:
                dimension *= 2

            if self.concatenate:
                dimension *= self.num_layers

            if self.pool == "x-vector":
                dimension *= 2

            return dimension

        return locals()

    dimension = property(**dimension())

    def intermediate_dimension(self, layer):
        if self.num_layers < 1:
            dimension = self.n_features
        else:
            dimension = self.hidden_size

        if self.bidirectional:
            dimension *= 2

        return dimension


class FF(nn.Module):
    """Feedforward layers

    Parameters
    ----------
    n_features : `int`
        Input dimension.
    hidden_size : `list` of `int`, optional
        Linear layers hidden dimensions. Defaults to [16, ].
    """

    def __init__(self, n_features, hidden_size=[16,]):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size

        self.linear_ = nn.ModuleList([])
        for hidden_size in self.hidden_size:
            linear = nn.Linear(n_features, hidden_size, bias=True)
            self.linear_.append(linear)
            n_features = hidden_size

    def forward(self, features):
        """

        Parameters
        ----------
        features : `torch.Tensor`
            (batch_size, n_samples, n_features) or (batch_size, n_features)

        Returns
        -------
        output : `torch.Tensor`
            (batch_size, n_samples, hidden_size[-1]) or (batch_size, hidden_size[-1])
        """

        output = features
        for linear in self.linear_:
            output = linear(output)
            output = torch.tanh(output)
        return output

    def dimension():
        doc = "Output dimension."

        def fget(self):
            if self.hidden_size:
                return self.hidden_size[-1]
            return self.n_features

        return locals()

    dimension = property(**dimension())


class Embedding(nn.Module):
    """Embedding

    Parameters
    ----------
    n_features : `int`
        Input dimension.
    batch_normalize : `boolean`, optional
        Apply batch normalization. This is more or less equivalent to
        embedding whitening.
    scale : {"fixed", "logistic"}, optional
        Scaling method. Defaults to no scaling.
    unit_normalize : deprecated in favor of 'scale'
    """

    def __init__(
        self,
        n_features: int,
        batch_normalize: bool = False,
        scale: Text = None,
        unit_normalize: bool = False,
    ):
        super().__init__()

        self.n_features = n_features

        self.batch_normalize = batch_normalize
        if self.batch_normalize:
            self.batch_normalize_ = nn.BatchNorm1d(
                n_features, eps=1e-5, momentum=0.1, affine=False
            )

        self.scale = scale
        self.scaling = Scaling(n_features, method=scale)

        if unit_normalize is True:
            msg = f"'unit_normalize' has been deprecated in favor of 'scale'."
            raise ValueError(msg)

    def forward(self, embedding):

        if self.batch_normalize:
            embedding = self.batch_normalize_(embedding)

        return self.scaling(embedding)

    @property
    def dimension(self):
        """Output dimension."""
        return self.n_features


class PyanNet(Model):
    """waveform -> SincNet -> RNN [-> merge] [-> time_pool] -> FC -> output

    Parameters
    ----------
    sincnet : `dict`, optional
        SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
        default parameters. Use {'skip': True} to use handcrafted features
        instead of waveforms: [ waveform -> SincNet -> RNN -> ... ] then
        becomes [ features -> RNN -> ...].
    rnn : `dict`, optional
        Recurrent network parameters. Defaults to `RNN` default parameters.
    ff : `dict`, optional
        Feed-forward layers parameters. Defaults to `FF` default parameters.
    embedding : `dict`, optional
        Embedding parameters. Defaults to `Embedding` default parameters. This
        only has effect when model is used for representation learning.
    """

    @staticmethod
    def get_alignment(task: Task, sincnet=None, **kwargs):
        """Get frame alignment"""

        if sincnet is None:
            sincnet = dict()

        if sincnet.get("skip", False):
            return "center"

        return SincNet.get_alignment(task, **sincnet)

    @staticmethod
    def get_resolution(
        task: Task,
        sincnet: Optional[dict] = None,
        rnn: Optional[dict] = None,
        **kwargs,
    ) -> Resolution:
        """Get sliding window used for feature extraction

        Parameters
        ----------
        task : Task
        sincnet : dict, optional
        rnn : dict, optional

        Returns
        -------
        sliding_window : `pyannote.core.SlidingWindow` or {`window`, `frame`}
            Returns RESOLUTION_CHUNK if model returns one vector per input
            chunk, RESOLUTION_FRAME if model returns one vector per input
            frame, and specific sliding window otherwise.
        """

        if rnn is None:
            rnn = {"pool": None}

        if rnn.get("pool", None) is not None:
            return RESOLUTION_CHUNK

        if sincnet is None:
            sincnet = {"skip": False}

        if sincnet.get("skip", False):
            return RESOLUTION_FRAME

        return SincNet.get_resolution(task, **sincnet)

    def init(
        self,
        sincnet: Optional[dict] = None,
        rnn: Optional[dict] = None,
        ff: Optional[dict] = None,
        embedding: Optional[dict] = None,
    ):
        """waveform -> SincNet -> RNN [-> merge] [-> time_pool] -> FC -> output

        Parameters
        ----------
        sincnet : `dict`, optional
            SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
            default parameters. Use {'skip': True} to use handcrafted features
            instead of waveforms: [ waveform -> SincNet -> RNN -> ... ] then
            becomes [ features -> RNN -> ...].
        rnn : `dict`, optional
            Recurrent network parameters. Defaults to `RNN` default parameters.
        ff : `dict`, optional
            Feed-forward layers parameters. Defaults to `FF` default parameters.
        embedding : `dict`, optional
            Embedding parameters. Defaults to `Embedding` default parameters. This
            only has effect when model is used for representation learning.
        """

        n_features = self.n_features

        if sincnet is None:
            sincnet = dict()
        self.sincnet = sincnet

        if not sincnet.get("skip", False):
            if n_features != 1:
                msg = (
                    f"SincNet only supports mono waveforms. "
                    f"Here, waveform has {n_features} channels."
                )
                raise ValueError(msg)
            self.sincnet_ = SincNet(**sincnet)
            n_features = self.sincnet_.dimension

        if rnn is None:
            rnn = dict()
        self.rnn = rnn
        self.rnn_ = RNN(n_features, **rnn)
        n_features = self.rnn_.dimension

        if ff is None:
            ff = dict()
        self.ff = ff
        self.ff_ = FF(n_features, **ff)
        n_features = self.ff_.dimension

        if self.task.is_representation_learning:
            if embedding is None:
                embedding = dict()
            self.embedding = embedding
            self.embedding_ = Embedding(n_features, **embedding)
            return

        self.linear_ = nn.Linear(n_features, len(self.classes), bias=True)
        self.activation_ = self.task.default_activation

    def forward(self, waveforms, return_intermediate=None):
        """Forward pass

        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1) `torch.Tensor`
            Batch of waveforms. In case SincNet is skipped, a tensor with shape
            (batch_size, n_samples, n_features) is expected.
        return_intermediate : `int`, optional
            Index of RNN layer. Returns RNN intermediate hidden state.
            Defaults to only return the final output.

        Returns
        -------
        output : `torch.Tensor`
            Final network output.
        intermediate : `torch.Tensor`
            Intermediate network output (only when `return_intermediate`
            is provided).
        """

        if self.sincnet.get("skip", False):
            output = waveforms
        else:
            output = self.sincnet_(waveforms)

        if return_intermediate is None:
            output = self.rnn_(output)
        else:
            if return_intermediate == 0:
                intermediate = output
                output = self.rnn_(output)
            else:
                return_intermediate -= 1
                # get RNN final AND intermediate outputs
                output, intermediate = self.rnn_(output, return_intermediate=True)
                # only keep hidden state of requested layer
                intermediate = intermediate[return_intermediate]

        output = self.ff_(output)

        if self.task.is_representation_learning:
            return self.embedding_(output)

        output = self.linear_(output)
        output = self.activation_(output)

        if return_intermediate is None:
            return output
        return output, intermediate

    @property
    def dimension(self):
        if self.task.is_representation_learning:
            return self.embedding_.dimension

        return Model.dimension.fget(self)

    def intermediate_dimension(self, layer):
        if layer == 0:
            return self.sincnet_.dimension
        return self.rnn_.intermediate_dimension(layer - 1)


class SincTDNN(Model):
    """waveform -> SincNet -> XVectorNet (TDNN -> FC) -> output

    Parameters
    ----------
    sincnet : `dict`, optional
        SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
        default parameters.
    tdnn : `dict`, optional
        X-Vector Time-Delay neural network parameters.
        Defaults to `pyannote.audio.models.tdnn.XVectorNet` default parameters.
    embedding : `dict`, optional
        Embedding parameters. Defaults to `Embedding` default parameters. This
        only has effect when model is used for representation learning.
    """

    @staticmethod
    def get_alignment(task: Task, sincnet=None, **kwargs):
        """Get frame alignment"""

        if sincnet is None:
            sincnet = dict()

        return SincNet.get_alignment(task, **sincnet)

    supports_packed = False

    @staticmethod
    def get_resolution(
        task: Task, sincnet: Optional[dict] = None, **kwargs
    ) -> Resolution:
        """Get sliding window used for feature extraction

        Parameters
        ----------
        task : Task
        sincnet : dict, optional

        Returns
        -------
        sliding_window : `pyannote.core.SlidingWindow` or {`window`, `frame`}
        """

        # TODO add support for frame-wise and sequence labeling tasks
        # TODO https://github.com/pyannote/pyannote-audio/issues/290
        return RESOLUTION_CHUNK

    def init(
        self,
        sincnet: Optional[dict] = None,
        tdnn: Optional[dict] = None,
        embedding: Optional[dict] = None,
    ):
        """waveform -> SincNet -> XVectorNet (TDNN -> FC) -> output

        Parameters
        ----------
        sincnet : `dict`, optional
            SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
            default parameters.
        tdnn : `dict`, optional
            X-Vector Time-Delay neural network parameters.
            Defaults to `pyannote.audio.models.tdnn.XVectorNet` default parameters.
        embedding : `dict`, optional
            Embedding parameters. Defaults to `Embedding` default parameters. This
            only has effect when model is used for representation learning.
        """

        n_features = self.n_features

        if sincnet is None:
            sincnet = dict()
        self.sincnet = sincnet

        if n_features != 1:
            raise ValueError(
                "SincNet only supports mono waveforms. "
                f"Here, waveform has {n_features} channels."
            )
        self.sincnet_ = SincNet(**sincnet)
        n_features = self.sincnet_.dimension

        if tdnn is None:
            tdnn = dict()
        self.tdnn = tdnn
        self.tdnn_ = XVectorNet(n_features, **tdnn)
        n_features = self.tdnn_.dimension

        if self.task.is_representation_learning:
            if embedding is None:
                embedding = dict()
            self.embedding = embedding
            self.embedding_ = Embedding(n_features, **embedding)
        else:
            self.linear_ = nn.Linear(n_features, len(self.classes), bias=True)
            self.activation_ = self.task.default_activation

    def forward(self, waveforms: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1) `torch.Tensor`
            Batch of waveforms

        Returns
        -------
        output : `torch.Tensor`
            Final network output or intermediate network output
            (only when `return_intermediate` is provided).
        """

        output = self.sincnet_(waveforms)

        return_intermediate = (
            "segment6" if self.task.is_representation_learning else None
        )
        output = self.tdnn_(output, return_intermediate=return_intermediate)

        if self.task.is_representation_learning:
            return self.embedding_(output)

        return self.activation_(self.linear_(output))

    @property
    def dimension(self):
        if self.task.is_representation_learning:
            return self.embedding_.dimension

        return Model.dimension.fget(self)


class ACRoPoLiS(Model):
    """Audio -> Convolutional -> Recurrent (-> optional Pooling) -> Linear -> Scores

    Parameters
    ----------
    specifications : dict
        Task specifications.
    convolutional : dict, optional
        Definition of convolutional layers.
        Defaults to convolutional.Convolutional default hyper-parameters.
    recurrent : dict, optional
        Definition of recurrent layers.
        Defaults to recurrent.Recurrent default hyper-parameters.
    pooling : {"last", ""}, optional
        Definition of pooling layer. Only used when self.task.returns_vector is
        True, in which case it defaults to "last" pooling.
    linear : dict, optional
        Definition of linear layers.
        Defaults to linear.Linear default hyper-parameters.
    scale : dict, optional
    """

    def init(
        self,
        convolutional: dict = None,
        recurrent: dict = None,
        linear: dict = None,
        pooling: Text = None,
        scale: dict = None,
    ):

        self.normalize = nn.InstanceNorm1d(self.n_features)

        if convolutional is None:
            convolutional = dict()
        self.convolutional = convolutional
        self.cnn = Convolutional(self.n_features, **convolutional)

        if recurrent is None:
            recurrent = dict()
        self.recurrent = recurrent
        self.rnn = Recurrent(self.cnn.dimension, **recurrent)

        if pooling is None and self.task.returns_vector:
            pooling = "last"
        if pooling is not None and self.task.returns_sequence:
            msg = f"'pooling' should not be used for labeling tasks (is: {pooling})."
            raise ValueError(msg)

        self.pooling = pooling
        self.pool = Pooling(
            self.rnn.dimension,
            method=self.pooling,
            bidirectional=self.rnn.bidirectional,
        )

        if linear is None:
            linear = dict()
        self.linear = linear
        self.ff = Linear(self.pool.dimension, **linear)

        if self.task.is_representation_learning and scale is None:
            scale = dict()
        if not self.task.is_representation_learning and scale is not None:
            msg = (
                f"'scale' should not be used for representation learning (is: {scale})."
            )
            raise ValueError(msg)

        self.scale = scale
        self.scaling = Scaling(self.ff.dimension, **scale)

        if not self.task.is_representation_learning:
            self.final_linear = nn.Linear(
                self.scaling.dimension, len(self.classes), bias=True
            )
            self.final_activation = self.task.default_activation

    def forward(self, waveforms: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1) `torch.Tensor`
            Batch of waveforms

        Returns
        -------
        output : `torch.Tensor`
            Final network output or intermediate network output
            (only when `return_intermediate` is provided).
        """

        output = self.normalize(waveforms.transpose(1, 2)).transpose(1, 2)
        output = self.cnn(output)
        output = self.rnn(output)
        output = self.pool(output)
        output = self.ff(output)

        if not self.task.is_representation_learning:
            output = self.final_linear(output)
            output = self.final_activation(output)
        return output

    @property
    def dimension(self):
        if self.task.is_representation_learning:
            return self.ff.dimension
        else:
            return len(self.classes)

    @staticmethod
    def get_alignment(task: Task, convolutional: Optional[dict] = None, **kwargs):
        """Get frame alignment"""

        if convolutional is None:
            convolutional = dict()

        return Convolutional.get_alignment(task, **convolutional)

    @staticmethod
    def get_resolution(
        task: Task, convolutional: Optional[dict] = None, **kwargs
    ) -> Resolution:
        """Get frame resolution

        Parameters
        ----------
        task : Task
        convolutional : dict, optional

        Returns
        -------
        sliding_window : `pyannote.core.SlidingWindow` or {`window`, `frame`}
        """

        if task.returns_vector:
            return RESOLUTION_CHUNK

        if convolutional is None:
            convolutional = dict()

        return Convolutional.get_resolution(task, **convolutional)
