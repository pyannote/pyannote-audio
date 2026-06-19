# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pooling functions to aggregate frame-level deep features
into segment-level speaker embeddings

High-order statistics are surprisingly effective, TSDP acts similarly as TSTP,
even though we remove the mean statistic, on Voxceleb.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _prepare_weights(weights: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Reshape and (linearly) interpolate frame weights to match features.

    Parameters
    ----------
    weights : (batch, frames) torch.Tensor
        Per-frame weights, e.g. the activation of the speaker of interest.
    num_frames : int
        Number of frames in the (temporal) feature axis the weights should match.

    Returns
    -------
    weights : (batch, 1, num_frames) torch.Tensor
        Weights interpolated to `num_frames` and ready to broadcast over channels.

    Note
    ----
    `weights` and the features they apply to might use a different number of frames,
    in which case `weights` are interpolated to reach the number of feature frames.
    """
    weights = weights.unsqueeze(dim=1)
    # (batch, frames) -> (batch, 1, frames)

    if weights.shape[-1] != num_frames:
        weights = F.interpolate(weights, size=num_frames, mode="nearest")

    return weights


class TAP(nn.Module):
    """
    Temporal average pooling, only first-order mean is considered
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TAP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x, weights=None):
        if weights is not None:
            # weighted temporal average over the (last) time axis
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
            w = _prepare_weights(weights, x.shape[-1])
            v1 = w.sum(dim=-1) + 1e-8
            pooling_mean = torch.sum(x * w, dim=-1) / v1
            return pooling_mean.flatten(start_dim=1)
        if x.ndim == 3:
            pooling_mean = x.mean(dim=-1)
        elif x.ndim == 4:
            pooling_mean = x.mean(dim=(-1,-2))
        # To be compatable with 2D input
        pooling_mean = pooling_mean.flatten(start_dim=1)
        return pooling_mean

    def get_out_dim(self):
        self.out_dim = self.in_dim
        return self.out_dim


class TMAP(nn.Module):
    """
    Temporal max-average pooling, only first-order mean is considered
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TMAP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        if x.ndim == 3:
            (x1, _) = torch.max(x, dim=-1)
            x2 = torch.mean(x, dim=-1)
        elif x.ndim == 4:
            # (x1, _) = torch.max(x, dim=(-1,-2))
            x1 = torch.max(
                torch.max(x, dim=-1)[0],
            dim=-1)[0]
            x2 = torch.mean(x, dim=(-1,-2))
        max_avg = x1 + x2
        return max_avg

    def get_out_dim(self):
        self.out_dim = self.in_dim
        return self.out_dim

class TSDP(nn.Module):
    """
    Temporal standard deviation pooling, only second-order std is considered
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TSDP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x, weights=None):
        if weights is not None:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
            w = _prepare_weights(weights, x.shape[-1])
            v1 = w.sum(dim=-1) + 1e-8
            mean = torch.sum(x * w, dim=-1) / v1
            v2 = torch.square(w).sum(dim=-1)
            var = torch.sum(torch.square(x - mean.unsqueeze(-1)) * w, dim=-1) / (
                v1 - v2 / v1 + 1e-8
            )
            pooling_std = torch.sqrt(var.clamp(min=1e-7))
            return pooling_std.flatten(start_dim=1)
        # The last dimension is the temporal axis
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-7)
        pooling_std = pooling_std.flatten(start_dim=1)
        return pooling_std

    def get_out_dim(self):
        self.out_dim = self.in_dim
        return self.out_dim


class TSTP(nn.Module):
    """
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TSTP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x, weights=None):
        if weights is not None:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
            w = _prepare_weights(weights, x.shape[-1])
            v1 = w.sum(dim=-1) + 1e-8
            pooling_mean = torch.sum(x * w, dim=-1) / v1
            v2 = torch.square(w).sum(dim=-1)
            var = torch.sum(
                torch.square(x - pooling_mean.unsqueeze(-1)) * w, dim=-1
            ) / (v1 - v2 / v1 + 1e-8)
            pooling_std = torch.sqrt(var.clamp(min=1e-7))
            return torch.cat(
                (pooling_mean.flatten(start_dim=1), pooling_std.flatten(start_dim=1)), 1
            )
        # The last dimension is the temporal axis
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-7)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_std = pooling_std.flatten(start_dim=1)
        stats = torch.cat((pooling_mean, pooling_std), 1)
        return stats

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2
        return self.out_dim

class TSTP_var(nn.Module):
    """
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TSTP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        # The last dimension is the temporal axis
        pooling_mean = x.mean(dim=-1)
        pooling_var = torch.var(x, dim=-1)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_var = pooling_var.flatten(start_dim=1)
        stats = torch.cat((pooling_mean, pooling_var), 1)
        return stats

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2
        return self.out_dim

class ASTP(nn.Module):
    """ Attentive statistics pooling: Channel- and context-dependent
        statistics pooling, first used in ECAPA_TDNN.
    """

    def __init__(self,
                 in_dim,
                 bottleneck_dim=128,
                 global_context_att=False,
                 **kwargs):
        super(ASTP, self).__init__()
        self.in_dim = in_dim
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't
        # need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(
                in_dim * 3, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                in_dim, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim,
                                 kernel_size=1)  # equals V and k in the paper

    def forward(self, x, weights=None):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
            or a 4-dimensional tensor in resnet architecture (B,C,F,T)
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        weights: an optional (B,T) tensor giving more weight to some frames than
            others (e.g. the activation of the speaker of interest). The attention
            weights are multiplied by `weights` and renormalized over time, so that
            statistics focus on the highly-weighted frames. When `weights` is None,
            this is the standard attentive statistics pooling.
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-7).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! ReLU may be hard to converge.
        alpha = torch.tanh(
            self.linear1(x_in))  # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)

        if weights is not None:
            # give more weight to some frames than others by modulating the
            # attention weights, then renormalize so they still sum to one
            alpha = alpha * _prepare_weights(weights, alpha.shape[-1])
            alpha = alpha / (torch.sum(alpha, dim=2, keepdim=True) + 1e-8)

        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-7))
        return torch.cat([mean, std], dim=1)

    def get_out_dim(self):
        self.out_dim = 2 * self.in_dim
        return self.out_dim


class MHASTP(torch.nn.Module):
    """ Multi head attentive statistics pooling
    Reference:
        Self Multi-Head Attention for Speaker Recognition
        https://arxiv.org/pdf/1906.09890.pdf
    """

    def __init__(self,
                 in_dim,
                 layer_num=2,
                 head_num=2,
                 d_s=1,
                 bottleneck_dim=64,
                 **kwargs):
        super(MHASTP, self).__init__()
        assert (in_dim % head_num
                ) == 0  # make sure that head num can be divided by input_dim
        self.in_dim = in_dim
        self.head_num = head_num
        d_model = int(in_dim / head_num)
        channel_dims = [bottleneck_dim for i in range(layer_num + 1)]
        if d_s > 1:
            d_s = d_model
        else:
            d_s = 1
        self.d_s = d_s
        channel_dims[0], channel_dims[-1] = d_model, d_s
        heads_att_trans = []
        for i in range(self.head_num):
            att_trans = nn.Sequential()
            for i in range(layer_num - 1):
                att_trans.add_module(
                    'att_' + str(i),
                    nn.Conv1d(channel_dims[i], channel_dims[i + 1], 1, 1))
                att_trans.add_module('tanh' + str(i), nn.Tanh())
            att_trans.add_module(
                'att_' + str(layer_num - 1),
                nn.Conv1d(channel_dims[layer_num - 1], channel_dims[layer_num],
                          1, 1))
            heads_att_trans.append(att_trans)
        self.heads_att_trans = nn.ModuleList(heads_att_trans)

    def forward(self, input):
        """
        input: a 3-dimensional tensor in xvector architecture
            or a 4-dimensional tensor in resnet architecture
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(input.shape) == 4:  # B x F x T
            input = input.reshape(input.shape[0],
                                  input.shape[1] * input.shape[2],
                                  input.shape[3])
        assert len(input.shape) == 3
        bs, f_dim, t_dim = input.shape
        chunks = torch.chunk(input, self.head_num, 1)
        # split
        chunks_out = []
        # for i in range(self.head_num):
        #     att_score = self.heads_att_trans[i](chunks[i])
        for i, layer in enumerate(self.heads_att_trans):
            att_score = layer(chunks[i])
            alpha = F.softmax(att_score, dim=-1)
            mean = torch.sum(alpha * chunks[i], dim=2)
            var = torch.sum(alpha * chunks[i]**2, dim=2) - mean**2
            std = torch.sqrt(var.clamp(min=1e-7))
            chunks_out.append(torch.cat((mean, std), dim=1))
        out = torch.cat(chunks_out, dim=1)
        return out

    def get_out_dim(self):
        self.out_dim = 2 * self.in_dim
        return self.out_dim


class MQMHASTP(torch.nn.Module):
    """ An attentive pooling
    Reference:
        multi query multi head attentive statistics pooling
        https://arxiv.org/pdf/2110.05042.pdf
    Args:
        in_dim: the feature dimension of input
        layer_num: the number of layer in the pooling layer
        query_num: the number of querys
        head_num: the number of heads
        bottleneck_dim: the bottleneck dimension

    SA (H = 1, Q = 1, n = 2, d_s = 1) ref:
        https://www.danielpovey.com/files/2018_interspeech_xvector_attention.pdf
    MHA (H > 1, Q = 1, n = 1, d_s = 1) ref:
        https://arxiv.org/pdf/1906.09890.pdf
    AS (H = 1, Q > 1, n = 2, d_s = 1) ref:
        https://arxiv.org/pdf/1803.10963.pdf
    VSA (H = 1, Q > 1, n = 2, d_s = d_h) ref:
        http://www.interspeech2020.org/uploadfile/pdf/Mon-2-10-5.pdf
    """

    def __init__(self,
                 in_dim,
                 layer_num=2,
                 query_num=2,
                 head_num=8,
                 d_s=2,
                 bottleneck_dim=64,
                 **kwargs):
        super(MQMHASTP, self).__init__()
        self.n_query = nn.ModuleList([
            MHASTP(in_dim,
                   layer_num=layer_num,
                   head_num=head_num,
                   d_s=d_s,
                   bottleneck_dim=bottleneck_dim) for i in range(query_num)
        ])
        self.query_num = query_num
        self.in_dim = in_dim

    def forward(self, input):
        """
        input: a 3-dimensional tensor in xvector architecture
            or a 4-dimensional tensor in resnet architecture
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(input.shape) == 4:  # B x F x T
            input = input.reshape(input.shape[0],
                                  input.shape[1] * input.shape[2],
                                  input.shape[3])
        assert len(input.shape) == 3
        res = []
        for i, layer in enumerate(self.n_query):
            res.append(layer(input))
        out = torch.cat(res, dim=-1)
        return out

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2 * self.query_num
        return self.out_dim


if __name__ == '__main__':
    data = torch.randn(16, 512, 10, 35)
    # model = StatisticsPooling()
    model = MQMHASTP(512 * 10)
    model = MHASTP(512 * 10)
    model = MQMHASTP(512 * 10, context=False)
    print(model)

    out = model(data)
    print(out.shape)
    print(model.get_out_dim())
