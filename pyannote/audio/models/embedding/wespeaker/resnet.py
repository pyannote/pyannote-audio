# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
#               2022 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2023 Bing Han (hanbing97@sjtu.edu.cn)
#               2023 CNRS
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

from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pyannote.audio.models.blocks.pooling import StatsPool
from pyannote.audio.utils.receptive_field import (
    conv1d_num_frames,
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
    multi_conv_num_frames,
    multi_conv_receptive_field_center,
    multi_conv_receptive_field_size,
)


class TSTP(nn.Module):
    """
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TSTP, self).__init__()
        self.in_dim = in_dim
        self.stats_pool = StatsPool()

    def forward(self, features, weights: Optional[torch.Tensor] = None):
        """

        Parameters
        ----------
        features : (batch, dimension, channel, frames) torch.Tensor
            Batch of features
        weights: (batch, frames) torch.Tensor, optional
            Batch of weights

        """

        features = rearrange(
            features,
            "batch dimension channel frames -> batch (dimension channel) frames",
        )

        return self.stats_pool(features, weights=weights)

        # # The last dimension is the temporal axis
        # pooling_mean = features.mean(dim=-1)
        # pooling_std = torch.sqrt(torch.var(features, dim=-1) + 1e-7)
        # pooling_mean = pooling_mean.flatten(start_dim=1)
        # pooling_std = pooling_std.flatten(start_dim=1)
        # stats = torch.cat((pooling_mean, pooling_std), 1)
        # return stats

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2
        return self.out_dim


POOLING_LAYERS = {"TSTP": TSTP}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        return multi_conv_num_frames(
            num_samples,
            kernel_size=[3, 3],
            stride=[self.stride, 1],
            padding=[1, 1],
            dilation=[1, 1],
        )

    def receptive_field_size(self, num_frames: int = 1) -> int:
        return multi_conv_receptive_field_size(
            num_frames,
            kernel_size=[3, 3],
            stride=[self.stride, 1],
            padding=[1, 1],
            dilation=[1, 1],
        )

    def receptive_field_center(self, frame: int = 0) -> int:
        return multi_conv_receptive_field_center(
            frame,
            kernel_size=[3, 3],
            stride=[self.stride, 1],
            padding=[1, 1],
            dilation=[1, 1],
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        return multi_conv_num_frames(
            num_samples,
            kernel_size=[1, 3, 1],
            stride=[1, self.stride, 1],
            padding=[0, 1, 0],
            dilation=[1, 1, 1],
        )

    def receptive_field_size(self, num_frames: int = 1) -> int:
        return multi_conv_receptive_field_size(
            num_frames,
            kernel_size=[1, 3, 1],
            stride=[1, self.stride, 1],
            padding=[0, 1, 0],
            dilation=[1, 1, 1],
        )

    def receptive_field_center(self, frame: int = 0) -> int:
        return multi_conv_receptive_field_center(
            frame,
            kernel_size=[1, 3, 1],
            stride=[1, self.stride, 1],
            padding=[0, 1, 0],
            dilation=[1, 1, 1],
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        m_channels=32,
        feat_dim=40,
        embed_dim=128,
        pooling_func="TSTP",
        two_emb_layer=True,
    ):
        super(ResNet, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, m_channels * 8, num_blocks[3], stride=2)

        self.pool = POOLING_LAYERS[pooling_func](
            in_dim=self.stats_dim * block.expansion
        )
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg_1 = nn.Linear(self.pool_out_dim, embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames

        Parameters
        ----------
        num_samples : int
            Number of input samples.

        Returns
        -------
        num_frames : int
            Number of output frames.
        """

        num_frames = num_samples
        num_frames = conv1d_num_frames(
            num_frames, kernel_size=3, stride=1, padding=1, dilation=1
        )
        for layers in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for layer in layers:
                num_frames = layer.num_frames(num_frames)

        return num_frames

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """

        receptive_field_size = num_frames
        for layers in reversed([self.layer1, self.layer2, self.layer3, self.layer4]):
            for layer in reversed(layers):
                receptive_field_size = layer.receptive_field_size(receptive_field_size)

        receptive_field_size = conv1d_receptive_field_size(
            num_frames=receptive_field_size,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )

        return receptive_field_size

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """

        receptive_field_center = frame
        for layers in reversed([self.layer1, self.layer2, self.layer3, self.layer4]):
            for layer in reversed(layers):
                receptive_field_center = layer.receptive_field_center(
                    frame=receptive_field_center
                )

        receptive_field_center = conv1d_receptive_field_center(
            frame=receptive_field_center,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )

        return receptive_field_center

    def forward_frames(self, fbank: torch.Tensor) -> torch.Tensor:
        """Extract frame-wise embeddings

        Parameters
        ----------
        fbanks : (batch, frames, features) torch.Tensor
            Batch of fbank features

        Returns
        -------
        embeddings : (batch, ..., embedding_frames) torch.Tensor
            Batch of frame-wise embeddings.

        """
        fbank = fbank.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        fbank = fbank.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(fbank)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    def forward_embedding(
        self, frames: torch.Tensor, weights: torch.Tensor = None
    ) -> torch.Tensor:
        """Extract speaker embeddings

        Parameters
        ----------
        frames : torch.Tensor
            Batch of frames with shape (batch, ..., embedding_frames).
        weights : (batch, frames) or (batch, speakers, frames) torch.Tensor, optional
            Batch of weights passed to statistics pooling layer.

        Returns
        -------
        embeddings : (batch, dimension) or (batch, speakers, dimension) torch.Tensor
            Batch of embeddings.
        """

        stats = self.pool(frames, weights=weights)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_a, embed_b
        else:
            return torch.tensor(0.0), embed_a

    def forward(self, fbank: torch.Tensor, weights: Optional[torch.Tensor] = None):
        """Extract speaker embeddings

        Parameters
        ----------
        fbank : (batch, frames, features) torch.Tensor
            Batch of features
        weights : (batch, frames) torch.Tensor, optional
            Batch of weights

        Returns
        -------
        embedding : (batch, embedding_dim) torch.Tensor
        """
        fbank = fbank.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        fbank = fbank.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(fbank)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        stats = self.pool(out, weights=weights)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_a, embed_b
        else:
            return torch.tensor(0.0), embed_a


def ResNet18(feat_dim, embed_dim, pooling_func="TSTP", two_emb_layer=True):
    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        two_emb_layer=two_emb_layer,
    )


def ResNet34(feat_dim, embed_dim, pooling_func="TSTP", two_emb_layer=True):
    return ResNet(
        BasicBlock,
        [3, 4, 6, 3],
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        two_emb_layer=two_emb_layer,
    )


def ResNet50(feat_dim, embed_dim, pooling_func="TSTP", two_emb_layer=True):
    return ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        two_emb_layer=two_emb_layer,
    )


def ResNet101(feat_dim, embed_dim, pooling_func="TSTP", two_emb_layer=True):
    return ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        two_emb_layer=two_emb_layer,
    )


def ResNet152(feat_dim, embed_dim, pooling_func="TSTP", two_emb_layer=True):
    return ResNet(
        Bottleneck,
        [3, 8, 36, 3],
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        two_emb_layer=two_emb_layer,
    )


def ResNet221(feat_dim, embed_dim, pooling_func="TSTP", two_emb_layer=True):
    return ResNet(
        Bottleneck,
        [6, 16, 48, 3],
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        two_emb_layer=two_emb_layer,
    )


def ResNet293(feat_dim, embed_dim, pooling_func="TSTP", two_emb_layer=True):
    return ResNet(
        Bottleneck,
        [10, 20, 64, 3],
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        two_emb_layer=two_emb_layer,
    )
