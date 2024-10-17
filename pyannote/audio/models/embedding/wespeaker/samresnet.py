# Copyright (c) 2024 XiaoyiQin, Yuke Lin (linyuke0609@gmail.com)
#               2024 Shuai Wang (wsstriving@gmail.com)
#               2024 CNRS
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

from pyannote.audio.utils.receptive_field import conv1d_num_frames


class ASP(nn.Module):
    # Attentive statistics pooling
    def __init__(self, in_planes, acoustic_dim):
        super(ASP, self).__init__()
        outmap_size = int(acoustic_dim / 8)
        self.out_dim = in_planes * 8 * outmap_size * 2

        self.attention = nn.Sequential(
            nn.Conv1d(in_planes * 8 * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, in_planes * 8 * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x, weights: Optional[torch.Tensor] = None):
        x = x.reshape(x.size()[0], -1, x.size()[-1])
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
        x = torch.cat((mu, sg), 1)
        x = x.view(x.size()[0], -1)
        return x


class SimAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1, block_id=1):
        super(SimAMBasicBlock, self).__init__()
        self.conv1 = ConvLayer(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = NormLayer(planes)
        self.conv2 = ConvLayer(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = NormLayer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                ConvLayer(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                NormLayer(self.expansion * planes),
            )

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        ...

    def receptive_field_size(self, num_frames: int = 1) -> int:
        ...

    def receptive_field_center(self, frame: int = 0) -> int:
        ...

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.SimAM(out)
        out += self.downsample(x)
        out = self.relu(out)
        return out

    def SimAM(self, X, lambda_p=1e-4):
        n = X.shape[2] * X.shape[3] - 1
        d = (X - X.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        E_inv = d / (4 * (v + lambda_p)) + 0.5
        return X * self.sigmoid(E_inv)


class ResNet(nn.Module):
    def __init__(self, in_planes, block, num_blocks, in_ch=1, **kwargs):
        super(ResNet, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(
            in_ch, in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, in_planes, num_blocks[0], stride=1, block_id=1
        )
        self.layer2 = self._make_layer(
            block, in_planes * 2, num_blocks[1], stride=2, block_id=2
        )
        self.layer3 = self._make_layer(
            block, in_planes * 4, num_blocks[2], stride=2, block_id=3
        )
        self.layer4 = self._make_layer(
            block, in_planes * 8, num_blocks[3], stride=2, block_id=4
        )

    def _make_layer(self, block, planes, num_blocks, stride, block_id=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    nn.Conv2d,
                    nn.BatchNorm2d,
                    self.in_planes,
                    planes,
                    stride,
                    block_id,
                )
            )
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

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class SimAM_ResNet34_ASP(nn.Module):
    def __init__(self, in_planes=64, embed_dim=256, acoustic_dim=80, dropout=0):
        super(SimAM_ResNet34_ASP, self).__init__()
        self.front = ResNet(in_planes, SimAMBasicBlock, [3, 4, 6, 3])
        self.pooling = ASP(in_planes, acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embed_dim)
        self.drop = nn.Dropout(dropout) if dropout else None

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
        out = self.relu(self.bn1(self.conv1(fbank)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    def forward_embedding(
        self, frames: torch.Tensor, weights: Optional[torch.Tensor] = None
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

        out = self.pooling(frames, weights=weights)
        if self.drop:
            out = self.drop(out)
        return self.bottleneck(out)

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

        frames = self.forward_frames(fbank)
        return self.forward_embedding(frames, weights=weights)


if __name__ == "__main__":
    x = torch.zeros(1, 200, 80)
    model = SimAM_ResNet34_ASP(embed_dim=256)
    model.eval()
    out = model(x)
    print(out[-1].size())

    num_params = sum(p.numel() for p in model.parameters())
    print("{} M".format(num_params / 1e6))
