# The MIT License (MIT)
#
# Copyright (c) 2019-2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# AUTHOR
# Clément Pagés

from typing import Optional, Union, List
import re
import torch
from torch import nn
import torch.nn.functional as F

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.audio.models.blocks.basic_block import BasicBlock
from pyannote.audio.models.blocks.bottleneck import Bottleneck
from pyannote.audio.utils.params import merge_dict


class ResNet(Model):
    """Implementation of a modified version of a resnet architecture"""

    SINCNET_DEFAULTS = {"stride": 10}

    def __init__(self, block: Union[BasicBlock, Bottleneck],
                    num_blocks: List[int], sample_rate=16000,
                    num_channels=1, m_channels=32, feat_dim=60,
                    embed_dim=512, task: Optional[Task] = None, sincnet=None,
                    seg_part_insert : Union[str, None] = None):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)
        self.inplanes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim

        # Define sincnet module:
        sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        sincnet["sample_rate"] = sample_rate
        self.save_hyperparameters("sincnet", "feat_dim", "embed_dim")
        self.sincnet = SincNet(**self.hparams.sincnet)
        self.seg_part_insert = seg_part_insert
        if block is BasicBlock:
            self.conv1 = nn.Conv2d(
                1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(m_channels)
            self.layer1 = self.__make_layer(
                block, m_channels, num_blocks[0], stride=1)
            self.layer2 = self.__make_layer(
                block, m_channels * 2, num_blocks[1], stride=2)
            current_freq_dim = int((self.feat_dim - 1) / 2) + 1
            self.layer3 = self.__make_layer(
                block, m_channels * 4, num_blocks[2], stride=2)
            current_freq_dim = int((current_freq_dim - 1) / 2) + 1
            self.layer4 = self.__make_layer(
                block, m_channels * 8, num_blocks[3], stride=2)
            current_freq_dim = int((current_freq_dim - 1) / 2) + 1
            self.embedding = nn.Linear(
                m_channels * 8 * 2 * current_freq_dim, embed_dim)
        elif block is Bottleneck:
            self.conv1 = nn.Conv2d(
                1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(m_channels)
            self.layer1 = self.__make_layer(
                block, m_channels, num_blocks[0], stride=1)
            self.layer2 = self.__make_layer(
                block, m_channels * 2, num_blocks[0], stride=2)
            self.layer3 = self.__make_layer(
                block, m_channels * 4, num_blocks[0], stride=2)
            self.layer4 = self.__make_layer(
                block, m_channels * 8, num_blocks[0], stride=2)
            self.embedding = nn.Linear(
                (self.feat_dim/8) * m_channels * 16 * block.expansion, embed_dim)
        else:
            raise ValueError()

    def __make_layer(self, block, planes, num_blocks, stride, ) -> nn.Sequential:
        """"""
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for current_stride in strides:
            layers.append(block(self.inplanes, planes, stride=current_stride))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, waveforms: torch.Tensor):
        """

        Parameters
        ----------
        waveforms : torch.Tensor
            Batch of waveforms with shape (batch, channel, sample)
        """
        filterbank = self.sincnet(waveforms)
        outputs = filterbank.unsqueeze_(1)
        outputs = F.relu(self.bn1(self.conv1(outputs)))
        # To retrieve only layers whose name is of the form "layer1.1" for instance:
        name_filter = re.compile('^layer[0-9]{1,}.[0-9]{1,}$')
        for name, block in [(name, block) for name, block in self.named_modules()
                            if name_filter.match(name)]:
            outputs = block(outputs)
            if self.seg_part_insert is not None and self.seg_part_insert == name:
                seg_part_input = outputs
        # TODO add segmentation part to the model
        pooling_mean = torch.mean(outputs, dim=-1)
        meansq = torch.mean(outputs * outputs, dim=-1)
        pooling_std = torch.sqrt(meansq - pooling_mean ** 2 + 1e-10)
        outputs = torch.cat((torch.flatten(pooling_mean, start_dim=1),
                         torch.flatten(pooling_std, start_dim=1)), 1)
        embedding = self.embedding(outputs)
        return embedding
