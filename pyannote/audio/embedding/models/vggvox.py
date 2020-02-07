#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018 CNRS

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

# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
# Update VGGVox to latest API. Commenting it out for now.
# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .utils import get_conv2d_output_shape


# class VGGVox(nn.Module):
#     """VGGVox implementation
#
#     Reference
#     ---------
#     Arsha Nagrani, Joon Son Chung, Andrew Zisserman. "VoxCeleb: a large-scale
#     speaker identification dataset."
#
#     """
#
#     def __init__(self, specifications, dimension=256):
#
#         super().__init__()
#         self.specifications = specifications
#         self.n_features_ = specifications['X']['dimension']
#
#         if self.n_features_ < 97:
#             msg = (f'VGGVox expects features with at least 97 dimensions '
#                    f'(here, n_features = {self.n_features_:d})')
#             raise ValueError(msg)
#
#         self.dimension = dimension
#
#         h = self.n_features_  # 512 in VoxCeleb paper. 201 in practice.
#         w = 301 # typically 3s with 10ms steps
#
#         self.conv1_ = nn.Conv2d(1, 96, (7, 7), stride=(2, 2), padding=1)
#         # 254 x 148 when n_features = 512
#         # 99 x 148 when n_features = 201
#         h, w = get_conv2d_output_shape((h, w), (7, 7), stride=(2, 2), padding=1)
#
#         self.bn1_ = nn.BatchNorm2d(96)
#         self.mpool1_ = nn.MaxPool2d((3, 3), stride=(2, 2))
#         # 126 x 73 when n_features = 512
#         # 49 x 73 when n_features = 201
#         h, w = get_conv2d_output_shape((h, w), (3, 3), stride=(2, 2))
#
#         self.conv2_ = nn.Conv2d(96, 256, (5, 5), stride=(2, 2), padding=1)
#         # 62 x 36 when n_features = 512
#         # 24 x 36 when n_features = 201
#         h, w = get_conv2d_output_shape((h, w), (5, 5), stride=(2, 2), padding=1)
#
#         self.bn2_ = nn.BatchNorm2d(256)
#         self.mpool2_ = nn.MaxPool2d((3, 3), stride=(2, 2))
#         # 30 x 17 when n_features = 512
#         # 11 x 17 when n_features = 201
#         h, w = get_conv2d_output_shape((h, w), (3, 3), stride=(2, 2))
#
#         self.conv3_ = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1)
#         # 30 x 17 when n_features = 512
#         # 11 x 17 when n_features = 201
#         h, w = get_conv2d_output_shape((h, w), (3, 3), stride=(1, 1), padding=1)
#
#         self.bn3_ = nn.BatchNorm2d(256)
#
#         self.conv4_ = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1)
#         # 30 x 17 when n_features = 512
#         # 11 x 17 when n_features = 201
#         h, w = get_conv2d_output_shape((h, w), (3, 3), stride=(1, 1), padding=1)
#
#         self.bn4_ = nn.BatchNorm2d(256)
#
#         self.conv5_ = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1)
#         # 30 x 17 when n_features = 512
#         # 11 x 17 when n_features = 201
#         h, w = get_conv2d_output_shape((h, w), (3, 3), stride=(1, 1), padding=1)
#
#         self.bn5_ = nn.BatchNorm2d(256)
#
#         self.mpool5_ = nn.MaxPool2d((5, 3), stride=(3, 2))
#         # 9 x 8 when n_features = 512
#         # 3 x 8 when n_features = 201
#         h, w = get_conv2d_output_shape((h, w), (5, 3), stride=(3, 2))
#
#         self.fc6_ = nn.Conv2d(256, 4096, (h, 1), stride=(1, 1))
#         # 1 x 8
#         h, w = get_conv2d_output_shape((h, w), (h, 1), stride=(1, 1))
#
#         self.fc7_ = nn.Linear(4096, 1024)
#         self.fc8_ = nn.Linear(1024, self.dimension)
#
#
#     def forward(self, sequences):
#         """Embed sequences
#
#         Parameters
#         ----------
#         sequences : torch.Tensor (batch_size, n_samples, n_features)
#             Batch of sequences.
#
#         Returns
#         -------
#         embeddings : torch.Tensor (batch_size, dimension)
#             Batch of embeddings.
#         """
#
#         # reshape batch to (batch_size, n_channels, n_features, n_samples)
#         batch_size, n_samples, n_features = sequences.size()
#
#         if n_features != self.n_features_:
#             msg = (f'Mismatch in feature dimension '
#                    f'(should be: {self.n_features_:d}, is: {n_features:d})')
#             raise ValueError(msg)
#
#         if n_samples < 65:
#             msg = (f'VGGVox expects sequences with at least 65 samples '
#                    f'(here, n_samples = {n_samples:d})')
#             raise ValueError(msg)
#
#         x = torch.transpose(sequences, 1, 2).view(
#             batch_size, 1, n_features, n_samples)
#
#         # conv1. shape => 254 x 148 => 126 x 73
#         x = self.mpool1_(F.relu(self.bn1_(self.conv1_(x))))
#
#         # conv2. shape =>
#         x = self.mpool2_(F.relu(self.bn2_(self.conv2_(x))))
#
#         # conv3. shape = 62 x 36
#         x = F.relu(self.bn3_(self.conv3_(x)))
#
#         # conv4. shape = 30 x 17
#         x = F.relu(self.bn4_(self.conv4_(x)))
#
#         # conv5. shape = 30 x 17
#         x = self.mpool5_(F.relu(self.bn5_(self.conv5_(x))))
#
#         # fc6. shape =
#         x = F.dropout(F.relu(self.fc6_(x)))
#
#         # (average) temporal pooling. shape =
#         x = torch.mean(x, dim=-1)
#
#         # fc7. shape =
#         x = x.view(x.size(0), -1)
#         x = F.dropout(F.relu(self.fc7_(x)))
#
#         # fc8. shape =
#         x = self.fc8_(x)
#
#         return x
