# MIT License
#
# Copyright (c) 2020- CNRS
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


from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn.functional as F
import numpy as np

import pytorch_metric_learning.losses
from pyannote.database import Protocol
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric

from pyannote.audio.core.task import Task

from .mixins import SupervisedRepresentationLearningTaskMixin


class ArcFaceLoss(torch.nn.Module):
    def __init__(self, num_classes, embedding_size, margin, scale):
        """
        ArcFace: Additive Angular Margin Loss for Deep Face Recognition
        (https://arxiv.org/pdf/1801.07698.pdf)
        Args:
            num_classes: The number of classes in your training dataset
            embedding_size: The size of the embeddings that you pass into
            margin: m in the paper, the angular margin penalty in radians
            scale: s in the paper, feature scale
        """


    def __init__(self, num_classes, embedding_size, margin=4, scale=1):
        super().__init__()
        self.margin = margin
        self.num_classes = num_classes
        self.scale = scale
        self.init_margin()
        self.W = torch.nn.Parameter(torch.Tensor(embedding_size, num_classes))
        torch.nn.init.normal_(self.W)
        self.cross_entropy = torch.nn.CrossEntropyLoss()


    def init_margin(self):
        self.margin = np.radians(self.margin)

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (None, embedding_size)
            labels: (None,)
        Returns:
            loss: scalar
        """
        cosine = self.get_cosine(embeddings) # (None, n_classes)
        mask = self.get_target_mask(labels) # (None, n_classes)
        cosine_of_target_classes = cosine[mask == 1] # (None, )
        modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(
            cosine_of_target_classes
        ) # (None, )
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1) # (None,1)
        logits = cosine + (mask * diff) # (None, n_classes)
        logits = self.scale_logits(logits) # (None, n_classes)
        unweighted_loss = self.cross_entropy(logits, labels)
        return unweighted_loss

    def get_cosine(self, embeddings):
        """
        Args:
            embeddings: (None, embedding_size)
        Returns:
            cosine: (None, n_classes)
        """
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.W.t()))
        return cosine

    def get_target_mask(self, labels):
        """
        Args:
            labels: (None,)
        Returns:
            mask: (None, n_classes)
        """
        batch_size = labels.size(0)
        onehot = torch.zeros(batch_size, self.num_classes, device=labels.device)
        onehot.scatter_(1, labels.unsqueeze(-1), 1)
        return onehot

    def get_angles(self, cosine_of_target_classes):
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1, 1))
        return angles

    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        angles = self.get_angles(cosine_of_target_classes)

        # Compute cos of (theta + margin) and cos of theta
        cos_theta_plus_margin = torch.cos(angles + self.margin)
        cos_theta = torch.cos(angles)

        # Keep the cost function monotonically decreasing
        unscaled_logits = torch.where(
            angles <= np.deg2rad(180) - self.margin,
            cos_theta_plus_margin,
            cos_theta - self.margin * np.sin(self.margin),
        )
        return unscaled_logits

    def scale_logits(self, logits):
        """
        Args:
            logits: (None, n_classes)
        Returns:
            scaled_logits: (None, n_classes)
        """
        return logits * self.scale



class SupervisedRepresentationLearningWithArcFace(
    SupervisedRepresentationLearningTaskMixin,
    Task,
):
    """Supervised representation learning with ArcFace loss

    Representation learning is the task of ...

    Parameters
    ----------
    protocol : Protocol
        pyannote.database protocol
    duration : float, optional
        Chunks duration in seconds. Defaults to two seconds (2.).
    min_duration : float, optional
        Sample training chunks duration uniformely between `min_duration`
        and `duration`. Defaults to `duration` (i.e. fixed length chunks).
    num_classes_per_batch : int, optional
        Number of classes per batch. Defaults to 32.
    num_chunks_per_class : int, optional
        Number of chunks per class. Defaults to 1.
    margin : float, optional
        Margin. Defaults to 28.6.
    scale : float, optional
        Scale. Defaults to 64.
    num_workers : int, optional
        Number of workers used for generating training samples.
        Defaults to multiprocessing.cpu_count() // 2.
    pin_memory : bool, optional
        If True, data loaders will copy tensors into CUDA pinned
        memory before returning them. See pytorch documentation
        for more details. Defaults to False.
    augmentation : BaseWaveformTransform, optional
        torch_audiomentations waveform transform, used by dataloader
        during training.
    metric : optional
        Validation metric(s). Can be anything supported by torchmetrics.MetricCollection.
        Defaults to AUROC (area under the ROC curve).
    """

    #  TODO: add a ".metric" property that tells how speaker embedding trained with this approach
    #  should be compared. could be a string like "cosine" or "euclidean" or a pdist/cdist-like
    #  callable. this ".metric" property should be propagated all the way to Inference (via the model).

    def __init__(
        self,
        protocol: Protocol,
        min_duration: Optional[float] = None,
        duration: float = 2.0,
        num_classes_per_batch: int = 32,
        num_chunks_per_class: int = 1,
        margin: float = 28.6,
        scale: float = 64.0,
        num_workers: Optional[int] = None,
        pin_memory: bool = False,
        augmentation: Optional[BaseWaveformTransform] = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
    ):

        self.num_chunks_per_class = num_chunks_per_class
        self.num_classes_per_batch = num_classes_per_batch

        self.margin = margin
        self.scale = scale

        super().__init__(
            protocol,
            duration=duration,
            min_duration=min_duration,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
            metric=metric,
        )

    def setup_loss_func(self):

        _, embedding_size = self.model(self.model.example_input_array).shape

        self.model.loss_func = ArcFaceLoss(
            len(self.specifications.classes),
            embedding_size,
            margin=self.margin,
            scale=self.scale,
        )
