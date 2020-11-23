# MIT License
#
# Copyright (c) 2020 CNRS
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

from typing import Callable, Iterable, List, Text

from torch.nn import Parameter
from torch.optim import Optimizer
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

from pyannote.audio.core.task import Problem, Scale, Task, TaskSpecification
from pyannote.audio.tasks.segmentation.mixins import SegmentationTaskMixin
from pyannote.database import Protocol


class SpeakerTracking(SegmentationTaskMixin, Task):
    """Speaker tracking

    Speaker tracking is the process of determining if and when a (previously
    enrolled) person's voice can be heard in a given audio recording.

    Here, it is addressed with the same approach as voice activity detection,
    except {"non-speech", "speech"} classes are replaced by {"speaker1", ...,
    "speaker_N"} where N is the number of speakers in the training set.

    Parameters
    ----------
    protocol : Protocol
        pyannote.database protocol
    duration : float, optional
        Chunks duration. Defaults to 2s.
    batch_size : int, optional
        Number of training samples per batch. Defaults to 32.
    num_workers : int, optional
        Number of workers used for generating training samples.
    pin_memory : bool, optional
        If True, data loaders will copy tensors into CUDA pinned
        memory before returning them. See pytorch documentation
        for more details. Defaults to False.
    optimizer : callable, optional
        Callable that takes model parameters as input and returns
        an Optimizer instance. Defaults to `torch.optim.Adam`.
    learning_rate : float, optional
        Learning rate. Defaults to 1e-3.
    augmentation : BaseWaveformTransform, optional
        torch_audiomentations waveform transform, used by dataloader
        during training.
    """

    def __init__(
        self,
        protocol: Protocol,
        duration: float = 2.0,
        batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = False,
        optimizer: Callable[[Iterable[Parameter]], Optimizer] = None,
        learning_rate: float = 1e-3,
        augmentation: BaseWaveformTransform = None,
    ):

        super().__init__(
            protocol,
            duration=duration,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            optimizer=optimizer,
            learning_rate=learning_rate,
            augmentation=augmentation,
        )

        # for speaker tracking, task specification depends
        # on the data: we do not know in advance which
        # speakers should be tracked. therefore, we postpone
        # the definition of specifications.

    def setup(self, stage=None):

        super().setup(stage=stage)

        if stage == "fit":

            # build the list of speakers to be tracked.
            speakers = set()
            for f in self.train:
                speakers.update(f["annotation"].labels())

            # now that we now who the speakers are, we can
            # define the task specifications.

            # note that, since multiple speakers can be active
            # at once, the problem is multi-label classification.
            self.specifications = TaskSpecification(
                problem=Problem.MULTI_LABEL_CLASSIFICATION,
                scale=Scale.FRAME,
                duration=self.duration,
                classes=sorted(speakers),
            )

    @property
    def chunk_labels(self) -> List[Text]:
        """Ordered list of labels

        Used by `prepare_chunk` so that y[:, k] corresponds to activity of kth speaker
        """
        return self.specifications.classes
