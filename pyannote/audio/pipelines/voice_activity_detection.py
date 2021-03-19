# MIT License
#
# Copyright (c) 2018-2021 CNRS
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

"""Voice activity detection pipelines"""

import tempfile
from copy import deepcopy
from types import MethodType
from typing import Union

import numpy as np
from pytorch_lightning import Trainer
from torch.optim import SGD
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

from pyannote.audio import Inference
from pyannote.audio.core.callback import GraduallyUnfreeze
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.utils import (
    PipelineAugmentation,
    PipelineInference,
    PipelineModel,
    get_augmentation,
    get_devices,
    get_inference,
    get_model,
)
from pyannote.audio.tasks import VoiceActivityDetection as VoiceActivityDetectionTask
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, SlidingWindowFeature
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.metrics.detection import (
    DetectionErrorRate,
    DetectionPrecisionRecallFMeasure,
)
from pyannote.pipeline.parameter import Categorical, Integer, LogUniform, Uniform


class OracleVoiceActivityDetection(Pipeline):
    """Oracle voice activity detection pipeline"""

    @staticmethod
    def apply(file: AudioFile) -> Annotation:
        """Return groundtruth voice activity detection

        Parameter
        ---------
        file : AudioFile
            Must provide a "annotation" key.

        Returns
        -------
        hypothesis : `pyannote.core.Annotation`
            Speech regions
        """

        speech = file["annotation"].get_timeline().support()
        return speech.to_annotation(generator="string", modality="speech")


class VoiceActivityDetection(Pipeline):
    """Voice activity detection pipeline

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation (or voice activity detection) model.
        Defaults to "hbredin/VoiceActivityDetection-PyanNet-DIHARD".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    batch_size : int, optional
        Batch size. Defaults to 32.
    fscore : bool, optional
        Optimize (precision/recall) fscore. Defaults to optimizing detection
        error rate.

    Hyper-parameters
    ----------------
    onset, offset : float
        Onset/offset detection thresholds
    min_duration_on : float
        Remove speech regions shorter than that many seconds.
    min_duration_off : float
        Fill non-speech regions shorter than that many seconds.
    """

    def __init__(
        self,
        segmentation: PipelineModel = "hbredin/VoiceActivityDetection-PyanNet-DIHARD",
        batch_size: int = 32,
        fscore: bool = False,
    ):
        super().__init__()

        self.segmentation = segmentation
        self.batch_size = batch_size
        self.fscore = fscore

        # load model and send it to GPU (when available and not already on GPU)
        model = get_model(segmentation)
        if model.device.type == "cpu":
            (segmentation_device,) = get_devices(needs=1)
            model.to(segmentation_device)

        self.segmentation_inference_ = Inference(
            model,
            batch_size=self.batch_size,
        )

        #  hyper-parameters used for hysteresis thresholding
        self.onset = Uniform(0.0, 1.0)
        self.offset = Uniform(0.0, 1.0)

        # hyper-parameters used for post-processing i.e. removing short speech regions
        # or filling short gaps between speech regions
        self.min_duration_on = Uniform(0.0, 1.0)
        self.min_duration_off = Uniform(0.0, 1.0)

    def initialize(self):
        """Initialize pipeline with current set of parameters"""

        self._binarize = Binarize(
            onset=self.onset,
            offset=self.offset,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )

    def apply(self, file: AudioFile) -> Annotation:
        """Apply voice activity detection

        Parameters
        ----------
        file : AudioFile
            Processed file.

        Returns
        -------
        speech : `pyannote.core.Annotation`
            Speech regions.
        """

        segmentation = self.segmentation_inference_(file)

        if segmentation.data.shape[1] == 1:
            file["@voice_activity_detection/activation"] = segmentation
            activation = segmentation
        else:
            file["@voice_activity_detection/segmentation"] = segmentation
            activation = SlidingWindowFeature(
                np.max(segmentation.data, axis=1, keepdims=True),
                segmentation.sliding_window,
            )

        speech = self._binarize(activation)
        speech.uri = file["uri"]
        return speech

    def get_metric(self) -> Union[DetectionErrorRate, DetectionPrecisionRecallFMeasure]:
        """Return new instance of detection metric"""

        if self.fscore:
            return DetectionPrecisionRecallFMeasure(collar=0.0, skip_overlap=False)

        return DetectionErrorRate(collar=0.0, skip_overlap=False)

    def get_direction(self):
        if self.fscore:
            return "maximize"
        return "minimize"


class AdaptiveVoiceActivityDetection(Pipeline):
    """Adaptive voice activity detection pipeline

    Let M be a pretrained voice activity detection model.

    For each file f, this pipeline starts by applying the model to obtain a first set of
    speech/non-speech labels.

    Those (automatic, possibly erroneous) labels are then used to fine-tune M on the very
    same file f into a M_f model, in a self-supervised manner.

    Finally, the fine-tuned model M_f is applied to file f to obtain the final (and
    hopefully better) speech/non-speech labels.

    During fine-tuning, frames where the pretrained model M is very confident are weighted
    more than those with lower confidence: the intuition is that the model will use these
    high confidence regions to adapt to recording conditions (e.g. background noise) and
    hence will eventually be better on the parts of f where it was initially not quite
    confident.

    Conversely, to avoid overfitting too much to those high confidence regions, we use
    data augmentation and freeze all but the final few layers of the pretrained model M.

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation model.
        Defaults to "hbredin/VoiceActivityDetection-PyanNet-DIHARD".
    augmentation : BaseWaveformTransform, or dict, optional
        torch_audiomentations waveform transform, used during fine-tuning.
        Defaults to no augmentation.
    fscore : bool, optional
        Optimize (precision/recall) fscore.
        Defaults to optimizing detection error rate.

    Hyper-parameters
    ----------------
    num_epochs : int
        Number of epochs (where one epoch = going through the file once).
    batch_size : int
        Batch size.
    learning_rate : float
        Learning rate.

    See also
    --------
    pyannote.audio.pipelines.utils.get_inference
    """

    def __init__(
        self,
        segmentation: PipelineInference = "hbredin/VoiceActivityDetection-PyanNet-DIHARD",
        augmentation: PipelineAugmentation = None,
        fscore: bool = False,
    ):
        super().__init__()

        # pretrained segmentation model
        self.inference: Inference = get_inference(segmentation)
        self.augmentation: BaseWaveformTransform = get_augmentation(augmentation)

        self.fscore = fscore

        self.num_epochs = Integer(0, 10)
        self.batch_size = Categorical([1, 2, 4, 8, 16, 32])
        self.learning_rate = LogUniform(1e-6, 1)

    def apply(self, file: AudioFile) -> Annotation:

        # create a copy of file
        file = dict(file)

        # get segmentation scores from pretrained segmentation model
        file["seg"] = self.inference(file)

        # infer voice activity detection scores
        file["vad"] = np.max(file["seg"], axis=1, keepdims=True)

        # apply voice activity detection pipeline with default parameters
        vad_pipeline = VoiceActivityDetection("vad").instantiate(
            {
                "onset": 0.5,
                "offset": 0.5,
                "min_duration_on": 0.0,
                "min_duration_off": 0.0,
            }
        )
        file["annotation"] = vad_pipeline(file)

        # do not fine tune the model if num_epochs is zero
        if self.num_epochs == 0:
            return file["annotation"]

        # infer model confidence from segmentation scores
        # TODO: scale confidence differently (e.g. via an additional binarisation threshold hyper-parameter)
        file["confidence"] = np.min(
            np.abs((file["seg"] - 0.5) / 0.5), axis=1, keepdims=True
        )

        # create a dummy train-only protocol where `file` is the only training file
        class DummyProtocol(SpeakerDiarizationProtocol):
            name = "DummyProtocol"

            def train_iter(self):
                yield file

        vad_task = VoiceActivityDetectionTask(
            DummyProtocol(),
            duration=self.inference.duration,
            weight="confidence",
            batch_size=self.batch_size,
            augmentation=self.augmentation,
        )

        vad_model = deepcopy(self.inference.model)
        vad_model.task = vad_task

        def configure_optimizers(model):
            return SGD(model.parameters(), lr=self.learning_rate)

        vad_model.configure_optimizers = MethodType(configure_optimizers, vad_model)

        with tempfile.TemporaryDirectory() as default_root_dir:
            trainer = Trainer(
                max_epochs=self.num_epochs,
                gpus=1,
                callbacks=[GraduallyUnfreeze(epochs_per_stage=self.num_epochs + 1)],
                checkpoint_callback=False,
                default_root_dir=default_root_dir,
            )
            trainer.fit(vad_model)

        inference = Inference(
            vad_model,
            device=self.inference.device,
            batch_size=self.inference.batch_size,
            progress_hook=self.inference.progress_hook,
        )
        file["vad"] = inference(file)

        return vad_pipeline(file)

    def get_metric(self) -> Union[DetectionErrorRate, DetectionPrecisionRecallFMeasure]:
        """Return new instance of detection metric"""

        if self.fscore:
            return DetectionPrecisionRecallFMeasure(collar=0.0, skip_overlap=False)

        return DetectionErrorRate(collar=0.0, skip_overlap=False)

    def get_direction(self):
        if self.fscore:
            return "maximize"
        return "minimize"
