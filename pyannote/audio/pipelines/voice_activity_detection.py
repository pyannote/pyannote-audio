# MIT License
#
# Copyright (c) 2018- CNRS
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
from functools import partial
from types import MethodType
from typing import Callable, Optional, Text, Union

import numpy as np
from pyannote.core import Annotation, SlidingWindowFeature
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.metrics.detection import (
    DetectionErrorRate,
    DetectionPrecisionRecallFMeasure,
)
from pyannote.pipeline.parameter import Categorical, Integer, LogUniform, Uniform
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
    get_inference,
    get_model,
)
from pyannote.audio.tasks import VoiceActivityDetection as VoiceActivityDetectionTask
from pyannote.audio.utils.signal import Binarize


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
        Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    fscore : bool, optional
        Optimize (precision/recall) fscore. Defaults to optimizing detection
        error rate.
    use_auth_token : str, optional
        When loading private huggingface.co models, set `use_auth_token`
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running `huggingface-cli login`
    inference_kwargs : dict, optional
        Keywords arguments passed to Inference.

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
        segmentation: PipelineModel = "pyannote/segmentation",
        fscore: bool = False,
        use_auth_token: Union[Text, None] = None,
        **inference_kwargs,
    ):
        super().__init__()

        self.segmentation = segmentation
        self.fscore = fscore

        # load model and send it to GPU (when available and not already on GPU)
        model = get_model(segmentation, use_auth_token=use_auth_token)

        inference_kwargs["pre_aggregation_hook"] = lambda scores: np.max(
            scores, axis=-1, keepdims=True
        )
        self._segmentation = Inference(model, **inference_kwargs)

        if model.specifications.powerset:
            self.onset = self.offset = 0.5
        else:
            #  hyper-parameters used for hysteresis thresholding
            self.onset = Uniform(0.0, 1.0)
            self.offset = Uniform(0.0, 1.0)

        # hyper-parameters used for post-processing i.e. removing short speech regions
        # or filling short gaps between speech regions
        self.min_duration_on = Uniform(0.0, 1.0)
        self.min_duration_off = Uniform(0.0, 1.0)

    def default_parameters(self):
        if self.segmentation == "pyannote/segmentation":
            # parameters optimized for DIHARD 3 development set
            return {
                "onset": 0.767,
                "offset": 0.377,
                "min_duration_on": 0.136,
                "min_duration_off": 0.067,
            }

        elif self.segmentation == "pyannote/segmentation-3.0.0":
            return {
                "min_duration_on": 0.0,
                "min_duration_off": 0.0,
            }

        raise NotImplementedError()

    def classes(self):
        return ["SPEECH"]

    def initialize(self):
        """Initialize pipeline with current set of parameters"""

        self._binarize = Binarize(
            onset=self.onset,
            offset=self.offset,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )

    CACHED_SEGMENTATION = "cache/segmentation/inference"

    def apply(self, file: AudioFile, hook: Optional[Callable] = None) -> Annotation:
        """Apply voice activity detection

        Parameters
        ----------
        file : AudioFile
            Processed file.
        hook : callable, optional
            Callback called after each major steps of the pipeline as follows:
                hook(step_name,      # human-readable name of current step
                     step_artefact,  # artifact generated by current step
                     file=file)      # file being processed
            Time-consuming steps call `hook` multiple times with the same `step_name`
            and additional `completed` and `total` keyword arguments usable to track
            progress of current step.

        Returns
        -------
        speech : Annotation
            Speech regions.
        """

        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)

        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, 1)
        if self.training:
            if self.CACHED_SEGMENTATION in file:
                segmentations = file[self.CACHED_SEGMENTATION]
            else:
                segmentations = self._segmentation(
                    file, hook=partial(hook, "segmentation", None)
                )
                file[self.CACHED_SEGMENTATION] = segmentations
        else:
            segmentations: SlidingWindowFeature = self._segmentation(
                file, hook=partial(hook, "segmentation", None)
            )

        hook("segmentation", segmentations)

        speech: Annotation = self._binarize(segmentations)
        speech.uri = file["uri"]
        return speech.rename_labels({label: "SPEECH" for label in speech.labels()})

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
                accelerator="gpu",
                devices=1,
                callbacks=[GraduallyUnfreeze(epochs_per_stage=self.num_epochs + 1)],
                enable_checkpointing=False,
                default_root_dir=default_root_dir,
            )
            trainer.fit(vad_model)

        inference = Inference(
            vad_model,
            device=self.inference.device,
            batch_size=self.inference.batch_size,
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
