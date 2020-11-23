# The MIT License (MIT)
#
# Copyright (c) 2017-2020 CNRS
#
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

from typing import Optional, Text, Union

from pyannote.audio.core.inference import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.core import Annotation
from pyannote.database import get_annotated
from pyannote.metrics.diarization import (
    DiarizationPurityCoverageFMeasure,
    GreedyDiarizationErrorRate,
)
from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Uniform

from .speech_turn_assignment import SpeechTurnClosestAssignment
from .speech_turn_clustering import SpeechTurnClustering
from .speech_turn_segmentation import SpeechTurnSegmentation


class SpeakerDiarization(Pipeline):
    """Speaker diarization pipeline

    Parameters
    ----------
    vad_scores : Inference or str, optional
        `Inference` instance used to extract raw voice activity detection scores.
        When `str`, assumes that file already contains a corresponding key with
        precomputed scores. Defaults to "vad".
    scd_scores : Inference or str, optional
        `Inference` instance used to extract raw speaker change detection scores.
        When `str`, assumes that file already contains a corresponding key with
        precomputed scores. Defaults to "scd".
    embeddings : Inference or str, optional
        `Inference` instance used to extract speaker embeddings. When `str`,
        assumes that file already contains a corresponding key with precomputed
        embeddings. Defaults to "emb".
    metric : {'euclidean', 'cosine', 'angular'}, optional
        Metric used for comparing embeddings. Defaults to 'cosine'.
    purity : float, optional
        Optimize coverage for target purity.
        Defaults to optimizing diarization error rate.
    coverage : float, optional
        Optimize purity for target coverage.
        Defaults to optimizing diarization error rate.
    fscore : bool, optional
        Optimize for purity/coverage fscore.
        Defaults to optimizing for diarization error rate.

    # TODO: investigate the use of non-weighted fscore/purity/coverage

    Hyper-parameters
    ----------------
    cluster_min_duration : float
        Do not cluster speech turns shorter than `cluster_min_duration`.
        Assign them to the closest cluster (of long speech turns) instead.
    """

    def __init__(
        self,
        vad_scores: Union[Inference, Text] = "vad",
        scd_scores: Union[Inference, Text] = "scd",
        embeddings: Union[Inference, Text] = "emb",
        metric: Optional[str] = "cosine",
        purity: Optional[float] = None,
        coverage: Optional[float] = None,
        fscore: bool = False,
    ):

        super().__init__()

        self.vad_scores = vad_scores
        self.scd_scores = scd_scores
        self.speech_turn_segmentation = SpeechTurnSegmentation(
            vad_scores=vad_scores, scd_scores=scd_scores
        )

        self.embeddings = embeddings
        self.metric = metric
        self.speech_turn_clustering = SpeechTurnClustering(
            embeddings=self.embeddings, metric=self.metric
        )

        self.speech_turn_assignment = SpeechTurnClosestAssignment(
            embeddings=self.embeddings, metric=self.metric
        )

        # hyper-parameter
        self.cluster_min_duration = Uniform(0, 10)

        if sum((purity is not None, coverage is not None, fscore)):
            raise ValueError(
                "One must choose between optimizing for f-score, target purity, or target coverage."
            )

        self.purity = purity
        self.coverage = coverage
        self.fscore = fscore

    def __call__(self, file: AudioFile) -> Annotation:
        """Apply speaker diarization

        Parameters
        ----------
        file : AudioFile
            Processed file.

        Returns
        -------
        diarization : Annotation
            Speaker diarization
        """

        # segmentation into speech turns
        speech_turns = self.speech_turn_segmentation(file)

        # in case there is one speech turn or less, there is no need to apply
        # any kind of clustering approach.
        if len(speech_turns) < 2:
            return speech_turns

        # split short/long speech turns. the idea is to first cluster long
        # speech turns (i.e. those for which we can trust embeddings) and then
        # assign each speech turn to the closest cluster.
        long_speech_turns = speech_turns.empty()
        shrt_speech_turns = speech_turns.empty()
        for segment, track, label in speech_turns.itertracks(yield_label=True):
            if segment.duration < self.cluster_min_duration:
                shrt_speech_turns[segment, track] = label
            else:
                long_speech_turns[segment, track] = label

        # in case there are no long speech turn to cluster, we return the
        # original speech turns (= shrt_speech_turns)
        if len(long_speech_turns) < 1:
            return speech_turns

        # first: cluster long speech turns
        long_speech_turns = self.speech_turn_clustering(file, long_speech_turns)

        # then: assign short speech turns to clusters
        long_speech_turns.rename_labels(generator="string", copy=False)

        if len(shrt_speech_turns) > 0:
            shrt_speech_turns.rename_labels(generator="int", copy=False)
            shrt_speech_turns = self.speech_turn_assignment(
                file, shrt_speech_turns, long_speech_turns
            )
        # merge short/long speech turns
        return long_speech_turns.update(shrt_speech_turns, copy=False).support(
            collar=0.0
        )

        # TODO. add overlap detection
        # TODO. add overlap-aware resegmentation

    def loss(self, file: AudioFile, hypothesis: Annotation) -> float:
        """Compute coverage at target purity (or vice versa)

        Parameters
        ----------
        file : `dict`
            File as provided by a pyannote.database protocol.
        hypothesis : `pyannote.core.Annotation`
            Speech turns.

        Returns
        -------
        coverage (or purity) : float
            When optimizing for target purity:
                If purity < target_purity, returns (purity - target_purity).
                If purity > target_purity, returns coverage.
            When optimizing for target coverage:
                If coverage < target_coverage, returns (coverage - target_coverage).
                If coverage > target_coverage, returns purity.
        """

        fmeasure = DiarizationPurityCoverageFMeasure()

        reference = file["annotation"]
        _ = fmeasure(reference, hypothesis, uem=get_annotated(file))
        purity, coverage, _ = fmeasure.compute_metrics()

        if self.purity is not None:
            if purity > self.purity:
                return purity - self.purity
            else:
                return coverage

        elif self.coverage is not None:
            if coverage > self.coverage:
                return coverage - self.coverage
            else:
                return purity

    def get_metric(
        self,
    ) -> Union[GreedyDiarizationErrorRate, DiarizationPurityCoverageFMeasure]:
        """Return new instance of diarization metric"""

        if (self.purity is not None) or (self.coverage is not None):
            raise NotImplementedError(
                "pyannote.pipeline will use `loss` method fallback."
            )

        if self.fscore:
            return DiarizationPurityCoverageFMeasure(collar=0.0, skip_overlap=False)

        # defaults to optimizing diarization error rate
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)

    def get_direction(self):
        """Optimization direction"""

        if self.purity is not None:
            # we maximize coverage at target purity
            return "maximize"
        elif self.coverage is not None:
            # we maximize purity at target coverage
            return "maximize"
        elif self.fscore:
            # we maximize purity/coverage f-score
            return "maximize"
        else:
            # we minimize diarization error rate
            return "minimize"
