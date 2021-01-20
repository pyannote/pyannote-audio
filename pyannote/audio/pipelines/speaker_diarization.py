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

from typing import Mapping, Optional, Text, Union

from pyannote.audio import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.core import Annotation
from pyannote.database import get_annotated
from pyannote.metrics.diarization import (
    DiarizationPurityCoverageFMeasure,
    GreedyDiarizationErrorRate,
)
from pyannote.pipeline.parameter import Uniform

from .segmentation import Segmentation
from .speech_turn_assignment import SpeechTurnClosestAssignment
from .speech_turn_clustering import SpeechTurnClustering


class SpeakerDiarization(Pipeline):
    """Speaker diarization pipeline

    Parameters
    ----------
    segmentation : Inference or str, optional
        `Inference` instance used to extract raw segmentation scores.
        When `str`, assumes that file already contains a corresponding key with
        precomputed scores. Defaults to "seg".
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
        segmentation: Union[Inference, Text] = "seg",
        embeddings: Union[Inference, Text] = "emb",
        metric: Optional[str] = "cosine",
        purity: Optional[float] = None,
        coverage: Optional[float] = None,
        fscore: bool = False,
    ):

        super().__init__()

        # temporary hack -- need to bring back old Wrapper/Wrappable logic
        if isinstance(segmentation, Mapping):
            segmentation = Inference(**segmentation)
        if isinstance(embeddings, Mapping):
            embeddings = Inference(**embeddings)

        self.segmentation = Segmentation(scores=segmentation)

        self.embeddings = embeddings
        self.metric = metric

        self.clustering = SpeechTurnClustering(
            embeddings=self.embeddings, metric=self.metric
        )

        self.assignment = SpeechTurnClosestAssignment(
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

    def apply(self, file: AudioFile) -> Annotation:
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

        # segmentation where speech turns are already clustered locally (with letters A/B/C/...)
        segmentation = self.segmentation(file).rename_labels(generator="string")

        # corner case where there is just one cluster already
        if len(segmentation.labels()) < 2:
            return segmentation

        # split segmentation into two parts:
        # - clean (i.e. non-overlapping) and long-enough clusters
        # - the rest of it

        clean = segmentation.copy()
        rest = segmentation.empty()

        for (segment, track), (other_segment, other_track) in segmentation.co_iter(
            segmentation
        ):
            if segment == other_segment and track == other_track:
                continue

            rest[segment, track] = segmentation[segment, track]
            try:
                del clean[segment, track]
            except KeyError:
                pass

            rest[other_segment, other_track] = segmentation[other_segment, other_track]
            try:
                del clean[other_segment, other_track]
            except KeyError:
                pass

        # TODO: consider removing all short segments instead of short clusters

        long_enough_local_clusters = [
            local_cluster
            for local_cluster, duration in clean.chart()
            if duration > self.cluster_min_duration
        ]
        # corner case where there is no clean, long enough local clusters
        if not long_enough_local_clusters:
            return segmentation

        clean_long_enough = clean.subset(long_enough_local_clusters)

        rest.update(clean.subset(long_enough_local_clusters, invert=True), copy=False)

        # at this point, we have split "segmentation" into two parts:
        # - "clean_long_enough" uses A/B/C labels
        # - "rest" uses A/B/C labels as well

        # we apply clustering on "clean_long_enough" and use A/B/C labels
        clustered_clean_long_enough = self.clustering(
            file, clean_long_enough
        ).rename_labels(generator="string")

        # this will contain the final result using a combination of
        # - A/B/C labels coming from above "clean_long_enough" clusters)
        # - 1/2/3 labels coming from un-assigned "rest" segments
        global_hypothesis = clustered_clean_long_enough.copy()

        rest_copy = rest.copy()
        for local_cluster in rest_copy.labels():
            try:
                # for each local cluster remaining in "rest", we find which global cluster it has been assigned to
                segment, track = next(
                    clean_long_enough.subset([local_cluster]).itertracks()
                )
                global_cluster = clustered_clean_long_enough[segment, track]

                # we move its left-aside segments back into the corresponding global cluster
                # we also remove those segments from "rest" to remember they are now dealt with.
                for segment, track in rest_copy.subset([local_cluster]).itertracks():
                    global_hypothesis[segment, track] = global_cluster
                    del rest[segment, track]

            except StopIteration:
                # this happens if the original local cluster was not present at all in clean_long_enough
                # it will be passed over to the upcoming "assignment" step (see below).
                continue

        if len(rest) > 0:
            rest.rename_labels(generator="int", copy=False)
            assigned_rest = self.assignment(file, rest, global_hypothesis)

            # assigned_rest uses a combination of
            # - A/B/C labels for speech turns assigned to global_hypothesis clusters
            # - 1/2/3 labels for those that could not be assigned because they were too dissimlar

            global_hypothesis.update(assigned_rest, copy=False)

        return global_hypothesis

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

        reference: Annotation = file["annotation"]
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
