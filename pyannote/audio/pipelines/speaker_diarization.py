# The MIT License (MIT)
#
# Copyright (c) 2021 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Speaker diarization pipelines"""

import warnings
from copy import deepcopy
from typing import Any, Callable, Mapping, Optional, Text, Union

import numpy as np
import torch
from einops import rearrange
from scipy.spatial.distance import cdist, squareform

from pyannote.audio import Audio, Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import PipelineModel, get_devices, get_model
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.signal import Binarize, binarize
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature
from pyannote.core.utils.distance import pdist
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Uniform

from .clustering import Clustering
from .speaker_verification import PretrainedSpeakerEmbedding


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
    clustering : {"AffinityPropagation", "DBSCAN", "OPTICS", "AgglomerativeClustering"}, optional
        Defaults to "AffinityPropagation".
    expects_num_speakers : bool, optional
        Defaults to False.

    Hyper-parameters
    ----------------

    Usage
    -----
    >>> pipeline = SpeakerDiarization()
    >>> diarization = pipeline("/path/to/audio.wav")
    >>> diarization = pipeline("/path/to/audio.wav", num_speakers=2)

    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        embedding: PipelineModel = "pyannote/embedding",
        clustering: Text = "AffinityPropagation",
        expects_num_speakers: bool = False,
    ):

        super().__init__()

        self.segmentation = segmentation
        self.embedding = embedding
        self.expects_num_speakers = expects_num_speakers

        seg_device, emb_device = get_devices(needs=2)

        self.seg_model_: Model = get_model(segmentation)
        self.seg_model_.to(seg_device)
        self._segmentation_inference = Inference(self.seg_model_, skip_aggregation=True)

        self.emb_model_ = PretrainedSpeakerEmbedding(self.embedding, device=emb_device)
        self.emb_audio_ = Audio(sample_rate=self.emb_model_.sample_rate, mono=True)

        try:
            Klustering = Clustering[clustering]
        except KeyError:
            raise ValueError(
                f'clustering must be one of [{", ".join(list(Clustering.__members__))}]'
            )
        self.clustering = Klustering.value(
            expects_num_clusters=self.expects_num_speakers
        )

        self.warm_up = Uniform(0.0, 0.2)

        # hyper-parameters

        # onset/offset hysteresis thresholding
        self.onset = Uniform(0.05, 0.95)
        self.offset = Uniform(0.05, 0.95)

        # minimum amount of speech needed to use speaker in clustering
        self.min_activity = Uniform(0.0, self._segmentation_inference.duration)

        self.affinity_threshold_percentile = Uniform(0.0, 1.0)

        # weights of constraints in final (constrained) affinity matrix.
        # Between 0 and 1, where alpha = 0.0 means no constraint.
        self.constraint_propagate = Uniform(0.0, 1.0)

    def initialize(self):
        """Initialize pipeline with current set of parameters"""

        self._binarize = Binarize(
            onset=0.5,
            offset=0.5,
        )

    def trim_warmup(self, segmentations: SlidingWindowFeature) -> SlidingWindowFeature:
        """Trim left and right warm-up regions

        Parameters
        ----------
        segmentations : SlidingWindowFeature
            (num_chunks, num_frames, num_speakers)-shaped speaker activation scores

        Returns
        -------
        trimmed : SlidingWindowFeature
            (num_chunks, trimmed_num_frames, num_speakers)-shaped speaker activation scores
        """

        _, num_frames, _ = segmentations.data.shape
        new_data = segmentations.data[
            :,
            round(num_frames * self.warm_up) : round(num_frames * (1.0 - self.warm_up)),
        ]

        chunks = segmentations.sliding_window
        new_chunks = SlidingWindow(
            start=chunks.start + self.warm_up * chunks.duration,
            step=chunks.step,
            duration=(1.0 - 2 * self.warm_up) * chunks.duration,
        )

        return SlidingWindowFeature(new_data, new_chunks)

    def compute_constraints(
        self, binarized_segmentations: SlidingWindowFeature
    ) -> np.ndarray:
        """

        Parameters
        ----------
        binarized_segmentations : SlidingWindowFeature
            (num_chunks, num_frames, local_num_speakers)-shaped segmentation.

        Returns
        -------
        constraints : np.ndarray
            (num_chunks x local_num_speakers, num_chunks x local_num_speakers)-shaped constraint matrix

        """

        num_chunks, num_frames, local_num_speakers = binarized_segmentations.data.shape

        # 1. intra-chunk "cannot link" constraints (upper triangle only)
        chunk_idx = np.broadcast_to(
            np.arange(num_chunks), (local_num_speakers, num_chunks)
        )
        constraint = np.triu(
            squareform(
                -1.0 * pdist(rearrange(chunk_idx, "s c -> (c s)"), metric="equal")
            )
        )
        # (num_chunks x local_num_speakers, num_chunks x local_num_speakers)

        # 2. inter-chunk "must link" constraints
        # two speakers from two overlapping chunks are marked as "must-link"
        # if and only if the optimal permutation maps them and they are
        # both active in their common temporal support.
        chunks = binarized_segmentations.sliding_window

        # number of overlapping chunk
        num_overlapping_chunks = round(chunks.duration // chunks.step - 1.0)

        # loop on pairs of overlapping chunks
        # np.fill_diagonal(constraint, 1.0)
        for C, (_, binarized_segmentation) in enumerate(binarized_segmentations):
            for c in range(max(0, C - num_overlapping_chunks), C):

                # extract common temporal support
                shift = round((C - c) * num_frames * chunks.step / chunks.duration)
                this_segmentation = binarized_segmentation[: num_frames - shift]
                past_segmentation = binarized_segmentations[c, shift:]

                # find the optimal one-to-one mapping
                _, (permutation,) = permutate(
                    this_segmentation[np.newaxis], past_segmentation
                )

                # check whether speakers are active on the common temporal support
                # otherwise there is no point trying to match them
                this_active = np.any(this_segmentation, axis=0)
                past_active = np.any(past_segmentation, axis=0)

                for this, past in enumerate(permutation):
                    if this_active[this] and past_active[past]:
                        constraint[
                            c * local_num_speakers + past,
                            C * local_num_speakers + this,
                        ] = 1.0
                        # TODO: investigate weighting this by (num_frames - shift) / num_frames
                        # TODO: i.e. by the duration of the common temporal support

        # propagate cannot link constraints by "transitivity": if c_ij = -1 and c_jk = 1 then c_ik = -1
        # (only when this new constraint is not conflicting with existing constraint, i.e. when c_ik = 1)

        # loop on (i, j) pairs such that c_ij is either 1 or -1
        for i, j in zip(*np.where(constraint != 0)):

            # find all k for which c_ij = - c_jk and mark c_ik as cannot-link
            # unless it has been marked as must-link (c_ik = 1) before
            constraint[
                i, (constraint[i] != 1.0) & (constraint[j] + constraint[i, j] == 0.0)
            ] = -1.0

        # make constraint matrix symmetric
        constraint = squareform(squareform(constraint, checks=False))
        np.fill_diagonal(constraint, 1.0)

        return constraint

    def propagate_constraints(
        self, affinity: np.ndarray, constraint: np.ndarray
    ) -> np.ndarray:
        """Update affinity matrix by constraint propagation

        Stolen from
        https://github.com/wq2012/SpectralCluster/blob/34d155654dbbfcda61b808a4f61afa666476b3d2/spectralcluster/constraint.py

        Parameters
        ----------
        affinity : np.ndarray
            (N, N) affinity matrix with values in [0, 1].
            * affinity[i, j] = 1 indicates that i and j are very similar
            * affinity[i, j] = 0 indicates that i and j are very dissimilar
        constraint : np.ndarray
            (N, N) constraint matrix with values in [-1, 1].
            * constraint[i, j] > 0 indicates a must-link constraint
            * constraint[i, j] < 0 indicates a cannot-link constraint
            * constraint[i, j] = 0 indicates absence of constraint

        Returns
        -------
        constrained_affinity : np.ndarray
            Constrained affinity matrix.
        propagated_constraint : np.ndarray
            Propagated constraint matrix.

        Reference
        ---------
        Lu, Zhiwu, and IP, Horace HS.
        "Constrained spectral clustering via exhaustive and efficient constraint propagation."
        ECCV 2010
        """

        degree = np.diag(np.sum(affinity, axis=1))
        degree_norm = np.diag(1 / (np.sqrt(np.diag(degree)) + 1e-10))

        # Compute affinity_norm as D^(-1/2)AD^(-1/2)
        affinity_norm = degree_norm.dot(affinity).dot(degree_norm)

        # The closed form of the final converged constraint matrix is:
        # (1-alpha)^2 * (I-alpha*affinity_norm)^(-1) * constraint *
        # (I-alpha*affinity_norm)^(-1). We save (I-alpha*affinity_norm)^(-1) as a
        # `temp_value` for readibility
        temp_value = np.linalg.inv(
            np.eye(affinity.shape[0]) - (1 - self.constraint_propagate) * affinity_norm
        )
        propagated_constraint = self.constraint_propagate ** 2 * temp_value.dot(
            constraint
        ).dot(temp_value)

        # `is_positive` is a mask matrix where values of the propagated_constraint
        # are positive. The affinity matrix is adjusted by the final constraint
        # matrix using equation (4) in reference paper
        is_positive = propagated_constraint > 0
        affinity1 = 1 - (1 - propagated_constraint * is_positive) * (
            1 - affinity * is_positive
        )
        affinity2 = (1 + propagated_constraint * np.invert(is_positive)) * (
            affinity * np.invert(is_positive)
        )
        return affinity1 + affinity2, propagated_constraint

    CACHED_SEGMENTATION = "@diarization/segmentation/raw"

    def debug(self, file: Mapping, key: Text, value: Any):
        file[f"@diarization/{key}"] = deepcopy(value)

    def apply(
        self,
        file: AudioFile,
        num_speakers: int = None,
        debug: Optional[Union[bool, Callable]] = False,
    ) -> Annotation:
        """Apply speaker diarization

        Parameters
        ----------
        file : AudioFile
            Processed file.
        num_speakers : int, optional
            Expected number of speakers.
        debug : bool or callable, optional
            Set to True to add debugging keys into `file`.

        Returns
        -------
        diarization : Annotation
            Speaker diarization
        """

        if not callable(debug):
            debug = self.debug if debug else lambda *args: None

        # __ HANDLE EXPECTED NUMBER OF SPEAKERS ________________________________________
        if self.expects_num_speakers and num_speakers is None:

            if "annotation" in file:
                num_speakers = len(file["annotation"].labels())

                if not self.training:
                    warnings.warn(
                        f"This pipeline expects the number of speakers (num_speakers) to be given. "
                        f"It has been automatically set to {num_speakers:d} based on reference annotation. "
                    )

            else:
                raise ValueError(
                    "This pipeline expects the number of speakers (num_speakers) to be given."
                )

        # __ SPEAKER SEGMENTATION ______________________________________________________

        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, local_num_speakers)
        if (not self.training) or (
            self.training and self.CACHED_SEGMENTATION not in file
        ):
            file[self.CACHED_SEGMENTATION] = self._segmentation_inference(file)

        segmentations: SlidingWindowFeature = file[self.CACHED_SEGMENTATION]
        frames: SlidingWindow = self._segmentation_inference.model.introspection.frames
        duration: float = self._segmentation_inference.duration

        # apply hysteresis thresholding on each chunk
        binarized_segmentations: SlidingWindowFeature = binarize(
            segmentations, onset=self.onset, offset=self.offset, initial_state=False
        )

        # trim warm-up regions
        segmentations = self.trim_warmup(segmentations)
        binarized_segmentations = self.trim_warmup(binarized_segmentations)

        # mask overlapping speech regions
        masked_segmentations = SlidingWindowFeature(
            binarized_segmentations.data
            * (np.sum(binarized_segmentations.data, axis=-1, keepdims=True) == 1.0),
            binarized_segmentations.sliding_window,
        )

        # estimate frame-level number of instantaneous speakers
        speaker_count = Inference.aggregate(
            np.sum(binarized_segmentations, axis=-1, keepdims=True),
            frames,
        )
        speaker_count.data = np.round(speaker_count)

        # shape
        num_chunks, num_frames, local_num_speakers = segmentations.data.shape

        # __ SPEAKER STATUS ____________________________________________________________

        SKIP = 0  # SKIP this speaker because it is never active in current chunk
        KEEP = 1  # KEEP this speaker because it is active at least once within current chunk
        LONG = 2  # this speaker speaks LONG enough within current chunk to be used in clustering

        speaker_status = np.full((num_chunks, local_num_speakers), SKIP, dtype=np.int)
        speaker_status[np.any(binarized_segmentations.data, axis=1)] = KEEP
        speaker_status[
            np.mean(masked_segmentations, axis=1) * duration * (1.0 - self.warm_up)
            > self.min_activity
        ] = LONG

        if np.sum(speaker_status == LONG) == 0:
            warnings.warn("Please decrease 'min_activity' threshold.")
            return Annotation(uri=file["uri"])

        # TODO: handle corner case where there is 0 or 1 LONG speaker

        debug(file, "segmentation/binarized", binarized_segmentations)
        debug(file, "segmentation/speaker_count", speaker_count)

        # __ SPEAKER EMBEDDING _________________________________________________________

        embeddings = []

        for c, ((chunk, masked_segmentation), status) in enumerate(
            zip(masked_segmentations, speaker_status)
        ):

            if np.all(status == SKIP):
                chunk_embeddings = np.zeros(
                    (local_num_speakers, self.emb_model_.dimension), dtype=np.float32
                )

            else:

                waveforms: torch.Tensor = (
                    self.emb_audio_.crop(file, chunk)[0]
                    .unsqueeze(0)
                    .expand(local_num_speakers, -1, -1)
                    .to(self.emb_model_.device)
                )
                masks = torch.tensor(masked_segmentation.T, dtype=torch.float32)
                chunk_embeddings: np.ndarray = self.emb_model_(waveforms, masks=masks)
                # (local_num_speakers, dimension)

            embeddings.append(chunk_embeddings)

        # stack and unit-normalized embeddings
        embeddings = np.stack(embeddings)
        embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
        debug(file, "clustering/embeddings", embeddings)

        # skip speakers for which embedding extraction failed for some reason
        speaker_status[np.any(np.isnan(embeddings), axis=-1)] = SKIP

        if "annotation" in file:

            reference = file["annotation"].discretize(
                support=Segment(0.0, Audio().get_duration(file)),
                resolution=frames,
            )
            permutations = []

            for (
                c,
                (chunk, segmentation),
            ) in enumerate(segmentations):

                if np.all(speaker_status[c] != LONG):
                    continue

                segmentation = segmentation[np.newaxis, :, speaker_status[c] == LONG]

                local_reference = reference.crop(chunk)
                _, (permutation,) = permutate(
                    segmentation,
                    local_reference[:num_frames],
                )
                active_reference = np.any(local_reference > 0, axis=0)
                permutations.extend(
                    [
                        i if ((i is not None) and (active_reference[i])) else None
                        for i in permutation
                    ]
                )

            debug(file, "clustering/clusters/oracle", permutations)

        # __ RAW AFFINITY ______________________________________________________________

        affinity = squareform(
            1 - 0.5 * pdist(embeddings[speaker_status == LONG], metric="cosine")
        )
        np.fill_diagonal(affinity, 1.0)
        debug(file, "clustering/affinity/raw", affinity)

        # __ CONSTRAINED AFFINITY ______________________________________________________

        if self.constraint_propagate > 0:

            # compute (soft) {must/cannot}-link constraints based on local segmentation
            constraints = self.compute_constraints(binarized_segmentations)

            long = rearrange(
                speaker_status == LONG,
                "c s -> (c s)",
                c=num_chunks,
                s=local_num_speakers,
            )

            constraints = constraints[long][:, long]
            debug(file, "clustering/constraints", constraints)

            affinity, _ = self.propagate_constraints(affinity, constraints)
            debug(file, "clustering/affinity/constrained", affinity)

        # __ REFINED AFFINITY ___________________________________________________________

        affinity = affinity * (
            affinity
            >= np.percentile(affinity, 100 * self.affinity_threshold_percentile, axis=0)
        )
        affinity = 0.5 * (affinity + affinity.T)
        debug(file, "clustering/affinity/refined", affinity)

        # __ ACTIVE SPEAKER CLUSTERING _________________________________________________
        # clusters[chunk_id x local_num_speakers + speaker_id] = k
        # * k=SpeakerStatus.Inactive                if speaker is inactive
        # * k=-1                if speaker is active but not assigned to any cluster
        # * k in {0, ... K - 1} if speaker is active and is assigned to cluster k

        clusters = np.full((num_chunks, local_num_speakers), -1, dtype=np.int)
        clusters[speaker_status == SKIP] = -2

        if num_speakers == 1 or np.sum(speaker_status == LONG) < 2:
            clusters[speaker_status == LONG] = 0
            num_clusters = 1

        else:
            clusters[speaker_status == LONG] = self.clustering(
                affinity, num_clusters=num_speakers
            )
            num_clusters = np.max(clusters) + 1

            # corner case where clustering fails to converge and returns only -1 labels
            if num_clusters == 0:
                clusters[speaker_status == LONG] = 0
                num_clusters = 1

        debug(file, "clustering/clusters/raw", clusters)

        # __ FINAL SPEAKER ASSIGNMENT ___________________________________________________

        centroids = np.vstack(
            [np.mean(embeddings[clusters == k], axis=0) for k in range(num_clusters)]
        )
        unassigned = (speaker_status == KEEP) | (clusters == -1)
        distances = cdist(
            embeddings[unassigned],
            centroids,
            metric="cosine",
        )
        clusters[unassigned] = np.argmin(distances, axis=1)

        debug(file, "clustering/clusters/centroids", centroids)
        debug(file, "clustering/clusters/assigned", clusters)

        # __ CLUSTERING-BASED SEGMENTATION AGGREGATION _________________________________
        # build final aggregated speaker activations

        clustered_segmentations = np.NAN * np.zeros(
            (num_chunks, num_frames, num_clusters)
        )
        for c, (cluster, (chunk, segmentation)) in enumerate(
            zip(clusters, segmentations)
        ):

            # cluster is (local_num_speakers, )-shaped
            # segmentation is (num_frames, local_num_speakers)-shaped
            for k in np.unique(cluster):
                if k == -2:
                    continue

                clustered_segmentations[c, :, k] = np.max(
                    segmentation[:, cluster == k], axis=1
                )

        clustered_segmentations = SlidingWindowFeature(
            clustered_segmentations, segmentations.sliding_window
        )
        speaker_activations = Inference.aggregate(clustered_segmentations, frames)

        debug(file, "segmentation/clustered", clustered_segmentations)
        debug(file, "segmentation/aggregated", speaker_activations)

        # __ FINAL BINARIZATION ________________________________________________________
        sorted_speakers = np.argsort(-speaker_activations, axis=-1)
        final_binarized = np.zeros_like(speaker_activations.data)
        for t, ((_, count), speakers) in enumerate(zip(speaker_count, sorted_speakers)):
            # TODO: find a way to stop clustering early enough to avoid num_clusters < count
            count = min(num_clusters, int(count.item()))
            for i in range(count):
                final_binarized[t, speakers[i]] = 1.0

        diarization = self._binarize(SlidingWindowFeature(final_binarized, frames))
        diarization.uri = file["uri"]

        return diarization

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)
