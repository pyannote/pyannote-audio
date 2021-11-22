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
from functools import partial
from typing import Callable, Optional, Text

import numpy as np
import torch
from scipy.spatial.distance import cdist, squareform

from pyannote.audio import Audio, Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import PipelineModel, get_devices, get_model
from pyannote.audio.utils.permutation import mae_cost_func, permutate
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
    clustering : {"AgglomerativeClustering", "SpectralClustering"}, optional
        Defaults to "AgglomerativeClustering".
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
        clustering: Text = "AgglomerativeClustering",
        expects_num_speakers: bool = False,
    ):

        super().__init__()

        self.segmentation = segmentation
        self.embedding = embedding
        self.expects_num_speakers = expects_num_speakers

        seg_device, emb_device = get_devices(needs=2)

        model: Model = get_model(segmentation)
        model.to(seg_device)
        self._segmentation = Inference(model, skip_aggregation=True)
        self._frames: SlidingWindow = self._segmentation.model.introspection.frames

        self._embedding = PretrainedSpeakerEmbedding(self.embedding, device=emb_device)
        self._audio = Audio(sample_rate=self._embedding.sample_rate, mono=True)

        try:
            Klustering = Clustering[clustering]
        except KeyError:
            raise ValueError(
                f'clustering must be one of [{", ".join(list(Clustering.__members__))}]'
            )
        self.clustering = Klustering.value(
            metric=self._embedding.metric,
            expects_num_clusters=self.expects_num_speakers,
        )

        # hyper-parameters

        # stitching
        self.warm_up = 0.05
        self.stitch_threshold = Uniform(0.0, 1.0)

        # onset/offset hysteresis thresholding
        self.onset = Uniform(0.01, 0.99)
        self.offset = Uniform(0.01, 0.99)

        # minimum amount of speech needed to use speaker in clustering
        self.min_activity = Uniform(0.0, 10.0)

    CACHED_SEGMENTATION = "@diarization/segmentation/raw"

    @staticmethod
    def hook_default(*args, **kwargs):
        return

    def stitch_match_func(
        self, this: np.ndarray, that: np.ndarray, cost: float
    ) -> bool:
        return (
            np.any(this > self.onset)
            and np.any(that > self.onset)
            and cost < self.stitch_threshold
        )

    def apply(
        self,
        file: AudioFile,
        num_speakers: int = None,
        hook: Optional[Callable] = None,
    ) -> Annotation:
        """Apply speaker diarization

        Parameters
        ----------
        file : AudioFile
            Processed file.
        num_speakers : int, optional
            Expected number of speakers.
        hook : callable, optional
            Hook called after each major step of the pipeline with the following
            signature: hook("step_name", step_artefact, file=file)

        Returns
        -------
        diarization : Annotation
            Speaker diarization
        """

        if hook is None:
            hook = self.hook_default
            hook.missing = True
        else:
            hook = partial(hook, file=file)
            hook.missing = False

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
            file[self.CACHED_SEGMENTATION] = self._segmentation(file)

        segmentations: SlidingWindowFeature = file[self.CACHED_SEGMENTATION]
        hook("@segmentation/raw", segmentations)

        # trim warm-up regions
        segmentations = Inference.trim(
            segmentations, warm_up=(self.warm_up, self.warm_up)
        )
        hook("@segmentation/trim", segmentations)

        # stitch segmentations
        segmentations = Inference.stitch(
            segmentations,
            frames=self._frames,
            lookahead=None,  # TODO: make it an hyper-parameter?,
            cost_func=mae_cost_func,
            match_func=self.stitch_match_func,
        )
        hook("@segmentation/stitch", segmentations)

        chunk_duration: float = segmentations.sliding_window.duration

        # apply hysteresis thresholding on each chunk
        binarized_segmentations: SlidingWindowFeature = binarize(
            segmentations, onset=self.onset, offset=self.offset, initial_state=False
        )

        hook("@segmentation/binary", binarized_segmentations)

        # mask overlapping speech regions
        masked_segmentations = SlidingWindowFeature(
            binarized_segmentations.data
            * (np.sum(binarized_segmentations.data, axis=-1, keepdims=True) == 1.0),
            binarized_segmentations.sliding_window,
        )

        hook("@segmentation/mask", masked_segmentations)

        # estimate frame-level number of instantaneous speakers
        speaker_count = Inference.aggregate(
            np.sum(binarized_segmentations, axis=-1, keepdims=True),
            frames=self._frames,
            hamming=True,
            missing=0.0,
        )
        speaker_count.data = np.round(speaker_count)

        hook("@segmentation/count", speaker_count)

        # shape
        num_chunks, num_frames, local_num_speakers = segmentations.data.shape

        # __ SPEAKER STATUS ____________________________________________________________

        SKIP = 0  # SKIP this speaker because it is never active in current chunk
        KEEP = 1  # KEEP this speaker because it is active at least once within current chunk
        LONG = 2  # this speaker speaks LONG enough within current chunk to be used in clustering

        speaker_status = np.full((num_chunks, local_num_speakers), SKIP, dtype=np.int)
        speaker_status[np.any(binarized_segmentations.data, axis=1)] = KEEP
        speaker_status[
            np.mean(masked_segmentations, axis=1) * chunk_duration > self.min_activity
        ] = LONG

        if np.sum(speaker_status == LONG) == 0:
            warnings.warn("Please decrease 'min_activity' threshold.")

            return Annotation(uri=file["uri"])

        # TODO: handle corner case where there is 0 or 1 LONG speaker

        # __ SPEAKER EMBEDDING _________________________________________________________

        embeddings = []

        # TODO: batchify this loop
        for c, ((chunk, masked_segmentation), status) in enumerate(
            zip(masked_segmentations, speaker_status)
        ):

            if np.all(status == SKIP):
                chunk_embeddings = np.zeros(
                    (local_num_speakers, self._embedding.dimension), dtype=np.float32
                )

            else:
                waveforms: torch.Tensor = (
                    self._audio.crop(file, chunk, mode="pad")[0]
                    .unsqueeze(0)
                    .expand(local_num_speakers, -1, -1)
                )

                masks = torch.from_numpy(masked_segmentation).float().T
                chunk_embeddings: np.ndarray = self._embedding(waveforms, masks=masks)
                # (local_num_speakers, dimension)

            embeddings.append(chunk_embeddings)

        # stack and unit-normalized embeddings
        embeddings = np.stack(embeddings)
        with np.errstate(divide="ignore", invalid="ignore"):
            embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
        hook("@clustering/embedding", embeddings)

        # skip speakers for which embedding extraction failed for some reason
        speaker_status[np.any(np.isnan(embeddings), axis=-1)] = SKIP

        if not hook.missing and "annotation" in file:

            hook(
                "@clustering/distance",
                pdist(
                    embeddings[speaker_status == LONG], metric=self._embedding.metric
                ),
            )

            def oracle_cost_func(Y, y):
                return torch.from_numpy(
                    np.nanmean(np.abs(Y.numpy() - y.numpy()), axis=0)
                )

            reference = file["annotation"].discretize(
                support=Segment(0.0, Audio().get_duration(file)),
                resolution=self._frames,
            )
            oracle_clusters = []

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
                    cost_func=oracle_cost_func,
                )
                active_reference = np.any(local_reference > 0, axis=0)
                oracle_clusters.extend(
                    [
                        i if ((i is not None) and (active_reference[i])) else -1
                        for i in permutation
                    ]
                )

            oracle_clusters = np.array(oracle_clusters)
            oracle = 1.0 * squareform(pdist(oracle_clusters, metric="equal"))
            np.fill_diagonal(oracle, True)
            oracle[oracle_clusters == -1] = -1
            oracle[:, oracle_clusters == -1] = -1

            hook("@clustering/oracle", oracle)

        # __ ACTIVE SPEAKER CLUSTERING _________________________________________________
        # clusters[chunk_id x local_num_speakers + speaker_id] = k
        # * k=-2                if speaker is inactive
        # * k=-1                if speaker is active but not assigned to any cluster
        # * k in {0, ... K - 1} if speaker is active and is assigned to cluster k

        clusters = np.full((num_chunks, local_num_speakers), -1, dtype=np.int)
        clusters[speaker_status == SKIP] = -2

        if num_speakers == 1 or np.sum(speaker_status == LONG) < 2:
            clusters[speaker_status == LONG] = 0
            num_clusters = 1

        else:
            clusters[speaker_status == LONG] = self.clustering(
                embeddings[speaker_status == LONG], num_clusters=num_speakers
            )
            num_clusters = np.max(clusters) + 1

            # corner case where clustering fails to converge and returns only -1 labels
            if num_clusters == 0:
                clusters[speaker_status == LONG] = 0
                num_clusters = 1

        hook("@clustering/clusters", clusters)

        # __ FINAL SPEAKER ASSIGNMENT ___________________________________________________

        centroids = np.vstack(
            [np.mean(embeddings[clusters == k], axis=0) for k in range(num_clusters)]
        )
        unassigned = (speaker_status == KEEP) | (clusters == -1)
        distances = cdist(
            embeddings[unassigned],
            centroids,
            metric=self._embedding.metric,
        )
        clusters[unassigned] = np.argmin(distances, axis=1)

        hook("@clustering/centroids", centroids)
        hook("@clustering/assignment", clusters)

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

        hook("@segmentation/cluster", clustered_segmentations)

        speaker_activations = Inference.aggregate(
            clustered_segmentations,
            frames=self._frames,
            hamming=True,
            missing=0.0,
        )

        hook("@diarization/raw", speaker_activations)

        # __ FINAL BINARIZATION ________________________________________________________
        sorted_speakers = np.argsort(-speaker_activations, axis=-1)
        final_binarized = np.zeros_like(speaker_activations.data)
        for t, ((_, count), speakers) in enumerate(zip(speaker_count, sorted_speakers)):
            # TODO: find a way to stop clustering early enough to avoid num_clusters < count
            count = min(num_clusters, int(count.item()))
            for i in range(count):
                final_binarized[t, speakers[i]] = 1.0

        final_binarized = SlidingWindowFeature(
            final_binarized, speaker_activations.sliding_window
        )
        hook("@diarization/binary", final_binarized)

        diarization = Binarize()(final_binarized)
        diarization.uri = file["uri"]

        # TODO: map `diarization` labels to reference labels when the latter are available.

        return diarization

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)
