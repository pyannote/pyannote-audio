# The MIT License (MIT)
#
# Copyright (c) 2021- CNRS
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

import itertools
import warnings
from typing import Callable, Optional, Text

import numpy as np
import torch
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature
from pyannote.core.utils.distance import pdist
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Uniform
from scipy.spatial.distance import squareform

from pyannote.audio import Audio, Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import (
    PipelineModel,
    SpeakerDiarizationMixin,
    get_devices,
    get_model,
)
from pyannote.audio.utils.permutation import mae_cost_func, permutate
from pyannote.audio.utils.signal import binarize

from .clustering import Clustering, NearestClusterAssignment
from .speaker_verification import PretrainedSpeakerEmbedding


def batchify(iterable, batch_size: int = 32, fillvalue=None):
    """Batchify iterable"""
    # batchify('ABCDEFG', 3) --> ['A', 'B', 'C']  ['D', 'E', 'F']  [G, ]
    args = [iter(iterable)] * batch_size
    return itertools.zip_longest(*args, fillvalue=fillvalue)


class SpeakerDiarization(SpeakerDiarizationMixin, Pipeline):
    """Speaker diarization pipeline

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation model. Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    embedding : Model, str, or dict, optional
        Pretrained embedding model. Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    clustering : {"AgglomerativeClustering", "SpectralClustering"}, optional
        Defaults to "AgglomerativeClustering".
    expects_num_speakers : bool, optional
        Defaults to False.
    segmentation_batch_size : int, optional
        Batch size used for speaker segmentation. Defaults to 32.
    embedding_batch_size : int, optional
        Batch size used for speaker embedding. Defaults to 32.

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
        segmentation_batch_size: int = 32,
        embedding_batch_size: int = 32,
    ):

        super().__init__()

        self.segmentation = segmentation
        self.embedding = embedding
        self.klustering = clustering
        self.expects_num_speakers = expects_num_speakers
        self.segmentation_batch_size = segmentation_batch_size

        seg_device, emb_device = get_devices(needs=2)

        model: Model = get_model(segmentation)
        model.to(seg_device)
        self._segmentation = Inference(
            model, skip_aggregation=True, batch_size=self.segmentation_batch_size
        )
        self._frames: SlidingWindow = self._segmentation.model.introspection.frames

        self._embedding = PretrainedSpeakerEmbedding(self.embedding, device=emb_device)
        self.embedding_batch_size = embedding_batch_size or len(
            model.specifications.classes
        )

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
        self.assignment = NearestClusterAssignment(
            metric=self._embedding.metric, allow_reassignment=True
        )

        # hyper-parameters

        # stitching
        self.warm_up = 0.05
        self.stitch_threshold = Uniform(0.0, 1.0)

        # hyper-parameters used for hysteresis thresholding
        self.onset = Uniform(0.01, 0.99)
        self.offset = Uniform(0.01, 0.99)

        # hyper-parameters used for post-processing i.e. removing short speech turns
        # or filling short gaps between speech turns of one speaker
        self.min_duration_on = Uniform(0.0, 1.0)
        self.min_duration_off = Uniform(0.0, 1.0)

        # minimum amount of speech needed to use speaker in clustering
        self.min_activity = Uniform(0.0, 10.0)

    def default_parameters(self):
        # parameters optimized on DIHARD 3 development set
        if (
            self.segmentation == "pyannote/segmentation"
            and self.embedding == "speechbrain/spkrec-ecapa-voxceleb"
            and self.klustering == "AgglomerativeClustering"
            and not self.expects_num_speakers
        ):
            return {
                "onset": 0.810,
                "offset": 0.481,
                "min_duration_on": 0.055,
                "min_duration_off": 0.098,
                "min_activity": 6.073,
                "stitch_threshold": 0.040,
                "clustering": {"method": "average", "threshold": 0.595},
            }

        raise NotImplementedError()

    def classes(self):
        speaker = 0
        while True:
            yield f"SPEAKER_{speaker:02d}"
            speaker += 1

    CACHED_SEGMENTATION = "cache/segmentation/inference"

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
        min_speakers: int = None,
        max_speakers: int = None,
        hook: Optional[Callable] = None,
    ) -> Annotation:
        """Apply speaker diarization

        Parameters
        ----------
        file : AudioFile
            Processed file.
        num_speakers : int, optional
            Number of speakers, when known.
        min_speakers : int, optional
            Minimum number of speakers. Has no effect when `num_speakers` is provided.
        max_speakers : int, optional
            Maximum number of speakers. Has no effect when `num_speakers` is provided.
        hook : callable, optional
            Hook called after each major step of the pipeline with the following
            signature: hook("step_name", step_artefact, file=file)

        Returns
        -------
        diarization : Annotation
            Speaker diarization
        """

        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)

        num_speakers, min_speakers, max_speakers = self.set_num_speakers(
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

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

        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, local_num_speakers)
        if self.training:
            if self.CACHED_SEGMENTATION in file:
                segmentations = file[self.CACHED_SEGMENTATION]
            else:
                segmentations = self._segmentation(file)
                file[self.CACHED_SEGMENTATION] = segmentations
        else:
            segmentations: SlidingWindowFeature = self._segmentation(file)

        hook("segmentation", segmentations)

        # estimate frame-level number of instantaneous speakers
        count = self.speaker_count(
            segmentations,
            onset=self.onset,
            offset=self.offset,
            warm_up=(self.warm_up, self.warm_up),
            frames=self._frames,
        )
        hook("speaker_counting", count)

        # trim warm-up regions and stitch
        segmentations = Inference.stitch(
            Inference.trim(segmentations, warm_up=(self.warm_up, self.warm_up)),
            frames=self._frames,
            lookahead=None,  # TODO: make it an hyper-parameter?,
            cost_func=mae_cost_func,
            match_func=self.stitch_match_func,
        )
        hook("local_stitching", segmentations)

        # stitched chunks are longer than original chunks
        stitched_chunk_duration: float = segmentations.sliding_window.duration
        num_chunks, num_frames, local_num_speakers = segmentations.data.shape

        # apply hysteresis thresholding on each (stitched) chunk
        # bs[c, f, s] = 1 iff speaker s is active at frame f of chunk c
        binarized_segmentations: SlidingWindowFeature = binarize(
            segmentations, onset=self.onset, offset=self.offset, initial_state=False
        )
        hook("speaker_activation", binarized_segmentations)

        # mask overlapping speech regions
        # ms[c, f, s] = 1 iff speaker s is the only active speaker at frame f of chunk c
        masked_segmentations = SlidingWindowFeature(
            binarized_segmentations.data
            * (np.sum(binarized_segmentations.data, axis=-1, keepdims=True) == 1.0),
            binarized_segmentations.sliding_window,
        )
        hook("single_speaker_activation", masked_segmentations)

        # skip speakers that are not active
        SKIP = 0
        speaker_status = np.full((num_chunks, local_num_speakers), SKIP, dtype=np.int)

        # keep speakers that are active in at least one frame
        KEEP = 1
        speaker_status[np.any(binarized_segmentations.data, axis=1)] = KEEP

        # return empty annotation in the (corner) case where no speaker is ever active
        if np.sum(speaker_status == KEEP) == 0:
            return Annotation(uri=file["uri"])

        # mark speakers who speaker long enough as usable for clustering
        # (handle corner case where no speaker speaks long enough by lowering
        # value of min_activity until at least one speaker passes the threshold)
        LONG = 2
        min_activity = self.min_activity
        while np.sum(speaker_status == LONG) == 0:
            speaker_status[
                np.mean(masked_segmentations, axis=1) * stitched_chunk_duration
                > min_activity
            ] = LONG
            min_activity *= 0.9

        hook("speaker_status", speaker_status)

        # compute speaker embeddings
        def iter_waveform_and_mask():
            for status, (chunk, ms) in zip(speaker_status, masked_segmentations):
                active = status > SKIP
                if np.sum(active) == 0:
                    continue

                waveform: torch.Tensor = self._audio.crop(file, chunk, mode="pad")[
                    0
                ].unsqueeze(0)
                for s in np.where(active)[0]:
                    mask = torch.from_numpy(ms[:, s]).float()
                    yield waveform, mask

        batches = batchify(
            iter_waveform_and_mask(),
            batch_size=self.embedding_batch_size,
            fillvalue=(None, None),
        )

        embedding_batches = []

        for batch in batches:
            waveforms, masks = zip(*filter(lambda b: b[0] is not None, batch))
            waveform_batch = torch.vstack(waveforms)
            mask_batch = torch.vstack(masks)
            embedding_batch: np.ndarray = self._embedding(
                waveform_batch, masks=mask_batch
            )
            embedding_batches.append(embedding_batch)

        # stack and unit-normalized embeddings
        embeddings = np.zeros(
            (num_chunks, local_num_speakers, self._embedding.dimension)
        )
        embeddings[speaker_status > SKIP] = np.vstack(embedding_batches)

        with np.errstate(divide="ignore", invalid="ignore"):
            embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
        hook("speaker_embedding", embeddings)

        # skip speakers for which embedding extraction failed for some reason
        speaker_status[np.any(np.isnan(embeddings), axis=-1)] = SKIP

        # # compute "cannot link" constraints
        # chunk_idx = np.tile(np.arange(num_chunks), (local_num_speakers, 1)).T
        # same_chunk = 1.0 * squareform(
        #     pdist(rearrange(chunk_idx, "c s -> (c s)"), metric="equal")
        # )
        # # shape is (c s) x (c s)

        if not hook.missing and "annotation" in file:

            # log actual distance matrix

            hook(
                "@clustering/distance",
                squareform(
                    pdist(
                        embeddings[speaker_status == LONG],
                        metric=self._embedding.metric,
                    )
                ),
            )

            # log oracle distance matrix (0 = same speaker / 1 = different speaker)
            # oracle[i, j] = 0 if same speaker
            #              = 1 otherwise
            # same_chunk[i, j] = 1 if same chunk
            #                  = 0 otherwise
            def oracle_cost_func(Y, y):
                return torch.from_numpy(
                    np.nanmean(np.abs(Y.numpy() - y.numpy()), axis=0)
                )

            chunks = segmentations.sliding_window
            reference = file["annotation"].discretize(
                support=Segment(
                    chunks[0].start, chunks[num_chunks - 1].end + chunks.step
                ),
                resolution=self._frames,
            )

            oracle_clusters = []
            chunk_idx = []
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
                chunk_idx.extend(
                    [
                        c if ((i is not None) and (active_reference[i])) else -1
                        for i in permutation
                    ]
                )

            oracle_clusters = np.array(oracle_clusters)
            oracle = 1.0 * squareform(pdist(oracle_clusters, metric="equal"))
            np.fill_diagonal(oracle, True)
            oracle[oracle_clusters == -1] = -1
            oracle[:, oracle_clusters == -1] = -1
            hook("@clustering/oracle", oracle)

            chunk_idx = np.array(chunk_idx)
            same_chunk = 1.0 * squareform(pdist(chunk_idx, metric="equal"))
            np.fill_diagonal(same_chunk, True)
            hook("@clustering/same_chunk", same_chunk)

        # perform clustering on (LONG) speaker embedding
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
                embeddings[speaker_status == LONG],
                num_clusters=num_speakers,
                min_clusters=min_speakers,
                max_clusters=max_speakers,
            )
            num_clusters = np.max(clusters) + 1

            # corner case where clustering fails to converge and returns only -1 labels
            if num_clusters == 0:
                clusters[speaker_status == LONG] = 0
                num_clusters = 1

        hook("clustering", clusters)

        # assign (active) speakers to clusters

        clusters[speaker_status != SKIP] = self.assignment(
            embeddings[speaker_status != SKIP], clusters[speaker_status != SKIP]
        )
        hook("assignment", clusters)

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
        discrete_diarization = self.to_diarization(clustered_segmentations, count)

        hook("global_stitching", discrete_diarization)

        # convert to continuous diarization
        diarization = self.to_annotation(
            discrete_diarization,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )

        diarization.uri = file["uri"]

        if "annotation" in file:
            return self.optimal_mapping(file["annotation"], diarization)

        return diarization.rename_labels(
            {
                label: expected_label
                for label, expected_label in zip(diarization.labels(), self.classes())
            }
        )

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)


class SpeakerDiarizationWithOracleSegmentation(SpeakerDiarization):
    """Speaker diarization pipeline with oracle segmentation"""

    def oracle_segmentation(self, file) -> SlidingWindowFeature:
        file_duration: float = self._audio.get_duration(file)

        reference: Annotation = file["annotation"]
        labels = reference.labels()
        if self._oracle_num_speakers > len(labels):
            num_missing = self._oracle_num_speakers - len(labels)
            labels += [
                f"FakeSpeakerForOracleSegmentationInference{i:d}"
                for i in range(num_missing)
            ]

        window = SlidingWindow(
            start=0.0, duration=self._oracle_duration, step=self._oracle_step
        )

        segmentations = []
        for chunk in window(Segment(0.0, file_duration)):
            chunk_segmentation: SlidingWindowFeature = reference.discretize(
                chunk,
                resolution=self._frames,
                labels=labels,
                duration=self._oracle_duration,
            )
            # keep `self._oracle_num_speakers`` most talkative speakers
            most_talkative_index = np.argsort(-np.sum(chunk_segmentation, axis=0))[
                : self._oracle_num_speakers
            ]

            segmentations.append(chunk_segmentation[:, most_talkative_index])

        return SlidingWindowFeature(np.float32(np.stack(segmentations)), window)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._oracle_duration: float = self._segmentation.duration
        self._oracle_step: float = self._segmentation.step
        self._oracle_num_speakers: int = len(
            self._segmentation.model.specifications.classes
        )
        self._segmentation = self.oracle_segmentation
