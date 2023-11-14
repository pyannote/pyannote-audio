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

import functools
import itertools
import math
import textwrap
import warnings
from typing import Callable, Optional, Text, Union

import numpy as np
import torch
from einops import rearrange
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import ParamDict, Uniform

from pyannote.audio import Audio, Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.clustering import Clustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio.pipelines.utils import (
    PipelineModel,
    SpeakerDiarizationMixin,
    get_model,
)
from pyannote.audio.utils.signal import binarize


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
        Pretrained segmentation model. Defaults to "pyannote/segmentation@2022.07".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    segmentation_step: float, optional
        The segmentation model is applied on a window sliding over the whole audio file.
        `segmentation_step` controls the step of this window, provided as a ratio of its
        duration. Defaults to 0.1 (i.e. 90% overlap between two consecuive windows).
    embedding : Model, str, or dict, optional
        Pretrained embedding model. Defaults to "pyannote/embedding@2022.07".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    embedding_exclude_overlap : bool, optional
        Exclude overlapping speech regions when extracting embeddings.
        Defaults (False) to use the whole speech.
    clustering : str, optional
        Clustering algorithm. See pyannote.audio.pipelines.clustering.Clustering
        for available options. Defaults to "AgglomerativeClustering".
    segmentation_batch_size : int, optional
        Batch size used for speaker segmentation. Defaults to 1.
    embedding_batch_size : int, optional
        Batch size used for speaker embedding. Defaults to 1.
    der_variant : dict, optional
        Optimize for a variant of diarization error rate.
        Defaults to {"collar": 0.0, "skip_overlap": False}. This is used in `get_metric`
        when instantiating the metric: GreedyDiarizationErrorRate(**der_variant).
    use_auth_token : str, optional
        When loading private huggingface.co models, set `use_auth_token`
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running `huggingface-cli login`

    Usage
    -----
    # perform (unconstrained) diarization
    >>> diarization = pipeline("/path/to/audio.wav")

    # perform diarization, targetting exactly 4 speakers
    >>> diarization = pipeline("/path/to/audio.wav", num_speakers=4)

    # perform diarization, with at least 2 speakers and at most 10 speakers
    >>> diarization = pipeline("/path/to/audio.wav", min_speakers=2, max_speakers=10)

    # perform diarization and get one representative embedding per speaker
    >>> diarization, embeddings = pipeline("/path/to/audio.wav", return_embeddings=True)
    >>> for s, speaker in enumerate(diarization.labels()):
    ...     # embeddings[s] is the embedding of speaker `speaker`

    Hyper-parameters
    ----------------
    segmentation.threshold
    segmentation.min_duration_off
    clustering.???
    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation@2022.07",
        segmentation_step: float = 0.1,
        embedding: PipelineModel = "speechbrain/spkrec-ecapa-voxceleb@5c0be3875fda05e81f3c004ed8c7c06be308de1e",
        embedding_exclude_overlap: bool = False,
        clustering: str = "AgglomerativeClustering",
        embedding_batch_size: int = 1,
        segmentation_batch_size: int = 1,
        der_variant: dict = None,
        use_auth_token: Union[Text, None] = None,
    ):
        super().__init__()

        self.segmentation_model = segmentation
        model: Model = get_model(segmentation, use_auth_token=use_auth_token)

        self.segmentation_step = segmentation_step

        self.embedding = embedding
        self.embedding_batch_size = embedding_batch_size
        self.embedding_exclude_overlap = embedding_exclude_overlap

        self.klustering = clustering

        self.der_variant = der_variant or {"collar": 0.0, "skip_overlap": False}

        segmentation_duration = model.specifications.duration
        self._segmentation = Inference(
            model,
            duration=segmentation_duration,
            step=self.segmentation_step * segmentation_duration,
            skip_aggregation=True,
            batch_size=segmentation_batch_size,
        )
        self._frames: SlidingWindow = self._segmentation.model.example_output.frames

        if self._segmentation.model.specifications.powerset:
            self.segmentation = ParamDict(
                min_duration_off=Uniform(0.0, 1.0),
            )

        else:
            self.segmentation = ParamDict(
                threshold=Uniform(0.1, 0.9),
                min_duration_off=Uniform(0.0, 1.0),
            )

        if self.klustering == "OracleClustering":
            metric = "not_applicable"

        else:
            self._embedding = PretrainedSpeakerEmbedding(
                self.embedding, use_auth_token=use_auth_token
            )
            self._audio = Audio(sample_rate=self._embedding.sample_rate, mono="downmix")
            metric = self._embedding.metric

        try:
            Klustering = Clustering[clustering]
        except KeyError:
            raise ValueError(
                f'clustering must be one of [{", ".join(list(Clustering.__members__))}]'
            )
        self.clustering = Klustering.value(metric=metric)

    @property
    def segmentation_batch_size(self) -> int:
        return self._segmentation.batch_size

    @segmentation_batch_size.setter
    def segmentation_batch_size(self, batch_size: int):
        self._segmentation.batch_size = batch_size

    def default_parameters(self):
        raise NotImplementedError()

    def classes(self):
        speaker = 0
        while True:
            yield f"SPEAKER_{speaker:02d}"
            speaker += 1

    @property
    def CACHED_SEGMENTATION(self):
        return "training_cache/segmentation"

    def get_segmentations(self, file, hook=None) -> SlidingWindowFeature:
        """Apply segmentation model

        Parameter
        ---------
        file : AudioFile
        hook : Optional[Callable]

        Returns
        -------
        segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
        """

        if hook is not None:
            hook = functools.partial(hook, "segmentation", None)

        if self.training:
            if self.CACHED_SEGMENTATION in file:
                segmentations = file[self.CACHED_SEGMENTATION]
            else:
                segmentations = self._segmentation(file, hook=hook)
                file[self.CACHED_SEGMENTATION] = segmentations
        else:
            segmentations: SlidingWindowFeature = self._segmentation(file, hook=hook)

        return segmentations

    def get_embeddings(
        self,
        file,
        binary_segmentations: SlidingWindowFeature,
        exclude_overlap: bool = False,
        hook: Optional[Callable] = None,
    ):
        """Extract embeddings for each (chunk, speaker) pair

        Parameters
        ----------
        file : AudioFile
        binary_segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
            Binarized segmentation.
        exclude_overlap : bool, optional
            Exclude overlapping speech regions when extracting embeddings.
            In case non-overlapping speech is too short, use the whole speech.
        hook: Optional[Callable]
            Called during embeddings after every batch to report the progress

        Returns
        -------
        embeddings : (num_chunks, num_speakers, dimension) array
        """

        # when optimizing the hyper-parameters of this pipeline with frozen
        # "segmentation.threshold", one can reuse the embeddings from the first trial,
        # bringing a massive speed up to the optimization process (and hence allowing to use
        # a larger search space).
        if self.training:
            # we only re-use embeddings if they were extracted based on the same value of the
            # "segmentation.threshold" hyperparameter or if the segmentation model relies on
            # `powerset` mode
            cache = file.get("training_cache/embeddings", dict())
            if ("embeddings" in cache) and (
                self._segmentation.model.specifications.powerset
                or (cache["segmentation.threshold"] == self.segmentation.threshold)
            ):
                return cache["embeddings"]

        duration = binary_segmentations.sliding_window.duration
        num_chunks, num_frames, num_speakers = binary_segmentations.data.shape

        if exclude_overlap:
            # minimum number of samples needed to extract an embedding
            # (a lower number of samples would result in an error)
            min_num_samples = self._embedding.min_num_samples

            # corresponding minimum number of frames
            num_samples = duration * self._embedding.sample_rate
            min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)

            # zero-out frames with overlapping speech
            clean_frames = 1.0 * (
                np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2
            )
            clean_segmentations = SlidingWindowFeature(
                binary_segmentations.data * clean_frames,
                binary_segmentations.sliding_window,
            )

        else:
            min_num_frames = -1
            clean_segmentations = SlidingWindowFeature(
                binary_segmentations.data, binary_segmentations.sliding_window
            )

        def iter_waveform_and_mask():
            for (chunk, masks), (_, clean_masks) in zip(
                binary_segmentations, clean_segmentations
            ):
                # chunk: Segment(t, t + duration)
                # masks: (num_frames, local_num_speakers) np.ndarray

                waveform, _ = self._audio.crop(
                    file,
                    chunk,
                    duration=duration,
                    mode="pad",
                )
                # waveform: (1, num_samples) torch.Tensor

                # mask may contain NaN (in case of partial stitching)
                masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
                clean_masks = np.nan_to_num(clean_masks, nan=0.0).astype(np.float32)

                for mask, clean_mask in zip(masks.T, clean_masks.T):
                    # mask: (num_frames, ) np.ndarray

                    if np.sum(clean_mask) > min_num_frames:
                        used_mask = clean_mask
                    else:
                        used_mask = mask

                    yield waveform[None], torch.from_numpy(used_mask)[None]
                    # w: (1, 1, num_samples) torch.Tensor
                    # m: (1, num_frames) torch.Tensor

        batches = batchify(
            iter_waveform_and_mask(),
            batch_size=self.embedding_batch_size,
            fillvalue=(None, None),
        )

        batch_count = math.ceil(num_chunks * num_speakers / self.embedding_batch_size)

        embedding_batches = []

        if hook is not None:
            hook("embeddings", None, total=batch_count, completed=0)

        for i, batch in enumerate(batches, 1):
            waveforms, masks = zip(*filter(lambda b: b[0] is not None, batch))

            waveform_batch = torch.vstack(waveforms)
            # (batch_size, 1, num_samples) torch.Tensor

            mask_batch = torch.vstack(masks)
            # (batch_size, num_frames) torch.Tensor

            embedding_batch: np.ndarray = self._embedding(
                waveform_batch, masks=mask_batch
            )
            # (batch_size, dimension) np.ndarray

            embedding_batches.append(embedding_batch)

            if hook is not None:
                hook("embeddings", embedding_batch, total=batch_count, completed=i)

        embedding_batches = np.vstack(embedding_batches)

        embeddings = rearrange(embedding_batches, "(c s) d -> c s d", c=num_chunks)

        # caching embeddings for subsequent trials
        # (see comments at the top of this method for more details)
        if self.training:
            if self._segmentation.model.specifications.powerset:
                file["training_cache/embeddings"] = {
                    "embeddings": embeddings,
                }
            else:
                file["training_cache/embeddings"] = {
                    "segmentation.threshold": self.segmentation.threshold,
                    "embeddings": embeddings,
                }

        return embeddings

    def reconstruct(
        self,
        segmentations: SlidingWindowFeature,
        hard_clusters: np.ndarray,
        count: SlidingWindowFeature,
    ) -> SlidingWindowFeature:
        """Build final discrete diarization out of clustered segmentation

        Parameters
        ----------
        segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
            Raw speaker segmentation.
        hard_clusters : (num_chunks, num_speakers) array
            Output of clustering step.
        count : (total_num_frames, 1) SlidingWindowFeature
            Instantaneous number of active speakers.

        Returns
        -------
        discrete_diarization : SlidingWindowFeature
            Discrete (0s and 1s) diarization.
        """

        num_chunks, num_frames, local_num_speakers = segmentations.data.shape

        num_clusters = np.max(hard_clusters) + 1
        clustered_segmentations = np.NAN * np.zeros(
            (num_chunks, num_frames, num_clusters)
        )

        for c, (cluster, (chunk, segmentation)) in enumerate(
            zip(hard_clusters, segmentations)
        ):
            # cluster is (local_num_speakers, )-shaped
            # segmentation is (num_frames, local_num_speakers)-shaped
            for k in np.unique(cluster):
                if k == -2:
                    continue

                # TODO: can we do better than this max here?
                clustered_segmentations[c, :, k] = np.max(
                    segmentation[:, cluster == k], axis=1
                )

        clustered_segmentations = SlidingWindowFeature(
            clustered_segmentations, segmentations.sliding_window
        )

        return self.to_diarization(clustered_segmentations, count)

    def apply(
        self,
        file: AudioFile,
        num_speakers: int = None,
        min_speakers: int = None,
        max_speakers: int = None,
        return_embeddings: bool = False,
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
        return_embeddings : bool, optional
            Return representative speaker embeddings.
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
        diarization : Annotation
            Speaker diarization
        embeddings : np.array, optional
            Representative speaker embeddings such that `embeddings[i]` is the
            speaker embedding for i-th speaker in diarization.labels().
            Only returned when `return_embeddings` is True.
        """

        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)

        num_speakers, min_speakers, max_speakers = self.set_num_speakers(
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        segmentations = self.get_segmentations(file, hook=hook)
        hook("segmentation", segmentations)
        #   shape: (num_chunks, num_frames, local_num_speakers)

        # binarize segmentation
        if self._segmentation.model.specifications.powerset:
            binarized_segmentations = segmentations
        else:
            binarized_segmentations: SlidingWindowFeature = binarize(
                segmentations,
                onset=self.segmentation.threshold,
                initial_state=False,
            )

        # estimate frame-level number of instantaneous speakers
        count = self.speaker_count(
            binarized_segmentations,
            frames=self._frames,
            warm_up=(0.0, 0.0),
        )
        hook("speaker_counting", count)
        #   shape: (num_frames, 1)
        #   dtype: int

        # exit early when no speaker is ever active
        if np.nanmax(count.data) == 0.0:
            diarization = Annotation(uri=file["uri"])
            if return_embeddings:
                return diarization, np.zeros((0, self._embedding.dimension))

            return diarization

        if self.klustering == "OracleClustering" and not return_embeddings:
            embeddings = None
        else:
            embeddings = self.get_embeddings(
                file,
                binarized_segmentations,
                exclude_overlap=self.embedding_exclude_overlap,
                hook=hook,
            )
            hook("embeddings", embeddings)
            #   shape: (num_chunks, local_num_speakers, dimension)

        hard_clusters, _, centroids = self.clustering(
            embeddings=embeddings,
            segmentations=binarized_segmentations,
            num_clusters=num_speakers,
            min_clusters=min_speakers,
            max_clusters=max_speakers,
            file=file,  # <== for oracle clustering
            frames=self._frames,  # <== for oracle clustering
        )
        # hard_clusters: (num_chunks, num_speakers)
        # centroids: (num_speakers, dimension)

        # number of detected clusters is the number of different speakers
        num_different_speakers = np.max(hard_clusters) + 1

        # detected number of speakers can still be out of bounds
        # (specifically, lower than `min_speakers`), since there could be too few embeddings
        # to make enough clusters with a given minimum cluster size.
        if num_different_speakers < min_speakers or num_different_speakers > max_speakers:
            warnings.warn(textwrap.dedent(
                f"""
                The detected number of speakers ({num_different_speakers}) is outside
                the given bounds [{min_speakers}, {max_speakers}]. This can happen if the
                given audio file is too short to contain {min_speakers} or more speakers.
                Try to lower the desired minimal number of speakers.
                """
            ))

        # during counting, we could possibly overcount the number of instantaneous
        # speakers due to segmentation errors, so we cap the maximum instantaneous number
        # of speakers by the `max_speakers` value
        count.data = np.minimum(count.data, max_speakers).astype(np.int8)

        # reconstruct discrete diarization from raw hard clusters

        # keep track of inactive speakers
        inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
        #   shape: (num_chunks, num_speakers)

        hard_clusters[inactive_speakers] = -2
        discrete_diarization = self.reconstruct(
            segmentations,
            hard_clusters,
            count,
        )
        hook("discrete_diarization", discrete_diarization)

        # convert to continuous diarization
        diarization = self.to_annotation(
            discrete_diarization,
            min_duration_on=0.0,
            min_duration_off=self.segmentation.min_duration_off,
        )
        diarization.uri = file["uri"]

        # at this point, `diarization` speaker labels are integers
        # from 0 to `num_speakers - 1`, aligned with `centroids` rows.

        if "annotation" in file and file["annotation"]:
            # when reference is available, use it to map hypothesized speakers
            # to reference speakers (this makes later error analysis easier
            # but does not modify the actual output of the diarization pipeline)
            _, mapping = self.optimal_mapping(
                file["annotation"], diarization, return_mapping=True
            )

            # in case there are more speakers in the hypothesis than in
            # the reference, those extra speakers are missing from `mapping`.
            # we add them back here
            mapping = {key: mapping.get(key, key) for key in diarization.labels()}

        else:
            # when reference is not available, rename hypothesized speakers
            # to human-readable SPEAKER_00, SPEAKER_01, ...
            mapping = {
                label: expected_label
                for label, expected_label in zip(diarization.labels(), self.classes())
            }

        diarization = diarization.rename_labels(mapping=mapping)

        # at this point, `diarization` speaker labels are strings (or mix of
        # strings and integers when reference is available and some hypothesis
        # speakers are not present in the reference)

        if not return_embeddings:
            return diarization

        # this can happen when we use OracleClustering
        if centroids is None:
            return diarization, None

        # The number of centroids may be smaller than the number of speakers
        # in the annotation. This can happen if the number of active speakers
        # obtained from `speaker_count` for some frames is larger than the number
        # of clusters obtained from `clustering`. In this case, we append zero embeddings
        # for extra speakers
        if len(diarization.labels()) > centroids.shape[0]:
            centroids = np.pad(centroids, ((0, len(diarization.labels()) - centroids.shape[0]), (0, 0)))

        # re-order centroids so that they match
        # the order given by diarization.labels()
        inverse_mapping = {label: index for index, label in mapping.items()}
        centroids = centroids[
            [inverse_mapping[label] for label in diarization.labels()]
        ]

        return diarization, centroids

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(**self.der_variant)
