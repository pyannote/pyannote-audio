# The MIT License (MIT)
#
# Copyright (c) 2021-2025 CNRS
# Copyright (c) 2025- pyannoteAI
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
from pathlib import Path
from typing import Callable, Mapping, Optional, Text, Union, Any
from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange
from pyannote.audio import Audio, Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.clustering import Clustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio.pipelines.utils import (
    PipelineModel,
    PipelinePLDA,
    SpeakerDiarizationMixin,
    get_model,
    get_plda,
)
from pyannote.audio.pipelines.utils.diarization import set_num_speakers
from pyannote.audio.utils.signal import binarize
from pyannote.core import Annotation, SlidingWindowFeature
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import ParamDict, Uniform


def batchify(iterable, batch_size: int = 32, fillvalue=None):
    """Batchify iterable"""
    # batchify('ABCDEFG', 3) --> ['A', 'B', 'C']  ['D', 'E', 'F']  [G, ]
    args = [iter(iterable)] * batch_size
    return itertools.zip_longest(*args, fillvalue=fillvalue)


@dataclass
class DiarizeOutput:
    # speaker diarization
    speaker_diarization: Annotation

    # speaker diarization adapted to downstream transcription
    # (does not contain overlapping speech turns)
    exclusive_speaker_diarization: Annotation

    # one speaker embedding per speaker
    # as (num_speakers, dimension) array
    # sorted in speaker_diarization.labels() order
    speaker_embeddings: np.ndarray | None = None


    def serialize(self) -> dict[str, Any]:
        """Serialize diarization output

        Example
        -------
        {
            'diarization': [{
                'start': 6.665,
                'end': 7.165,
                'speaker': 'SPEAKER_00'},
                ...],
            'exclusive_diarization': [{
                'start': 6.665,
                'end': 7.165,
                'speaker': 'SPEAKER_00'},
                ...],
        }
        """

        diarization = []
        for speech_turn, _, speaker in self.speaker_diarization.itertracks(
            yield_label=True
        ):
            diarization.append(
                {
                    "start": round(speech_turn.start, 3),
                    "end": round(speech_turn.end, 3),
                    "speaker": speaker,
                }
            )

        exclusive_diarization = []
        for speech_turn, _, speaker in self.exclusive_speaker_diarization.itertracks(
            yield_label=True
        ):
            exclusive_diarization.append(
                {
                    "start": round(speech_turn.start, 3),
                    "end": round(speech_turn.end, 3),
                    "speaker": speaker,
                }
            )

        return {
            "diarization": diarization,
            "exclusive_diarization": exclusive_diarization,
        }


class SpeakerDiarization(SpeakerDiarizationMixin, Pipeline):
    """Speaker diarization pipeline

    Parameters
    ----------
    legacy : bool, optional
        Return only the diarization output. Defaults to return the full output
        with diarization, exclusive diarization, and speaker embeddings.
    segmentation : Model, str, or dict, optional
        Pretrained segmentation model. 
        See pyannote.audio.pipelines.utils.get_model for supported format.
    segmentation_step: float, optional
        The segmentation model is applied on a window sliding over the whole audio file.
        `segmentation_step` controls the step of this window, provided as a ratio of its
        duration. Defaults to 0.1 (i.e. 90% overlap between two consecuive windows).
    embedding : Model, str, or dict, optional
        Pretrained embedding model. 
        See pyannote.audio.pipelines.utils.get_model for supported format.
    embedding_exclude_overlap : bool, optional
        Exclude overlapping speech regions when extracting embeddings.
        Defaults (False) to use the whole speech.
    plda : PLDA, str, or dict, optional
        Pretrained PLDA.
        See pyannote.audio.pipelines.utils.get_plda for supported format.
    clustering : str, optional
        Clustering algorithm. See pyannote.audio.pipelines.clustering.Clustering
        for available options. 
    segmentation_batch_size : int, optional
        Batch size used for speaker segmentation. Defaults to 1.
    embedding_batch_size : int, optional
        Batch size used for speaker embedding. Defaults to 1.
    embedding_precision : torch.dtype, optional
        When set (e.g. torch.float16), use autocast on the embedding model's
        forward_frames pass to leverage GPU tensor cores. Only applies on CUDA
        and when the embedding model supports the forward_frames/forward_embedding
        split API (e.g. WeSpeaker). The pooling step remains in float32 to preserve
        embedding precision for cosine-similarity clustering. Defaults to None (fp32).
    der_variant : dict, optional
        Optimize for a variant of diarization error rate.
        Defaults to {"collar": 0.0, "skip_overlap": False}. This is used in `get_metric`
        when instantiating the metric: GreedyDiarizationErrorRate(**der_variant).
    token : str or bool, optional
        Huggingface token to be used for downloading from Huggingface hub.
    cache_dir: Path or str, optional
        Path to the folder where files downloaded from Huggingface hub are stored.
        
    Usage
    -----
    # process audio file
    >>> output = pipeline("/path/to/audio.wav")

    # print diarization
    >>> assert isinstance(output.speaker_diarization, pyannote.core.Annotation)
    >>> for turn, speaker in output.speaker_diarization:
    ...     print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    
    # get one speaker embedding per speaker
    >>> assert isinstance(output.speaker_embeddings, np.ndarray)
    >>> for s, speaker in enumerate(output.speaker_diarization.labels()):
    ...     # output.speaker_embeddings[s] is the embedding of speaker `speaker`

    # exclusive diarization is the same as diarization except 
    # that it does not contain overlapping speech segments
    >>> assert isinstance(output.exclusive_speaker_diarization, pyannote.core.Annotation)

    # force exactly 4 speakers
    >>> output = pipeline("/path/to/audio.wav", num_speakers=4)

    # force between 2 and 10 speakers
    >>> output = pipeline("/path/to/audio.wav", min_speakers=2, max_speakers=10)
    """

    def __init__(
        self,
        legacy: bool = False,
        segmentation: PipelineModel = {
            "checkpoint": "pyannote/speaker-diarization-community-1",
            "subfolder": "segmentation",
        },
        segmentation_step: float = 0.1,
        embedding: PipelineModel = {
            "checkpoint": "pyannote/speaker-diarization-community-1",
            "subfolder": "embedding",
        },
        embedding_exclude_overlap: bool = False,
        plda: PipelinePLDA = {
            "checkpoint": "pyannote/speaker-diarization-community-1",
            "subfolder": "plda",
        },
        clustering: str = "VBxClustering",
        embedding_batch_size: int = 1,
        segmentation_batch_size: int = 1,
        embedding_precision: Optional[torch.dtype] = None,
        der_variant: Optional[dict] = None,
        token: Union[Text, None] = None,
        cache_dir: Union[Path, Text, None] = None,
    ):
        super().__init__()

        self.legacy = legacy
        self.embedding_precision = embedding_precision

        self.segmentation_model = segmentation
        model: Model = get_model(segmentation, token=token, cache_dir=cache_dir)

        self.segmentation_step = segmentation_step

        self.embedding = embedding
        self.embedding_batch_size = embedding_batch_size
        self.embedding_exclude_overlap = embedding_exclude_overlap

        self.plda = plda
        self._plda = get_plda(plda, token=token, cache_dir=cache_dir)

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
                self.embedding, token=token, cache_dir=cache_dir
            )
            self._audio = Audio(sample_rate=self._embedding.sample_rate, mono="downmix")
            metric = self._embedding.metric

        try:
            Klustering = Clustering[clustering]
        except KeyError:
            raise ValueError(
                f"clustering must be one of [{', '.join(list(Clustering.__members__))}]"
            )

        if self.klustering == "VBxClustering":
            self.clustering = Klustering.value(self._plda, metric=metric)
        else:
            self.clustering = Klustering.value(metric=metric)

        self._expects_num_speakers = self.clustering.expects_num_clusters

    @property
    def segmentation_batch_size(self) -> int:
        return self._segmentation.batch_size

    @segmentation_batch_size.setter
    def segmentation_batch_size(self, batch_size: int):
        self._segmentation.batch_size = batch_size

    def default_parameters(self):
        return {
            "segmentation": {"min_duration_off": 0.0},
            "clustering": {"threshold": 0.6, "Fa": 0.07, "Fb": 0.8},
        }

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

    def _get_masks(
        self,
        binary_segmentations: SlidingWindowFeature,
        exclude_overlap: bool = False,
    ):
        """Compute masks for embedding extraction

        Parameters
        ----------
        binary_segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
            Binarized segmentation.
        exclude_overlap : bool, optional
            Exclude overlapping speech regions when extracting embeddings.

        Returns
        -------
        masks : (num_chunks, num_frames, num_speakers) np.ndarray
        clean_masks : (num_chunks, num_frames, num_speakers) np.ndarray
        min_num_frames : int
        """

        duration = binary_segmentations.sliding_window.duration
        num_chunks, num_frames, num_speakers = binary_segmentations.data.shape

        masks = np.nan_to_num(binary_segmentations.data, nan=0.0).astype(np.float32)

        if exclude_overlap:
            min_num_samples = self._embedding.min_num_samples
            num_samples = duration * self._embedding.sample_rate
            min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)

            overlap = np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2
            clean_masks = np.nan_to_num(
                binary_segmentations.data * overlap, nan=0.0
            ).astype(np.float32)
        else:
            min_num_frames = -1
            clean_masks = masks

        return masks, clean_masks, min_num_frames

    def _get_active_pairs(
        self,
        masks: np.ndarray,
        clean_masks: np.ndarray,
        min_num_frames: int,
    ):
        """Find active (chunk, speaker) pairs and their resolved masks

        Parameters
        ----------
        masks : (num_chunks, num_frames, num_speakers) np.ndarray
        clean_masks : (num_chunks, num_frames, num_speakers) np.ndarray
        min_num_frames : int

        Returns
        -------
        active_pairs : list of (chunk_idx, speaker_idx)
        active_masks : dict mapping (chunk_idx, speaker_idx) to (num_frames,) np.ndarray
        """

        num_chunks, _, num_speakers = masks.shape
        active_pairs = []
        active_masks = {}

        for c in range(num_chunks):
            for s in range(num_speakers):
                cm = clean_masks[c, :, s]
                m = masks[c, :, s]
                used = cm if np.sum(cm) > min_num_frames else m
                if np.sum(used) == 0.0:
                    continue
                active_pairs.append((c, s))
                active_masks[(c, s)] = used

        return active_pairs, active_masks

    def _preslice_waveforms(
        self,
        file,
        binary_segmentations: SlidingWindowFeature,
    ) -> torch.Tensor:
        """Pre-slice all chunk waveforms from in-memory audio

        Parameters
        ----------
        file : AudioFile
        binary_segmentations : SlidingWindowFeature

        Returns
        -------
        all_waveforms : (num_chunks, 1, window_samples) torch.Tensor
        """

        duration = binary_segmentations.sliding_window.duration
        num_chunks = binary_segmentations.data.shape[0]
        waveform = file["waveform"]
        sample_rate = file["sample_rate"]
        window_samples = int(duration * sample_rate)

        sw = binary_segmentations.sliding_window
        start_samples = np.array(
            [int(sw[i].start * sample_rate) for i in range(num_chunks)]
        )

        all_waveforms = torch.zeros(
            (num_chunks, 1, window_samples), dtype=waveform.dtype
        )
        total_samples = waveform.shape[1]

        for i, s in enumerate(start_samples):
            src_start = max(s, 0)
            src_end = min(s + window_samples, total_samples)
            dst_start = max(-s, 0)
            length = src_end - src_start
            if length > 0:
                all_waveforms[i, :, dst_start : dst_start + length] = waveform[
                    :, src_start:src_end
                ]

        return all_waveforms

    def _has_split_embedding_api(self) -> bool:
        """Check if the embedding model supports forward_frames/forward_embedding"""
        model = getattr(self._embedding, "model_", None)
        return model is not None and hasattr(model, "forward_frames") and hasattr(model, "forward_embedding")

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
            cache = file.get("training_cache/embeddings", dict())
            if ("embeddings" in cache) and (
                self._segmentation.model.specifications.powerset
                or (cache["segmentation.threshold"] == self.segmentation.threshold)
            ):
                return cache["embeddings"]

        num_chunks, num_frames, num_speakers = binary_segmentations.data.shape

        # use the fast path when:
        # - audio is already in memory (no disk I/O per chunk)
        # - embedding model supports split forward_frames/forward_embedding
        has_waveform = isinstance(file, Mapping) and "waveform" in file
        has_split_api = self._has_split_embedding_api()

        if has_waveform and has_split_api:
            embeddings = self._get_embeddings_fast(
                file, binary_segmentations, exclude_overlap, hook
            )
        else:
            embeddings = self._get_embeddings_legacy(
                file, binary_segmentations, exclude_overlap, hook
            )

        # caching embeddings for subsequent trials
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

    def _get_embeddings_fast(
        self,
        file,
        binary_segmentations: SlidingWindowFeature,
        exclude_overlap: bool = False,
        hook: Optional[Callable] = None,
    ):
        """Fast embedding extraction path

        Optimizations over the legacy path:
        - pre-slices all chunk waveforms at once (no per-chunk audio.crop)
        - skips inactive (chunk, speaker) pairs with all-zero masks
        - computes forward_frames once per chunk, then forward_embedding per active pair

        Parameters
        ----------
        file : AudioFile
            Must contain "waveform" and "sample_rate" keys.
        binary_segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
        exclude_overlap : bool, optional
        hook : Optional[Callable]

        Returns
        -------
        embeddings : (num_chunks, num_speakers, dimension) np.ndarray
        """

        num_chunks, num_frames, num_speakers = binary_segmentations.data.shape
        batch_size = self.embedding_batch_size
        model = self._embedding.model_
        device = next(model.parameters()).device

        masks, clean_masks, min_num_frames = self._get_masks(
            binary_segmentations, exclude_overlap
        )
        active_pairs, active_masks = self._get_active_pairs(
            masks, clean_masks, min_num_frames
        )

        # pre-slice all chunk waveforms from in-memory audio
        all_waveforms = self._preslice_waveforms(file, binary_segmentations)

        # Step 1: compute frame-level features once per chunk
        # (fbank extraction + ResNet forward, the expensive part)
        # When embedding_precision is set (e.g. torch.float16), use autocast
        # on the forward_frames pass to leverage GPU tensor cores.
        # The resulting frames are cast back to float32 before pooling
        # to preserve embedding precision for cosine-similarity clustering.
        use_autocast = (
            self.embedding_precision is not None and device.type == "cuda"
        )

        all_frames = []
        with torch.inference_mode():
            for i in range(0, num_chunks, batch_size):
                batch = all_waveforms[i : i + batch_size].to(device)
                if use_autocast:
                    with torch.autocast(
                        device_type="cuda", dtype=self.embedding_precision
                    ):
                        frames = model.forward_frames(batch)
                    frames = frames.float()
                else:
                    frames = model.forward_frames(batch)
                all_frames.append(frames)
            all_frames = torch.cat(all_frames, dim=0)

        # Step 2: compute embeddings only for active (chunk, speaker) pairs
        # (statistics pooling only, much cheaper than the full forward pass)
        dimension = self._embedding.dimension
        embeddings = np.full(
            (num_chunks * num_speakers, dimension), np.nan, dtype=np.float32
        )

        total_pool_batches = math.ceil(max(1, len(active_pairs)) / batch_size)

        if hook is not None:
            hook("embeddings", None, total=total_pool_batches, completed=0)

        with torch.inference_mode():
            for bi, i in enumerate(
                range(0, len(active_pairs), batch_size), start=1
            ):
                batch_pairs = active_pairs[i : i + batch_size]

                frames_batch = torch.stack(
                    [all_frames[c] for c, _ in batch_pairs], dim=0
                )
                masks_batch = torch.stack(
                    [
                        torch.from_numpy(active_masks[(c, s)])
                        for c, s in batch_pairs
                    ],
                    dim=0,
                ).to(device=device, dtype=torch.float32)

                emb = model.forward_embedding(frames_batch, weights=masks_batch)
                emb_np = emb.cpu().numpy()

                for j, (c, s) in enumerate(batch_pairs):
                    embeddings[c * num_speakers + s] = emb_np[j]

                if hook is not None:
                    hook(
                        "embeddings", emb_np, total=total_pool_batches, completed=bi
                    )

        return rearrange(embeddings, "(c s) d -> c s d", c=num_chunks)

    def _get_embeddings_legacy(
        self,
        file,
        binary_segmentations: SlidingWindowFeature,
        exclude_overlap: bool = False,
        hook: Optional[Callable] = None,
    ):
        """Legacy embedding extraction path

        Used as fallback when audio is not in memory or the embedding model
        does not support the forward_frames/forward_embedding split API.

        Parameters
        ----------
        file : AudioFile
        binary_segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
        exclude_overlap : bool, optional
        hook : Optional[Callable]

        Returns
        -------
        embeddings : (num_chunks, num_speakers, dimension) np.ndarray
        """

        duration = binary_segmentations.sliding_window.duration
        num_chunks, num_frames, num_speakers = binary_segmentations.data.shape

        if exclude_overlap:
            min_num_samples = self._embedding.min_num_samples
            num_samples = duration * self._embedding.sample_rate
            min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)

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
                waveform, _ = self._audio.crop(
                    file,
                    chunk,
                    mode="pad",
                )

                masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
                clean_masks = np.nan_to_num(clean_masks, nan=0.0).astype(np.float32)

                for mask, clean_mask in zip(masks.T, clean_masks.T):
                    if np.sum(clean_mask) > min_num_frames:
                        used_mask = clean_mask
                    else:
                        used_mask = mask

                    yield waveform[None], torch.from_numpy(used_mask)[None]

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
            mask_batch = torch.vstack(masks)

            embedding_batch: np.ndarray = self._embedding(
                waveform_batch, masks=mask_batch
            )

            embedding_batches.append(embedding_batch)

            if hook is not None:
                hook("embeddings", embedding_batch, total=batch_count, completed=i)

        embedding_batches = np.vstack(embedding_batches)

        return rearrange(embedding_batches, "(c s) d -> c s d", c=num_chunks)

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
        clustered_segmentations = np.nan * np.zeros(
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
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        hook: Optional[Callable] = None,
        **kwargs,
    ) -> DiarizeOutput | Annotation:
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
            Callback called after each major steps of the pipeline as follows:
                hook(step_name,      # human-readable name of current step
                     step_artefact,  # artifact generated by current step
                     file=file)      # file being processed
            Time-consuming steps call `hook` multiple times with the same `step_name`
            and additional `completed` and `total` keyword arguments usable to track
            progress of current step.

        Returns
        -------
        output : DiarizeOutput (or Annotation if `self.legacy` is True)
        """

        # warn about unsupported kwargs
        if len(kwargs) > 0:
            warnings.warn(
                f"Ignoring unexpected keyword arguments: {', '.join(list(kwargs.keys()))}"
            )

        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)

        num_speakers, min_speakers, max_speakers = set_num_speakers(
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        # when using KMeans clustering (or equivalent), the number of speakers must
        # be provided alongside the audio file. also, during pipeline training, we
        # infer the number of speakers from the reference annotation to avoid the
        # pipeline complaining about missing number of speakers.
        if self._expects_num_speakers and num_speakers is None:
            if isinstance(file, Mapping) and "annotation" in file:
                num_speakers = len(file["annotation"].labels())

            else:
                raise ValueError(
                    f"num_speakers must be provided when using {self.klustering} clustering"
                )

        segmentations = self.get_segmentations(file, hook=hook)
        hook("segmentation", segmentations)
        #   shape: (num_chunks, num_frames, local_num_speakers)
        num_chunks, num_frames, local_num_speakers = segmentations.data.shape

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
            self._segmentation.model.receptive_field,
            warm_up=(0.0, 0.0),
        )
        hook("speaker_counting", count)
        #   shape: (num_frames, 1)
        #   dtype: int

        # exit early when no speaker is ever active
        if np.nanmax(count.data) == 0.0:
            output = DiarizeOutput(
                speaker_diarization=Annotation(uri=file["uri"]),
                exclusive_speaker_diarization=Annotation(uri=file["uri"]),
                speaker_embeddings=np.zeros((0, self._embedding.dimension)),
            )

            if self.legacy:
                return output.speaker_diarization

            return output

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
            frames=self._segmentation.model.receptive_field,  # <== for oracle clustering
        )
        # hard_clusters: (num_chunks, num_speakers)
        # centroids: (num_speakers, dimension)

        # number of detected clusters is the number of different speakers
        num_different_speakers = np.max(hard_clusters) + 1

        # detected number of speakers can still be out of bounds
        # (specifically, lower than `min_speakers`), since there could be too few embeddings
        # to make enough clusters with a given minimum cluster size.
        if (
            num_different_speakers < min_speakers
            or num_different_speakers > max_speakers
        ):
            warnings.warn(
                textwrap.dedent(
                    f"""
                The detected number of speakers ({num_different_speakers}) for {file["uri"]} is outside
                the given bounds [{min_speakers}, {max_speakers}]. This can happen if the
                given audio file is too short to contain {min_speakers} or more speakers.
                Try to lower the desired minimal number of speakers.
                """
                )
            )

        # during counting, we could possibly overcount the number of instantaneous
        # speakers due to segmentation errors, so we cap the maximum instantaneous number
        # of speakers by the `max_speakers` value
        count.data = np.minimum(count.data, max_speakers).astype(np.int8)

        # reconstruct discrete diarization from raw hard clusters

        # keep track of inactive speakers
        inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
        #   shape: (num_chunks, num_speakers)

        # force-assign them to throw-away cluster
        hard_clusters[inactive_speakers] = -2

        # convert to continuous diarization
        discrete_diarization = self.reconstruct(
            segmentations,
            hard_clusters,
            count,
        )
        hook("discrete_diarization", discrete_diarization)
        diarization = self.to_annotation(
            discrete_diarization,
            min_duration_on=0.0,
            min_duration_off=self.segmentation.min_duration_off,
        )
        diarization.uri = file["uri"]

        # convert to continuous exclusive diarization
        count.data = np.minimum(count.data, 1).astype(np.int8)
        exclusive_discrete_diarization = self.reconstruct(
            segmentations,
            hard_clusters,
            count,
        )
        exclusive_diarization = self.to_annotation(
            exclusive_discrete_diarization,
            min_duration_on=0.0,
            min_duration_off=self.segmentation.min_duration_off,
        )
        exclusive_diarization.uri = file["uri"]

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
        exclusive_diarization = exclusive_diarization.rename_labels(mapping=mapping)

        # at this point, `diarization` speaker labels are strings (or mix of
        # strings and integers when reference is available and some hypothesis
        # speakers are not present in the reference)

        # centroids may be None when we use OracleClustering
        if centroids is None:
            output = DiarizeOutput(
                speaker_diarization=diarization,
                exclusive_speaker_diarization=exclusive_diarization,
                speaker_embeddings=centroids,
            )
            if self.legacy:
                return output.speaker_diarization

            return output

        # The number of centroids may be smaller than the number of speakers
        # in the annotation. This can happen if the number of active speakers
        # obtained from `speaker_count` for some frames is larger than the number
        # of clusters obtained from `clustering`. In this case, we append zero embeddings
        # for extra speakers
        if len(diarization.labels()) > centroids.shape[0]:
            centroids = np.pad(
                centroids, ((0, len(diarization.labels()) - centroids.shape[0]), (0, 0))
            )

        # re-order centroids so that they match
        # the order given by diarization.labels()
        inverse_mapping = {label: index for index, label in mapping.items()}
        centroids = centroids[
            [inverse_mapping[label] for label in diarization.labels()]
        ]

        output = DiarizeOutput(
            speaker_diarization=diarization,
            exclusive_speaker_diarization=exclusive_diarization,
            speaker_embeddings=centroids,
        )

        if self.legacy:
            return output.speaker_diarization

        return output

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(**self.der_variant)
