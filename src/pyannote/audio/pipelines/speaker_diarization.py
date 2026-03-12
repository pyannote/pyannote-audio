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
import os
import textwrap
import warnings
from pathlib import Path
from typing import Callable, Mapping, Optional, Text, Union, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
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
        der_variant: Optional[dict] = None,
        token: Union[Text, None] = None,
        cache_dir: Union[Path, Text, None] = None,
        # --- Optimization parameters ---
        torch_compile: bool = False,
        mixed_precision: bool = False,
        onnx_cpu: bool = False,
        onnx_quantize: bool = True,
        onnx_num_threads: int = 0,
    ):
        super().__init__()

        self.legacy = legacy

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

        # --- Optimization setup ---
        self._mixed_precision = mixed_precision
        self._onnx_cpu = onnx_cpu

        # torch.compile() for kernel fusion (10-30% faster after warmup)
        if torch_compile and self._segmentation.device.type == "cuda":
            try:
                if self.klustering != "OracleClustering":
                    self._embedding.model_ = torch.compile(
                        self._embedding.model_, mode="reduce-overhead"
                    )
                self._segmentation.model = torch.compile(
                    self._segmentation.model, fullgraph=False
                )
            except Exception:
                pass  # Graceful fallback if torch.compile unavailable

        # ONNX CPU-only mode: export models to ONNX and replace forward passes
        if onnx_cpu:
            self._setup_onnx_cpu(onnx_quantize, onnx_num_threads)

    @property
    def segmentation_batch_size(self) -> int:
        return self._segmentation.batch_size

    @segmentation_batch_size.setter
    def segmentation_batch_size(self, batch_size: int):
        self._segmentation.batch_size = batch_size

    def _setup_onnx_cpu(self, quantize: bool = True, num_threads: int = 0) -> None:
        """Load pre-cached ONNX models for CPU inference (no runtime conversion).

        ONNX models must be pre-converted using scripts/preconvert-onnx-models.py
        before calling this method. This method only loads cached models.

        This frees the GPU entirely for other workloads (e.g., Whisper).

        Args:
            quantize: Whether to prefer INT8 model (if available)
            num_threads: Number of CPU threads for ONNX Runtime
        """
        import os
        try:
            import onnxruntime as ort
        except ImportError:
            warnings.warn("onnxruntime not installed, ONNX CPU mode disabled")
            self._onnx_cpu = False
            return

        cache_dir = os.environ.get("MODEL_CACHE_DIR", "./models")
        onnx_dir = Path(cache_dir) / "onnx"

        # Check for pre-cached ONNX models
        int8_path = onnx_dir / "pyannote_segmentation_int8.onnx"
        fp32_path = onnx_dir / "pyannote_segmentation_fp32.onnx"

        # Prefer INT8 if quantize=True and it exists
        if quantize and int8_path.exists():
            seg_path = int8_path
        elif fp32_path.exists():
            seg_path = fp32_path
        else:
            # No cached models found
            raise FileNotFoundError(
                f"ONNX models not found in {onnx_dir}.\n"
                f"Please pre-convert models once using:\n"
                f"  python scripts/preconvert-onnx-models.py --cache-dir {cache_dir}\n"
                f"This creates cached ONNX models for production use (no runtime conversion)."
            )

        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)

        # Load pre-cached ONNX model
        print(f"Loading ONNX model from cache: {seg_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = max(1, num_threads // 4)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        seg_session = ort.InferenceSession(str(seg_path), opts, ["CPUExecutionProvider"])

        # Monkey-patch segmentation infer() to use ONNX
        def onnx_infer(chunks):
            chunks_np = chunks.numpy() if hasattr(chunks, 'numpy') else np.array(chunks)
            return seg_session.run(["scores"], {"waveforms": chunks_np.astype(np.float32)})[0]
        self._segmentation.infer = onnx_infer

        # Move PyTorch models to CPU to free GPU VRAM
        self._segmentation.model.cpu()
        if hasattr(self, '_embedding') and hasattr(self._embedding, 'model_'):
            self._embedding.model_.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        # --- Vectorized chunk extraction (Phase 2.1) ---
        if hook is not None:
            hook("embedding_chunk_extraction", None)
        # Pre-compute all audio chunks in one pass instead of individual crop() calls
        waveform = file["waveform"]  # (1, total_samples)
        sample_rate = file["sample_rate"]
        window_samples = int(duration * sample_rate)
        step_samples = int(binary_segmentations.sliding_window.step * sample_rate)

        # Unfold creates all chunks at once: (num_possible_chunks, window_samples)
        if waveform.shape[1] >= window_samples:
            all_chunks = waveform.unfold(1, window_samples, step_samples).squeeze(0)
        else:
            # Short audio: just pad
            all_chunks = F.pad(
                waveform, (0, window_samples - waveform.shape[1])
            ).squeeze(0).unsqueeze(0)

        # Pad if we have fewer chunks than segmentation expects
        if all_chunks.shape[0] < num_chunks:
            pad_chunks = num_chunks - all_chunks.shape[0]
            all_chunks = torch.cat([
                all_chunks,
                torch.zeros(pad_chunks, window_samples, dtype=all_chunks.dtype)
            ], dim=0)

        # Truncate if we have more
        all_chunks = all_chunks[:num_chunks]
        # Shape: (num_chunks, window_samples)

        # --- Vectorized mask selection (Phase 2.2) ---
        # Pre-compute which mask to use for all chunk x speaker pairs at once
        binary_data = np.nan_to_num(
            binary_segmentations.data, nan=0.0
        ).astype(np.float32)
        clean_data = np.nan_to_num(
            clean_segmentations.data, nan=0.0
        ).astype(np.float32)

        if exclude_overlap:
            # For each (chunk, speaker): use clean_mask if sum > min_num_frames
            # binary_data shape: (num_chunks, num_frames, num_speakers)
            # clean_data shape: (num_chunks, num_frames, num_speakers)
            clean_sums = np.sum(clean_data, axis=1)  # (num_chunks, num_speakers)
            use_clean = clean_sums > min_num_frames  # (num_chunks, num_speakers)
            # Transpose to (num_chunks, num_speakers, num_frames) for per-speaker selection
            binary_transposed = np.transpose(binary_data, (0, 2, 1))
            clean_transposed = np.transpose(clean_data, (0, 2, 1))
            final_masks = np.where(
                use_clean[:, :, np.newaxis],
                clean_transposed,
                binary_transposed,
            )
        else:
            # (num_chunks, num_speakers, num_frames)
            final_masks = np.transpose(binary_data, (0, 2, 1))

        # Reshape to flat list: (num_chunks * num_speakers, num_frames)
        flat_masks = final_masks.reshape(-1, final_masks.shape[-1])
        flat_masks_tensor = torch.from_numpy(flat_masks)
        # (num_chunks * num_speakers, num_frames)

        # Repeat chunks for each speaker: each chunk appears num_speakers times
        # (num_chunks * num_speakers, 1, window_samples)
        flat_waveforms = all_chunks.repeat_interleave(num_speakers, dim=0).unsqueeze(1)

        total_items = flat_waveforms.shape[0]

        # Auto-select embedding batch size based on GPU VRAM if not explicitly
        # set to a high value. Each item needs ~60MB VRAM for fbank+resnet.
        # Optimal throughput is at batch_size 64-128; above 256 returns diminish.
        # VRAM per batch: 32→1.6GB, 64→3.8GB, 128→7.6GB, 256→15.2GB
        effective_batch_size = self.embedding_batch_size
        if (
            effective_batch_size <= 32
            and hasattr(self._embedding, "device")
            and self._embedding.device.type == "cuda"
        ):
            try:
                free_vram_mb = (
                    torch.cuda.get_device_properties(self._embedding.device).total_mem
                    - torch.cuda.memory_reserved(self._embedding.device)
                ) / (1024 * 1024)
                # Use 40% of free VRAM, clamp to [64, 256] for best throughput
                auto_bs = max(64, min(256, int(free_vram_mb * 0.4 / 60)))
                # Round down to nearest power of 2 for GPU efficiency
                auto_bs = 2 ** int(math.log2(auto_bs))
                effective_batch_size = auto_bs
            except Exception:
                effective_batch_size = 128

        batch_count = math.ceil(total_items / effective_batch_size)

        embedding_batches = []

        if hook is not None:
            hook("embedding_inference_start", None)
            hook("embeddings", None, total=batch_count, completed=0)

        # --- Optimized embedding pipeline ---
        # Strategy: pre-compute fbank features for all chunks, then run resnet
        # in batches with double-buffered GPU prefetching.
        # This separates the FFT-heavy fbank (benefits from large batches) from
        # the CNN-heavy resnet (benefits from GPU pipelining).
        use_split_pipeline = (
            batch_count > 1
            and hasattr(self._embedding, "device")
            and self._embedding.device.type == "cuda"
            and hasattr(self._embedding, "model_")
            and hasattr(self._embedding.model_, "compute_fbank")
            and hasattr(self._embedding.model_, "resnet")
        )

        if use_split_pipeline:
            import warnings as _w
            device = self._embedding.device

            # Enable TF32 for Tensor Core acceleration on Ampere+ GPUs.
            # Must be set here because fix_reproducibility() disables it
            # during segmentation. TF32 is safe for inference and provides
            # ~15-20% speedup on Ampere+ (RTX 3000+, A-series, RTX 4000+).
            # On pre-Ampere GPUs, these flags have no effect.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Combined fbank + resnet pipeline with double-buffered prefetch
            # Compute fbank and resnet together per batch (keeps data on GPU),
            # while prefetching the next batch's waveforms via a separate stream.
            transfer_stream = torch.cuda.Stream(device=device)

            def prefetch_waveforms(idx):
                """Pin and transfer waveform batch to GPU on transfer_stream."""
                s = idx * effective_batch_size
                e = min(s + effective_batch_size, total_items)
                with torch.cuda.stream(transfer_stream):
                    wf_gpu = flat_waveforms[s:e].pin_memory().to(
                        device, non_blocking=True
                    )
                    mk_gpu = flat_masks_tensor[s:e].pin_memory().to(
                        device, non_blocking=True
                    )
                return wf_gpu, mk_gpu

            # Prefetch first batch
            next_wf, next_mk = prefetch_waveforms(0)

            for i in range(batch_count):
                # Wait for current batch transfer to complete
                torch.cuda.current_stream(device).wait_stream(
                    transfer_stream
                )
                cur_wf, cur_mk = next_wf, next_mk

                # Start prefetching next batch while GPU computes
                if i + 1 < batch_count:
                    next_wf, next_mk = prefetch_waveforms(i + 1)

                with torch.inference_mode():
                    with _w.catch_warnings():
                        _w.simplefilter("ignore")
                        # fbank + resnet in one shot, data stays on GPU
                        fbank = self._embedding.model_.compute_fbank(cur_wf)
                        # Interpolate masks to fbank frame-level
                        num_frames = fbank.shape[1]
                        imasks = F.interpolate(
                            cur_mk.unsqueeze(1).to(device),
                            size=num_frames, mode="nearest"
                        ).squeeze(1)
                        imasks = (imasks > 0.5).float()
                        _, emb_tensor = self._embedding.model_.resnet(
                            fbank, weights=imasks
                        )

                embedding_batch = emb_tensor.cpu().numpy()
                embedding_batches.append(embedding_batch)
                del fbank, imasks, cur_wf, cur_mk

                if hook is not None:
                    hook(
                        "embeddings", embedding_batch,
                        total=batch_count, completed=i + 1,
                    )
        else:
            # Fallback: simple sequential loop (CPU, ONNX, or single batch)
            for i in range(batch_count):
                start = i * effective_batch_size
                end = min(start + effective_batch_size, total_items)

                waveform_batch = flat_waveforms[start:end]
                mask_batch = flat_masks_tensor[start:end]

                embedding_batch: np.ndarray = self._embedding(
                    waveform_batch, masks=mask_batch
                )

                embedding_batches.append(embedding_batch)

                if hook is not None:
                    hook(
                        "embeddings", embedding_batch,
                        total=batch_count, completed=i + 1,
                    )

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

        # Mixed precision context for Ampere+ GPUs (~40% faster)
        from contextlib import nullcontext
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=torch.float16)
            if self._mixed_precision and self._segmentation.device.type == "cuda"
            else nullcontext()
        )

        with amp_ctx:
            segmentations = self.get_segmentations(file, hook=hook)
        hook("segmentation", segmentations)

        hook("vram_cleanup_post_segmentation", None)
        # Free GPU memory used by segmentation before embedding stage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        hook("binarization", binarized_segmentations)

        with amp_ctx:
            embeddings = self.get_embeddings(
                file,
                binarized_segmentations,
                exclude_overlap=self.embedding_exclude_overlap,
                hook=hook,
            )
        hook("embeddings", embeddings)

        # Phase 5: Release waveform tensor after embedding extraction (~1GB for 4.7h audio)
        if "waveform" in file and hasattr(file["waveform"], "storage"):
            del file["waveform"]

        hook("vram_cleanup_post_embedding", None)
        # Free GPU memory used by embeddings before clustering stage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        #   shape: (num_chunks, local_num_speakers, dimension)

        hook("clustering_start", None)
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

        hook("clustering_done", None)

        # convert to continuous diarization
        hook("reconstruction_start", None)
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
