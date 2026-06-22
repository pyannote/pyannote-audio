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
import logging
import math
import os
import textwrap
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Mapping, Optional, Text, Union, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.profiler import record_function

logger = logging.getLogger(__name__)
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
        # --- VRAM budget (Phase B) ---
        vram_budget_mb: Optional[int] = None,
        embedding_mixed_precision: bool = False,
    ):
        super().__init__()

        # Phase B: optional explicit VRAM budget in megabytes. When provided,
        # the embedding-stage batch sizer uses this instead of querying live
        # free VRAM (useful when coexisting with another model like Whisper).
        # None means "query device"; see _budget.recommend_embedding_batch.
        self.vram_budget_mb = vram_budget_mb
        # Separate from the existing ``mixed_precision`` (which wraps the
        # segmentation forward pass). This controls autocast around the
        # embedding forward pass. Phase A DER data shows enabling this
        # collapses speaker count on the measured corpus -- default False.
        self.embedding_mixed_precision = embedding_mixed_precision

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

        # torch.compile() for kernel fusion (10-30% faster after warmup).
        # Surface failures via the logger so silent fallbacks are visible;
        # historically this was `except: pass` which hid Dynamo graph breaks
        # that cost 10-20% of embedding-stage throughput without any signal.
        if torch_compile and self._segmentation.device.type in ("cuda", "mps"):
            try:
                if self.klustering != "OracleClustering":
                    # Attempted in Phase 6.3 follow-up (2026-04-23): compile
                    # ``resnet`` explicitly because the pipeline calls
                    # ``model_.resnet(...)`` directly. Result: E2E REGRESSED
                    # (170s mean with cv=35% on 2.2h, embedding stage 92s vs
                    # baseline 75s). Root cause: Dynamo recompiles on every
                    # unique batch shape the pipeline emits (same class of
                    # issue as the TRT EP rebuild storm). Without dynamic=True
                    # torch.compile punishes pipelines with variable shapes.
                    # Reverted to compiling only model_ (matches prior
                    # Phase 2.1/6.1 behavior — noise-level change, not a win
                    # but not a regression either).
                    self._embedding.model_ = torch.compile(
                        self._embedding.model_, mode="reduce-overhead"
                    )
                self._segmentation.model = torch.compile(
                    self._segmentation.model, fullgraph=False
                )
                logger.info(
                    "torch.compile enabled for segmentation + embedding on %s",
                    self._segmentation.device.type,
                )
            except Exception as exc:  # noqa: BLE001 (torch.compile raises bare Exception)
                logger.warning(
                    "torch.compile unavailable, falling back to eager execution: %s",
                    exc,
                )

        # cuDNN algorithm autotuning for fixed-shape segmentation/embedding
        # convolutions. Safe for inference; pyannote's fix_reproducibility()
        # does not touch this flag so one-time init is sufficient.
        if self._segmentation.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # ONNX CPU-only mode: export models to ONNX and replace forward passes
        if onnx_cpu:
            self._setup_onnx_cpu(onnx_quantize, onnx_num_threads)

        # Phase 6.2 ONNX Runtime mode (CUDA EP / CoreML EP / CPU EP — opt-in
        # via PYANNOTE_USE_ONNX=1 env var). Patches the segmentation infer()
        # path and the embedding resnet/fbank pair without touching the hot
        # call sites. See pyannote.audio.onnx for the runtime wrappers.
        self._onnx_seg_runtime = None
        self._onnx_emb_runtime = None
        try:
            from pyannote.audio.onnx import onnx_enabled
            if onnx_enabled():
                self._setup_phase6_onnx()
        except Exception as exc:  # pragma: no cover — never break pipeline init
            logger.warning("Phase 6.2 ONNX setup failed, falling back to eager: %s", exc)
            self._onnx_seg_runtime = None
            self._onnx_emb_runtime = None

    @property
    def segmentation_batch_size(self) -> int:
        return self._segmentation.batch_size

    @segmentation_batch_size.setter
    def segmentation_batch_size(self, batch_size: int):
        self._segmentation.batch_size = batch_size

    def _setup_phase6_onnx(self) -> None:
        """Wire Phase 6.2 ONNX Runtime into the segmentation + embedding paths.

        Triggered by ``PYANNOTE_USE_ONNX=1`` + ``PYANNOTE_ONNX_MODELS_DIR``
        env vars. Loads per-device ORT sessions (CUDA EP, CoreML EP, or CPU
        EP) and monkey-patches the inference call sites — the hot path in
        ``get_segmentations()`` / ``get_embeddings()`` stays unchanged.

        Provider order when ENABLE_TENSORRT=1 is set: TRT EP → CUDA EP → CPU
        EP. ORT falls back automatically if any provider fails to build.

        Falls back to eager PyTorch silently if artifacts are missing; the
        attribute references stay ``None`` and the hot path branches use
        PyTorch as before.
        """
        from pyannote.audio.onnx import (
            ONNXSegmentationRuntime,
            ONNXEmbeddingRuntime,
            compute_fbank_batched,
        )
        from pyannote.audio.onnx.runtime import _onnx_models_dir

        models_dir = _onnx_models_dir()
        if models_dir is None:
            logger.warning(
                "PYANNOTE_USE_ONNX=1 but PYANNOTE_ONNX_MODELS_DIR is unset or "
                "missing; ONNX runtime disabled."
            )
            return

        seg_path = models_dir / "segmentation.onnx"
        emb_path = models_dir / "embedding.onnx"
        device = self._segmentation.device

        # Phase 6.3 hybrid mode: allow skipping segmentation ONNX (keeps the
        # fast eager PyTorch path) while still routing embedding through
        # ORT+TensorRT. The segmentation ONNX is either a regression on
        # CUDA EP (LSTM/If fallback) or roughly-parity on TRT EP; until its
        # shape profile is tuned it is safer to leave it eager. Gate on
        # PYANNOTE_ONNX_SEG_ENABLED (default "1" to preserve prior behavior).
        seg_enabled = os.environ.get("PYANNOTE_ONNX_SEG_ENABLED", "1").strip().lower() in {
            "1", "true", "yes", "on"
        }
        emb_enabled = os.environ.get("PYANNOTE_ONNX_EMB_ENABLED", "1").strip().lower() in {
            "1", "true", "yes", "on"
        }

        # ---- segmentation ---------------------------------------------------
        if seg_enabled and seg_path.exists():
            try:
                self._onnx_seg_runtime = ONNXSegmentationRuntime(seg_path, device)
                logger.info(
                    "Phase 6.2 ONNX: segmentation via %s",
                    self._onnx_seg_runtime.providers[0],
                )
                orig_seg_runtime = self._onnx_seg_runtime

                def _onnx_seg_infer(chunks):
                    if not isinstance(chunks, torch.Tensor):
                        chunks = torch.as_tensor(chunks, dtype=torch.float32)
                    out = orig_seg_runtime(chunks)
                    return out.detach().cpu().numpy()

                self._segmentation.infer = _onnx_seg_infer
            except Exception as exc:
                logger.warning("ONNX segmentation runtime init failed: %s", exc)
                self._onnx_seg_runtime = None

        # ---- embedding ------------------------------------------------------
        if emb_enabled and emb_path.exists():
            try:
                self._onnx_emb_runtime = ONNXEmbeddingRuntime(emb_path, device)
                logger.info(
                    "Phase 6.2 ONNX: embedding via %s",
                    self._onnx_emb_runtime.providers[0],
                )

                # Monkey-patch the WeSpeakerResNet34 backbone: the pipeline
                # calls ``self._embedding.model_.resnet(fbank, weights=imasks)``
                # and unpacks (_, emb_tensor). We return a 2-tuple whose last
                # element is the ONNX embedding so the unpack still works.
                resnet = self._embedding.model_.resnet
                onnx_emb = self._onnx_emb_runtime

                def _onnx_resnet_forward(fbank, weights=None):
                    emb = onnx_emb(fbank, weights=weights)
                    return torch.zeros((), device=emb.device, dtype=emb.dtype), emb

                # Replace the bound forward with our callable. The wrapping
                # preserves __call__ → forward dispatch so the pipeline's
                # ``resnet(fbank, weights=imasks)`` syntax keeps working.
                resnet.forward = _onnx_resnet_forward

                # Replace compute_fbank with the batched helper (no vmap).
                wespeaker_model = self._embedding.model_
                fbank_fn = wespeaker_model._fbank
                centering_span = wespeaker_model.hparams.fbank_centering_span
                sample_rate = wespeaker_model.hparams.sample_rate
                frame_length_ms = wespeaker_model.hparams.frame_length
                frame_shift_ms = wespeaker_model.hparams.frame_shift

                def _batched_compute_fbank(waveforms):
                    return compute_fbank_batched(
                        waveforms, fbank_fn,
                        fbank_centering_span=centering_span,
                        sample_rate=sample_rate,
                        frame_length_ms=frame_length_ms,
                        frame_shift_ms=frame_shift_ms,
                    )

                wespeaker_model.compute_fbank = _batched_compute_fbank
            except Exception as exc:
                logger.warning("ONNX embedding runtime init failed: %s", exc)
                self._onnx_emb_runtime = None

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
        self._gpu_empty_cache()

    @staticmethod
    def _gpu_empty_cache():
        """Release cached GPU memory on CUDA or MPS devices."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

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

        with record_function("pyannote::segmentation"):
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

        # Total items = num_chunks * num_speakers. Instead of materializing
        # all combinations via repeat_interleave (which copies the full
        # waveform data — 59GB for 4.7h/21-speaker files), we index into
        # all_chunks on-the-fly per batch using: chunk_idx = item_idx // num_speakers
        total_items = num_chunks * num_speakers

        # Budget-aware embedding batch selection (Phase B, see _budget.py).
        # Phase A (2026-04-20) empirically showed embedding throughput
        # saturates at bs=16 for fp32; higher batches waste 3-7 GB of VRAM
        # for <3 % speed gain. The prior auto-scaler chose bs=64-256 based
        # on a scan of free VRAM and is removed here.
        from pyannote.audio.pipelines._budget import recommend_embedding_batch

        effective_batch_size = self.embedding_batch_size
        device_type = (
            self._embedding.device.type
            if hasattr(self._embedding, "device")
            else "cpu"
        )

        # Escape hatch preserved for the OpenTranscribe Phase A probe harness
        # and for reproducing the measurement matrix. Not advertised as user
        # API; the supported surface is the pipeline's vram_budget_mb kwarg
        # (see __init__) plus embedding_batch_size.
        _force = os.environ.get("PYANNOTE_FORCE_EMBEDDING_BATCH_SIZE")
        if _force and _force.isdigit() and int(_force) > 0:
            effective_batch_size = int(_force)
        elif effective_batch_size <= 32 and device_type in ("cuda", "mps"):
            # Only auto-select when caller left the default; >32 means the
            # caller is explicitly overriding and we must honor it.
            budget_mb = getattr(self, "vram_budget_mb", None)
            try:
                if budget_mb is None:
                    if device_type == "cuda":
                        budget_mb = (
                            torch.cuda.get_device_properties(
                                self._embedding.device
                            ).total_mem
                            - torch.cuda.memory_reserved(self._embedding.device)
                        ) / (1024 * 1024)
                    else:
                        # MPS: Apple Unified Memory — estimate a safe budget.
                        allocated = (
                            torch.mps.driver_allocated_memory() / (1024 * 1024)
                            if hasattr(torch.mps, "driver_allocated_memory")
                            else 0
                        )
                        if hasattr(torch.mps, "recommended_max_memory"):
                            total_mb = (
                                torch.mps.recommended_max_memory() / (1024 * 1024)
                            )
                        else:
                            try:
                                import os as _os

                                total_ram = _os.sysconf("SC_PHYS_PAGES") * _os.sysconf(
                                    "SC_PAGE_SIZE"
                                )
                                total_mb = total_ram * 0.75 / (1024 * 1024)
                            except (ValueError, OSError):
                                total_mb = 8192  # 8 GB safe fallback
                        budget_mb = max(0, total_mb - allocated)
                rec = recommend_embedding_batch(
                    free_mb=int(budget_mb), device=device_type
                )
                effective_batch_size = rec.batch_size
            except Exception:
                # Fall back to the old ceiling only if something upstream
                # broke the budget query — never above 16.
                effective_batch_size = 16

        batch_count = math.ceil(total_items / effective_batch_size)

        embedding_batches = []

        if hook is not None:
            hook("embedding_inference_start", None)
            hook("embeddings", None, total=batch_count, completed=0)

        # --- Optimized embedding pipeline ---
        # Strategy: call compute_fbank() + resnet() directly per batch,
        # bypassing the PyAnnote wrapper overhead. On CUDA, uses double-
        # buffered CUDA stream prefetch and TF32 acceleration. On MPS,
        # uses direct model calls with native MPS FFT (PyTorch 2.3+).
        use_split_pipeline = (
            batch_count > 1
            and hasattr(self._embedding, "device")
            and self._embedding.device.type in ("cuda", "mps")
            and hasattr(self._embedding, "model_")
            and hasattr(self._embedding.model_, "compute_fbank")
            and hasattr(self._embedding.model_, "resnet")
        )

        def _get_batch_waveforms(start: int, end: int) -> torch.Tensor:
            """Build waveform batch from all_chunks without materializing all combinations.

            Each flat index i corresponds to chunk i // num_speakers. We gather
            the relevant chunks and add a channel dimension.
            """
            chunk_indices = torch.arange(start, end) // num_speakers
            return all_chunks[chunk_indices].unsqueeze(1)

        if use_split_pipeline:
            import warnings as _w
            device = self._embedding.device
            is_cuda = device.type == "cuda"

            if is_cuda:
                # Enable TF32 for Tensor Core acceleration on Ampere+ GPUs.
                # Must be set here because fix_reproducibility() disables it
                # during segmentation. TF32 is safe for inference and provides
                # ~15-20% speedup on Ampere+ (RTX 3000+, A-series, RTX 4000+).
                # On pre-Ampere GPUs, these flags have no effect.
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

                # Double-buffered prefetch: transfer batch N+1 while GPU
                # processes batch N via a separate CUDA stream.
                transfer_stream = torch.cuda.Stream(device=device)

                def prefetch_waveforms(idx):
                    """Pin and transfer waveform batch to GPU on transfer_stream."""
                    s = idx * effective_batch_size
                    e = min(s + effective_batch_size, total_items)
                    with torch.cuda.stream(transfer_stream):
                        wf_gpu = _get_batch_waveforms(s, e).pin_memory().to(
                            device, non_blocking=True
                        )
                        mk_gpu = flat_masks_tensor[s:e].pin_memory().to(
                            device, non_blocking=True
                        )
                    return wf_gpu, mk_gpu

                # Prefetch first batch
                next_wf, next_mk = prefetch_waveforms(0)

            for i in range(batch_count):
                if is_cuda:
                    # Wait for current batch transfer to complete
                    torch.cuda.current_stream(device).wait_stream(
                        transfer_stream
                    )
                    cur_wf, cur_mk = next_wf, next_mk

                    # Start prefetching next batch while GPU computes
                    if i + 1 < batch_count:
                        next_wf, next_mk = prefetch_waveforms(i + 1)
                else:
                    # MPS path: simple .to(device) transfer (no pin_memory,
                    # no CUDA streams — MPS uses unified memory)
                    s = i * effective_batch_size
                    e = min(s + effective_batch_size, total_items)
                    cur_wf = _get_batch_waveforms(s, e).to(device)
                    cur_mk = flat_masks_tensor[s:e].to(device)

                # Phase 2.4: embedding_mixed_precision flag plumbed through.
                # Default False preserves byte-exact fp32 behavior. Phase A DER
                # measurements rejected fp16 (26-33% DER collapse in WeSpeaker
                # std() pooling). The flag exists so future Phase B/C work can
                # A/B test bf16 without re-plumbing; it is NOT a green-lit
                # pipeline option yet.
                amp_ctx = (
                    torch.autocast(device.type, dtype=torch.float16)
                    if self.embedding_mixed_precision and device.type in ("cuda", "mps")
                    else nullcontext()
                )
                with torch.inference_mode(), _w.catch_warnings(), amp_ctx:
                    _w.simplefilter("ignore")
                    # fbank + resnet in one shot, data stays on device
                    with record_function("pyannote::embedding_fbank"):
                        fbank = self._embedding.model_.compute_fbank(cur_wf)
                        # Interpolate masks to fbank frame-level
                        num_frames = fbank.shape[1]
                        imasks = F.interpolate(
                            cur_mk.unsqueeze(1).to(device),
                            size=num_frames, mode="nearest"
                        ).squeeze(1)
                        imasks = (imasks > 0.5).float()
                    with record_function("pyannote::embedding_resnet"):
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

                waveform_batch = _get_batch_waveforms(start, end)
                mask_batch = flat_masks_tensor[start:end]

                with record_function("pyannote::embedding_batch"):
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

        # Phase 5.2: opt-in GPU scatter-reduce path (PYANNOTE_GPU_RECONSTRUCT=1).
        # Falls back to the CPU loop silently on budget/availability failure.
        try:
            from pyannote.audio.gpu_ops import try_reconstruct_gpu
            gpu_result = try_reconstruct_gpu(
                segmentations.data, hard_clusters, num_clusters
            )
        except Exception:
            gpu_result = None

        if gpu_result is not None:
            return SlidingWindowFeature(gpu_result, segmentations.sliding_window)

        clustered_segmentations = np.nan * np.zeros(
            (num_chunks, num_frames, num_clusters)
        )

        # Per-chunk loop intentionally preserved. Phase 3.5 measured a naive
        # broadcast-max rewrite against this form on the 4.7h/8-speaker
        # benchmark and it was 57% slower (10.74s vs 6.85s) — numpy's
        # contiguous-slice ``segmentation[:, cluster == k]`` + axis=1 max
        # beats scattered broadcast ops on CPU by a wide margin. Phase 5.2
        # provides an opt-in GPU port (see gpu_ops.py); the naive CPU
        # vectorization remains an anti-pattern.
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

        # Mixed precision context for Ampere+ GPUs / Apple Silicon
        from contextlib import nullcontext
        seg_device_type = self._segmentation.device.type
        amp_ctx = (
            torch.amp.autocast(seg_device_type, dtype=torch.float16)
            if self._mixed_precision and seg_device_type in ("cuda", "mps")
            else nullcontext()
        )

        with amp_ctx:
            segmentations = self.get_segmentations(file, hook=hook)
        hook("segmentation", segmentations)

        hook("vram_cleanup_post_segmentation", None)
        # Free GPU memory used by segmentation before embedding stage
        self._gpu_empty_cache()

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
        self._gpu_empty_cache()

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
