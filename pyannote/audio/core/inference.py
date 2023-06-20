# MIT License
#
# Copyright (c) 2020- CNRS
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

import math
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Text, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature
from pytorch_lightning.utilities.memory import is_oom_error

from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.model import Model, Specifications
from pyannote.audio.core.task import Resolution
from pyannote.audio.utils.multi_task import map_with_specifications
from pyannote.audio.utils.permutation import mae_cost_func, permutate
from pyannote.audio.utils.powerset import Powerset
from pyannote.audio.utils.reproducibility import fix_reproducibility


class BaseInference:
    pass


class Inference(BaseInference):
    """Inference

    Parameters
    ----------
    model : Model
        Model. Will be automatically set to eval() mode and moved to `device` when provided.
    window : {"sliding", "whole"}, optional
        Use a "sliding" window and aggregate the corresponding outputs (default)
        or just one (potentially long) window covering the "whole" file or chunk.
    duration : float, optional
        Chunk duration, in seconds. Defaults to duration used for training the model.
        Has no effect when `window` is "whole".
    step : float, optional
        Step between consecutive chunks, in seconds. Defaults to warm-up duration when
        greater than 0s, otherwise 10% of duration. Has no effect when `window` is "whole".
    pre_aggregation_hook : callable, optional
        When a callable is provided, it is applied to the model output, just before aggregation.
        Takes a (num_chunks, num_frames, dimension) numpy array as input and returns a modified
        (num_chunks, num_frames, other_dimension) numpy array passed to overlap-add aggregation.
    skip_aggregation : bool, optional
        Do not aggregate outputs when using "sliding" window. Defaults to False.
    skip_conversion: bool, optional
        In case a task has been trained with `powerset` mode, output is automatically
        converted to `multi-label`, unless `skip_conversion` is set to True.
    batch_size : int, optional
        Batch size. Larger values (should) make inference faster. Defaults to 32.
    device : torch.device, optional
        Device used for inference. Defaults to `model.device`.
        In case `device` and `model.device` are different, model is sent to device.
    use_auth_token : str, optional
        When loading a private huggingface.co model, set `use_auth_token`
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running `huggingface-cli login`
    """

    def __init__(
        self,
        model: Union[Model, Text, Path],
        window: Text = "sliding",
        duration: float = None,
        step: float = None,
        pre_aggregation_hook: Callable[[np.ndarray], np.ndarray] = None,
        skip_aggregation: bool = False,
        skip_conversion: bool = False,
        device: torch.device = None,
        batch_size: int = 32,
        use_auth_token: Union[Text, None] = None,
    ):
        # ~~~~ model ~~~~~

        self.model = (
            model
            if isinstance(model, Model)
            else Model.from_pretrained(
                model,
                map_location=device,
                strict=False,
                use_auth_token=use_auth_token,
            )
        )

        if device is None:
            device = self.model.device
        self.device = device

        self.model.eval()
        self.model.to(self.device)

        specifications = self.model.specifications

        # ~~~~ sliding window ~~~~~

        if window not in ["sliding", "whole"]:
            raise ValueError('`window` must be "sliding" or "whole".')

        if window == "whole" and any(
            s.resolution == Resolution.FRAME for s in specifications
        ):
            warnings.warn(
                'Using "whole" `window` inference with a frame-based model might lead to bad results '
                'and huge memory consumption: it is recommended to set `window` to "sliding".'
            )
        self.window = window

        training_duration = next(iter(specifications)).duration
        duration = duration or training_duration
        if training_duration != duration:
            warnings.warn(
                f"Model was trained with {training_duration:g}s chunks, and you requested "
                f"{duration:g}s chunks for inference: this might lead to suboptimal results."
            )
        self.duration = duration

        # ~~~~ powerset to multilabel conversion ~~~~

        self.skip_conversion = skip_conversion

        conversion = list()
        for s in specifications:
            if s.powerset and not skip_conversion:
                c = Powerset(len(s.classes), s.powerset_max_classes)
            else:
                c = nn.Identity()
            conversion.append(c.to(self.device))

        if isinstance(specifications, Specifications):
            self.conversion = conversion[0]
        else:
            self.conversion = nn.ModuleList(conversion)

        # ~~~~ overlap-add aggregation ~~~~~

        self.skip_aggregation = skip_aggregation
        self.pre_aggregation_hook = pre_aggregation_hook

        self.warm_up = next(iter(specifications)).warm_up
        # Use that many seconds on the left- and rightmost parts of each chunk
        # to warm up the model. While the model does process those left- and right-most
        # parts, only the remaining central part of each chunk is used for aggregating
        # scores during inference.

        # step between consecutive chunks
        step = step or (
            0.1 * self.duration if self.warm_up[0] == 0.0 else self.warm_up[0]
        )

        if step > self.duration:
            raise ValueError(
                f"Step between consecutive chunks is set to {step:g}s, while chunks are "
                f"only {self.duration:g}s long, leading to gaps between consecutive chunks. "
                f"Either decrease step or increase duration."
            )
        self.step = step

        self.batch_size = batch_size

    def to(self, device: torch.device) -> "Inference":
        """Send internal model to `device`"""

        if not isinstance(device, torch.device):
            raise TypeError(
                f"`device` must be an instance of `torch.device`, got `{type(device).__name__}`"
            )

        self.model.to(device)
        self.conversion.to(device)
        self.device = device
        return self

    def infer(self, chunks: torch.Tensor) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Forward pass

        Takes care of sending chunks to right device and outputs back to CPU

        Parameters
        ----------
        chunks : (batch_size, num_channels, num_samples) torch.Tensor
            Batch of audio chunks.

        Returns
        -------
        outputs : (tuple of) (batch_size, ...) np.ndarray
            Model output.
        """

        with torch.inference_mode():
            try:
                outputs = self.model(chunks.to(self.device))
            except RuntimeError as exception:
                if is_oom_error(exception):
                    raise MemoryError(
                        f"batch_size ({self.batch_size: d}) is probably too large. "
                        f"Try with a smaller value until memory error disappears."
                    )
                else:
                    raise exception

        def __convert(output: torch.Tensor, conversion: nn.Module, **kwargs):
            return conversion(output).cpu().numpy()

        return map_with_specifications(
            self.model.specifications, __convert, outputs, self.conversion
        )

    def slide(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        hook: Optional[Callable],
    ) -> Union[SlidingWindowFeature, Tuple[SlidingWindowFeature]]:
        """Slide model on a waveform

        Parameters
        ----------
        waveform: (num_channels, num_samples) torch.Tensor
            Waveform.
        sample_rate : int
            Sample rate.
        hook: Optional[Callable]
            When a callable is provided, it is called everytime a batch is
            processed with two keyword arguments:
            - `completed`: the number of chunks that have been processed so far
            - `total`: the total number of chunks

        Returns
        -------
        output : (tuple of) SlidingWindowFeature
            Model output. Shape is (num_chunks, dimension) for chunk-level tasks,
            and (num_frames, dimension) for frame-level tasks.
        """

        window_size: int = self.model.audio.get_num_samples(self.duration)
        step_size: int = round(self.step * sample_rate)
        _, num_samples = waveform.shape

        def __frames(
            example_output, specifications: Optional[Specifications] = None
        ) -> SlidingWindow:
            if specifications.resolution == Resolution.CHUNK:
                return SlidingWindow(start=0.0, duration=self.duration, step=self.step)
            return example_output.frames

        frames: Union[SlidingWindow, Tuple[SlidingWindow]] = map_with_specifications(
            self.model.specifications,
            __frames,
            self.model.example_output,
        )

        # prepare complete chunks
        if num_samples >= window_size:
            chunks: torch.Tensor = rearrange(
                waveform.unfold(1, window_size, step_size),
                "channel chunk frame -> chunk channel frame",
            )
            num_chunks, _, _ = chunks.shape
        else:
            num_chunks = 0

        # prepare last incomplete chunk
        has_last_chunk = (num_samples < window_size) or (
            num_samples - window_size
        ) % step_size > 0
        if has_last_chunk:
            # pad last chunk with zeros
            last_chunk: torch.Tensor = waveform[:, num_chunks * step_size :]
            _, last_window_size = last_chunk.shape
            last_pad = window_size - last_window_size
            last_chunk = F.pad(last_chunk, (0, last_pad))

        def __empty_list(**kwargs):
            return list()

        outputs: Union[
            List[np.ndarray], Tuple[List[np.ndarray]]
        ] = map_with_specifications(self.model.specifications, __empty_list)

        if hook is not None:
            hook(completed=0, total=num_chunks + has_last_chunk)

        def __append_batch(output, batch_output, **kwargs) -> None:
            output.append(batch_output)
            return

        # slide over audio chunks in batch
        for c in np.arange(0, num_chunks, self.batch_size):
            batch: torch.Tensor = chunks[c : c + self.batch_size]

            batch_outputs: Union[np.ndarray, Tuple[np.ndarray]] = self.infer(batch)

            _ = map_with_specifications(
                self.model.specifications, __append_batch, outputs, batch_outputs
            )

            if hook is not None:
                hook(completed=c + self.batch_size, total=num_chunks + has_last_chunk)

        # process orphan last chunk
        if has_last_chunk:
            last_outputs = self.infer(last_chunk[None])

            _ = map_with_specifications(
                self.model.specifications, __append_batch, outputs, last_outputs
            )

            if hook is not None:
                hook(
                    completed=num_chunks + has_last_chunk,
                    total=num_chunks + has_last_chunk,
                )

        def __vstack(output: List[np.ndarray], **kwargs) -> np.ndarray:
            return np.vstack(output)

        outputs: Union[np.ndarray, Tuple[np.ndarray]] = map_with_specifications(
            self.model.specifications, __vstack, outputs
        )

        def __aggregate(
            outputs: np.ndarray,
            frames: SlidingWindow,
            specifications: Optional[Specifications] = None,
        ) -> SlidingWindowFeature:
            # skip aggregation when requested,
            # or when model outputs just one vector per chunk
            # or when model is permutation-invariant (and not post-processed)
            if (
                self.skip_aggregation
                or specifications.resolution == Resolution.CHUNK
                or (
                    specifications.permutation_invariant
                    and self.pre_aggregation_hook is None
                )
            ):
                frames = SlidingWindow(
                    start=0.0, duration=self.duration, step=self.step
                )
                return SlidingWindowFeature(outputs, frames)

            if self.pre_aggregation_hook is not None:
                outputs = self.pre_aggregation_hook(outputs)

            aggregated = self.aggregate(
                SlidingWindowFeature(
                    outputs,
                    SlidingWindow(start=0.0, duration=self.duration, step=self.step),
                ),
                frames=frames,
                warm_up=self.warm_up,
                hamming=True,
                missing=0.0,
            )

            # remove padding that was added to last chunk
            if has_last_chunk:
                aggregated.data = aggregated.crop(
                    Segment(0.0, num_samples / sample_rate), mode="loose"
                )

            return aggregated

        return map_with_specifications(
            self.model.specifications, __aggregate, outputs, frames
        )

    def __call__(
        self, file: AudioFile, hook: Optional[Callable] = None
    ) -> Union[
        Tuple[Union[SlidingWindowFeature, np.ndarray]],
        Union[SlidingWindowFeature, np.ndarray],
    ]:
        """Run inference on a whole file

        Parameters
        ----------
        file : AudioFile
            Audio file.
        hook : callable, optional
            When a callable is provided, it is called everytime a batch is processed
            with two keyword arguments:
            - `completed`: the number of chunks that have been processed so far
            - `total`: the total number of chunks

        Returns
        -------
        output : (tuple of) SlidingWindowFeature or np.ndarray
            Model output, as `SlidingWindowFeature` if `window` is set to "sliding"
            and `np.ndarray` if is set to "whole".

        """

        fix_reproducibility(self.device)

        waveform, sample_rate = self.model.audio(file)

        if self.window == "sliding":
            return self.slide(waveform, sample_rate, hook=hook)

        outputs: Union[np.ndarray, Tuple[np.ndarray]] = self.infer(waveform[None])

        def __first_sample(outputs: np.ndarray, **kwargs) -> np.ndarray:
            return outputs[0]

        return map_with_specifications(
            self.model.specifications, __first_sample, outputs
        )

    def crop(
        self,
        file: AudioFile,
        chunk: Union[Segment, List[Segment]],
        duration: Optional[float] = None,
        hook: Optional[Callable] = None,
    ) -> Union[
        Tuple[Union[SlidingWindowFeature, np.ndarray]],
        Union[SlidingWindowFeature, np.ndarray],
    ]:
        """Run inference on a chunk or a list of chunks

        Parameters
        ----------
        file : AudioFile
            Audio file.
        chunk : Segment or list of Segment
            Apply model on this chunk. When a list of chunks is provided and
            window is set to "sliding", this is equivalent to calling crop on
            the smallest chunk that contains all chunks. In case window is set
            to "whole", this is equivalent to concatenating each chunk into one
            (artifical) chunk before processing it.
        duration : float, optional
            Enforce chunk duration (in seconds). This is a hack to avoid rounding
            errors that may result in a different number of audio samples for two
            chunks of the same duration.
        hook : callable, optional
            When a callable is provided, it is called everytime a batch is processed
            with two keyword arguments:
            - `completed`: the number of chunks that have been processed so far
            - `total`: the total number of chunks

        Returns
        -------
        output : (tuple of) SlidingWindowFeature or np.ndarray
            Model output, as `SlidingWindowFeature` if `window` is set to "sliding"
            and `np.ndarray` if is set to "whole".

        Notes
        -----
        If model needs to be warmed up, remember to extend the requested chunk with the
        corresponding amount of time so that it is actually warmed up when processing the
        chunk of interest:
        >>> chunk_of_interest = Segment(10, 15)
        >>> extended_chunk = Segment(10 - warm_up, 15 + warm_up)
        >>> inference.crop(file, extended_chunk).crop(chunk_of_interest, returns_data=False)
        """

        fix_reproducibility(self.device)

        if self.window == "sliding":
            if not isinstance(chunk, Segment):
                start = min(c.start for c in chunk)
                end = max(c.end for c in chunk)
                chunk = Segment(start=start, end=end)

            waveform, sample_rate = self.model.audio.crop(
                file, chunk, duration=duration
            )
            outputs: Union[
                SlidingWindowFeature, Tuple[SlidingWindowFeature]
            ] = self.slide(waveform, sample_rate, hook=hook)

            def __shift(output: SlidingWindowFeature, **kwargs) -> SlidingWindowFeature:
                frames = output.sliding_window
                shifted_frames = SlidingWindow(
                    start=chunk.start, duration=frames.duration, step=frames.step
                )
                return SlidingWindowFeature(output.data, shifted_frames)

            return map_with_specifications(self.model.specifications, __shift, outputs)

        if isinstance(chunk, Segment):
            waveform, sample_rate = self.model.audio.crop(
                file, chunk, duration=duration
            )
        else:
            waveform = torch.cat(
                [self.model.audio.crop(file, c)[0] for c in chunk], dim=1
            )

        outputs: Union[np.ndarray, Tuple[np.ndarray]] = self.infer(waveform[None])

        def __first_sample(outputs: np.ndarray, **kwargs) -> np.ndarray:
            return outputs[0]

        return map_with_specifications(
            self.model.specifications, __first_sample, outputs
        )

    @staticmethod
    def aggregate(
        scores: SlidingWindowFeature,
        frames: SlidingWindow = None,
        warm_up: Tuple[float, float] = (0.0, 0.0),
        epsilon: float = 1e-12,
        hamming: bool = False,
        missing: float = np.NaN,
        skip_average: bool = False,
    ) -> SlidingWindowFeature:
        """Aggregation

        Parameters
        ----------
        scores : SlidingWindowFeature
            Raw (unaggregated) scores. Shape is (num_chunks, num_frames_per_chunk, num_classes).
        frames : SlidingWindow, optional
            Frames resolution. Defaults to estimate it automatically based on `scores` shape
            and chunk size. Providing the exact frame resolution (when known) leads to better
            temporal precision.
        warm_up : (float, float) tuple, optional
            Left/right warm up duration (in seconds).
        missing : float, optional
            Value used to replace missing (ie all NaNs) values.
        skip_average : bool, optional
            Skip final averaging step.

        Returns
        -------
        aggregated_scores : SlidingWindowFeature
            Aggregated scores. Shape is (num_frames, num_classes)
        """

        num_chunks, num_frames_per_chunk, num_classes = scores.data.shape

        chunks = scores.sliding_window
        if frames is None:
            duration = step = chunks.duration / num_frames_per_chunk
            frames = SlidingWindow(start=chunks.start, duration=duration, step=step)
        else:
            frames = SlidingWindow(
                start=chunks.start,
                duration=frames.duration,
                step=frames.step,
            )

        masks = 1 - np.isnan(scores)
        scores.data = np.nan_to_num(scores.data, copy=True, nan=0.0)

        # Hamming window used for overlap-add aggregation
        hamming_window = (
            np.hamming(num_frames_per_chunk).reshape(-1, 1)
            if hamming
            else np.ones((num_frames_per_chunk, 1))
        )

        # anything before warm_up_left (and after num_frames_per_chunk - warm_up_right)
        # will not be used in the final aggregation

        # warm-up windows used for overlap-add aggregation
        warm_up_window = np.ones((num_frames_per_chunk, 1))
        # anything before warm_up_left will not contribute to aggregation
        warm_up_left = round(
            warm_up[0] / scores.sliding_window.duration * num_frames_per_chunk
        )
        warm_up_window[:warm_up_left] = epsilon
        # anything after num_frames_per_chunk - warm_up_right either
        warm_up_right = round(
            warm_up[1] / scores.sliding_window.duration * num_frames_per_chunk
        )
        warm_up_window[num_frames_per_chunk - warm_up_right :] = epsilon

        # aggregated_output[i] will be used to store the sum of all predictions
        # for frame #i
        num_frames = (
            frames.closest_frame(
                scores.sliding_window.start
                + scores.sliding_window.duration
                + (num_chunks - 1) * scores.sliding_window.step
            )
            + 1
        )
        aggregated_output: np.ndarray = np.zeros(
            (num_frames, num_classes), dtype=np.float32
        )

        # overlapping_chunk_count[i] will be used to store the number of chunks
        # that contributed to frame #i
        overlapping_chunk_count: np.ndarray = np.zeros(
            (num_frames, num_classes), dtype=np.float32
        )

        # aggregated_mask[i] will be used to indicate whether
        # at least one non-NAN frame contributed to frame #i
        aggregated_mask: np.ndarray = np.zeros(
            (num_frames, num_classes), dtype=np.float32
        )

        # loop on the scores of sliding chunks
        for (chunk, score), (_, mask) in zip(scores, masks):
            # chunk ~ Segment
            # score ~ (num_frames_per_chunk, num_classes)-shaped np.ndarray
            # mask ~ (num_frames_per_chunk, num_classes)-shaped np.ndarray

            start_frame = frames.closest_frame(chunk.start)
            aggregated_output[start_frame : start_frame + num_frames_per_chunk] += (
                score * mask * hamming_window * warm_up_window
            )

            overlapping_chunk_count[
                start_frame : start_frame + num_frames_per_chunk
            ] += (mask * hamming_window * warm_up_window)

            aggregated_mask[
                start_frame : start_frame + num_frames_per_chunk
            ] = np.maximum(
                aggregated_mask[start_frame : start_frame + num_frames_per_chunk],
                mask,
            )

        if skip_average:
            average = aggregated_output
        else:
            average = aggregated_output / np.maximum(overlapping_chunk_count, epsilon)

        average[aggregated_mask == 0.0] = missing

        return SlidingWindowFeature(average, frames)

    @staticmethod
    def trim(
        scores: SlidingWindowFeature,
        warm_up: Tuple[float, float] = (0.1, 0.1),
    ) -> SlidingWindowFeature:
        """Trim left and right warm-up regions

        Parameters
        ----------
        scores : SlidingWindowFeature
            (num_chunks, num_frames, num_classes)-shaped scores.
        warm_up : (float, float) tuple
            Left/right warm up ratio of chunk duration.
            Defaults to (0.1, 0.1), i.e. 10% on both sides.

        Returns
        -------
        trimmed : SlidingWindowFeature
            (num_chunks, trimmed_num_frames, num_speakers)-shaped scores
        """

        assert (
            scores.data.ndim == 3
        ), "Inference.trim expects (num_chunks, num_frames, num_classes)-shaped `scores`"
        _, num_frames, _ = scores.data.shape

        chunks = scores.sliding_window

        num_frames_left = round(num_frames * warm_up[0])
        num_frames_right = round(num_frames * warm_up[1])

        num_frames_step = round(num_frames * chunks.step / chunks.duration)
        if num_frames - num_frames_left - num_frames_right < num_frames_step:
            warnings.warn(
                f"Total `warm_up` is so large ({sum(warm_up) * 100:g}% of each chunk) "
                f"that resulting trimmed scores does not cover a whole step ({chunks.step:g}s)"
            )
        new_data = scores.data[:, num_frames_left : num_frames - num_frames_right]

        new_chunks = SlidingWindow(
            start=chunks.start + warm_up[0] * chunks.duration,
            step=chunks.step,
            duration=(1 - warm_up[0] - warm_up[1]) * chunks.duration,
        )

        return SlidingWindowFeature(new_data, new_chunks)

    @staticmethod
    def stitch(
        activations: SlidingWindowFeature,
        frames: SlidingWindow = None,
        lookahead: Optional[Tuple[int, int]] = None,
        cost_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        match_func: Callable[[np.ndarray, np.ndarray, float], bool] = None,
    ) -> SlidingWindowFeature:
        """

        Parameters
        ----------
        activations : SlidingWindowFeature
            (num_chunks, num_frames, num_classes)-shaped scores.
        frames : SlidingWindow, optional
            Frames resolution. Defaults to estimate it automatically based on `activations`
            shape and chunk size. Providing the exact frame resolution (when known) leads to better
            temporal precision.
        lookahead : (int, int) tuple
            Number of past and future adjacent chunks to use for stitching.
            Defaults to (k, k) with k = chunk_duration / chunk_step - 1
        cost_func : callable
            Cost function used to find the optimal mapping between two chunks.
            Expects two (num_frames, num_classes) torch.tensor as input
            and returns cost as a (num_classes, ) torch.tensor
            Defaults to mean absolute error (utils.permutations.mae_cost_func)
        match_func : callable
            Function used to decide whether two speakers mapped by the optimal
            mapping actually are a match.
            Expects two (num_frames, ) np.ndarray and the cost (from cost_func)
            and returns a boolean. Defaults to always returning True.
        """

        num_chunks, num_frames, num_classes = activations.data.shape

        chunks: SlidingWindow = activations.sliding_window

        if frames is None:
            duration = step = chunks.duration / num_frames
            frames = SlidingWindow(start=chunks.start, duration=duration, step=step)
        else:
            frames = SlidingWindow(
                start=chunks.start,
                duration=frames.duration,
                step=frames.step,
            )

        max_lookahead = math.floor(chunks.duration / chunks.step - 1)
        if lookahead is None:
            lookahead = 2 * (max_lookahead,)

        assert all(L <= max_lookahead for L in lookahead)

        if cost_func is None:
            cost_func = mae_cost_func

        if match_func is None:

            def always_match(this: np.ndarray, that: np.ndarray, cost: float):
                return True

            match_func = always_match

        stitches = []
        for C, (chunk, activation) in enumerate(activations):
            local_stitch = np.NAN * np.zeros(
                (sum(lookahead) + 1, num_frames, num_classes)
            )

            for c in range(
                max(0, C - lookahead[0]), min(num_chunks, C + lookahead[1] + 1)
            ):
                # extract common temporal support
                shift = round((C - c) * num_frames * chunks.step / chunks.duration)

                if shift < 0:
                    shift = -shift
                    this_activations = activation[shift:]
                    that_activations = activations[c, : num_frames - shift]
                else:
                    this_activations = activation[: num_frames - shift]
                    that_activations = activations[c, shift:]

                # find the optimal one-to-one mapping
                _, (permutation,), (cost,) = permutate(
                    this_activations[np.newaxis],
                    that_activations,
                    cost_func=cost_func,
                    return_cost=True,
                )

                for this, that in enumerate(permutation):
                    # only stitch under certain condiditions
                    matching = (c == C) or (
                        match_func(
                            this_activations[:, this],
                            that_activations[:, that],
                            cost[this, that],
                        )
                    )

                    if matching:
                        local_stitch[c - C + lookahead[0], :, this] = activations[
                            c, :, that
                        ]

                    # TODO: do not lookahead further once a mismatch is found

            stitched_chunks = SlidingWindow(
                start=chunk.start - lookahead[0] * chunks.step,
                duration=chunks.duration,
                step=chunks.step,
            )

            local_stitch = Inference.aggregate(
                SlidingWindowFeature(local_stitch, stitched_chunks),
                frames=frames,
                hamming=True,
            )

            stitches.append(local_stitch.data)

        stitches = np.stack(stitches)
        stitched_chunks = SlidingWindow(
            start=chunks.start - lookahead[0] * chunks.step,
            duration=chunks.duration + sum(lookahead) * chunks.step,
            step=chunks.step,
        )

        return SlidingWindowFeature(stitches, stitched_chunks)
