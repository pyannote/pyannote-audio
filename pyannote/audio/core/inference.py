# MIT License
#
# Copyright (c) 2020-2021 CNRS
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

import warnings
from pathlib import Path
from typing import Any, Callable, List, Optional, Text, Union

import numpy as np
import torch
from einops import rearrange
from pytorch_lightning.utilities.memory import is_oom_error

from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Resolution
from pyannote.audio.utils.progress import InferenceProgressHook
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature

TaskName = Union[Text, None]


class Inference:
    """Inference

    Parameters
    ----------
    model : Model
        Model. Will be automatically set to eval() mode and moved to `device` when provided.
    window : {"sliding", "whole"}, optional
        Use a "sliding" window and aggregate the corresponding outputs (default)
        or just one (potentially long) window covering the "whole" file or chunk.
    skip_aggregation : bool, optional
        Do not aggregate outputs when using "sliding" window. Defaults to False.
    duration : float, optional
        Chunk duration, in seconds. Defaults to duration used for training the model.
        Has no effect when `window` is "whole".
    step : float, optional
        Step between consecutive chunks, in seconds. Defaults to warm-up duration when
        greater than 0s, otherwise 10% of duration. Has no effect when `window` is "whole".
    batch_size : int, optional
        Batch size. Larger values make inference faster. Defaults to 32.
    device : torch.device, optional
        Device used for inference. Defaults to `model.device`.
        In case `device` and `model.device` are different, model is sent to device.
    pre_aggregation_hook : callable, optional
        When a callable is provided, it is applied to the model output, just before aggregation.
        Takes a (num_chunks, num_frames, dimension) numpy array as input and returns a modified
        (num_chunks, num_frames, other_dimension) numpy array passed to overlap-add aggregation.
    progress_hook : {callable, True, str}, optional
        When a callable is provided, it is called everytime a batch is processed
        with two integer arguments:
        - the number of chunks that have been processed so far
        - the total number of chunks
        Set to True (or a descriptive string) to display a tqdm progress bar.
    use_auth_token : str, optional
        When loading a private huggingface.co model, set `use_auth_token`
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running `huggingface-cli login`
    """

    def __init__(
        self,
        model: Union[Model, Text, Path],
        window: Text = "sliding",
        skip_aggregation: bool = False,
        device: torch.device = None,
        duration: float = None,
        step: float = None,
        batch_size: int = 32,
        pre_aggregation_hook: Callable[[np.ndarray], np.ndarray] = None,
        progress_hook: Union[bool, Text, Callable[[int, int], Any]] = False,
        use_auth_token: Union[Text, None] = None,
    ):

        self.model = (
            model
            if isinstance(model, Model)
            else Model.from_pretrained(
                Path(model),
                map_location=device,
                strict=False,
                use_auth_token=use_auth_token,
            )
        )

        if window not in ["sliding", "whole"]:
            raise ValueError('`window` must be "sliding" or "whole".')

        specifications = self.model.specifications
        if specifications.resolution == Resolution.FRAME and window == "whole":
            warnings.warn(
                'Using "whole" `window` inference with a frame-based model might lead to bad results '
                'and huge memory consumption: it is recommended to set `window` to "sliding".'
            )

        self.window = window
        self.skip_aggregation = skip_aggregation

        if device is None:
            device = self.model.device
        self.device = device

        self.pre_aggregation_hook = pre_aggregation_hook

        self.model.eval()
        self.model.to(self.device)

        # chunk duration used during training
        specifications = self.model.specifications
        training_duration = specifications.duration

        if duration is None:
            duration = training_duration
        elif training_duration != duration:
            warnings.warn(
                f"Model was trained with {training_duration:g}s chunks, and you requested "
                f"{duration:g}s chunks for inference: this might lead to suboptimal results."
            )
        self.duration = duration

        self.warm_up = specifications.warm_up
        # Use that many seconds on the left- and rightmost parts of each chunk
        # to warm up the model. While the model does process those left- and right-most
        # parts, only the remaining central part of each chunk is used for aggregating
        # scores during inference.

        # step between consecutive chunks
        if step is None:
            step = 0.1 * self.duration if self.warm_up[0] == 0.0 else self.warm_up[0]

        if step > self.duration:
            raise ValueError(
                f"Step between consecutive chunks is set to {step:g}s, while chunks are "
                f"only {self.duration:g}s long, leading to gaps between consecutive chunks. "
                f"Either decrease step or increase duration."
            )
        self.step = step

        self.batch_size = batch_size

        if callable(progress_hook):
            pass
        elif isinstance(progress_hook, Text):
            progress_hook = InferenceProgressHook(desc=progress_hook)
        elif progress_hook:
            progress_hook = InferenceProgressHook()
        else:
            progress_hook = None
        self.progress_hook = progress_hook

    def infer(self, chunks: torch.Tensor) -> np.ndarray:
        """Forward pass

        Takes care of sending chunks to right device and outputs back to CPU

        Parameters
        ----------
        chunks : (batch_size, num_channels, num_samples) torch.Tensor
            Batch of audio chunks.

        Returns
        -------
        outputs : (batch_size, ...) np.ndarray
            Model output.
        """

        with torch.no_grad():
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

        return outputs.cpu().numpy()

    def slide(self, waveform: torch.Tensor, sample_rate: int) -> SlidingWindowFeature:
        """Slide model on a waveform

        Parameters
        ----------
        waveform: (num_channels, num_samples) torch.Tensor
            Waveform.
        sample_rate : int
            Sample rate.

        Returns
        -------
        output : SlidingWindowFeature
            Model output. Shape is (num_chunks, dimension) for chunk-level tasks,
            and (num_frames, dimension) for frame-level tasks.
        """

        window_size: int = round(self.duration * sample_rate)
        step_size: int = round(self.step * sample_rate)
        num_channels, num_samples = waveform.shape

        specifications = self.model.specifications
        resolution = specifications.resolution
        introspection = self.model.introspection
        if resolution == Resolution.CHUNK:
            frames = SlidingWindow(start=0.0, duration=self.duration, step=self.step)
        elif resolution == Resolution.FRAME:
            frames = introspection.frames
            num_frames_per_chunk, dimension = introspection(window_size)

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
            last_chunk: torch.Tensor = waveform[:, num_chunks * step_size :]

        outputs: Union[List[np.ndarray], np.ndarray] = list()

        if self.progress_hook is not None:
            self.progress_hook(0, num_chunks + has_last_chunk)

        # slide over audio chunks in batch
        for c in np.arange(0, num_chunks, self.batch_size):
            batch: torch.Tensor = chunks[c : c + self.batch_size]
            outputs.append(self.infer(batch))
            if self.progress_hook is not None:
                self.progress_hook(c + 1, num_chunks + has_last_chunk)

        # process orphan last chunk
        if has_last_chunk:

            last_output = self.infer(last_chunk[None])

            if specifications.resolution == Resolution.FRAME:
                pad = num_frames_per_chunk - last_output.shape[1]
                last_output = np.pad(last_output, ((0, 0), (0, pad), (0, 0)))

            outputs.append(last_output)
            if self.progress_hook is not None:
                self.progress_hook(
                    num_chunks + has_last_chunk, num_chunks + has_last_chunk
                )

        outputs = np.vstack(outputs)

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
            frames = SlidingWindow(start=0.0, duration=self.duration, step=self.step)
            return SlidingWindowFeature(outputs, frames)

        if self.pre_aggregation_hook is not None:
            outputs = self.pre_aggregation_hook(outputs)
            _, _, dimension = outputs.shape

        # Hamming window used for overlap-add aggregation
        window = np.hamming(num_frames_per_chunk).reshape(-1, 1)

        # anything before warm_up_left (and after num_frames_per_chunk - warm_up_right)
        # will not be used in the final aggregation

        # warm-up windows used for overlap-add aggregation
        warm_up = np.ones((num_frames_per_chunk, 1))
        # anything before warm_up_left will not contribute to aggregation
        warm_up_left = round(self.warm_up[0] / self.duration * num_frames_per_chunk)
        warm_up[:warm_up_left] = 1e-12
        # anything after num_frames_per_chunk - warm_up_right either
        warm_up_right = round(self.warm_up[1] / self.duration * num_frames_per_chunk)
        warm_up[num_frames_per_chunk - warm_up_right :] = 1e-12

        # aggregated_output[i] will be used to store the sum of all predictions
        # for frame #i
        num_frames = frames.closest_frame(self.duration + num_chunks * self.step) + 1
        aggregated_output: np.ndarray = np.zeros(
            (num_frames, dimension), dtype=np.float32
        )

        # overlapping_chunk_count[i] will be used to store the number of chunks
        # that contributed to frame #i
        overlapping_chunk_count: np.ndarray = np.zeros(
            (num_frames, 1), dtype=np.float32
        )

        # loop on the outputs of sliding chunks
        for c, output in enumerate(outputs):
            start_frame = frames.closest_frame(c * self.step)
            aggregated_output[start_frame : start_frame + num_frames_per_chunk] += (
                output * window * warm_up
            )

            overlapping_chunk_count[
                start_frame : start_frame + num_frames_per_chunk
            ] += (window * warm_up)

        if has_last_chunk:
            aggregated_output = aggregated_output[: num_frames - pad, :]
            overlapping_chunk_count = overlapping_chunk_count[: num_frames - pad, :]

        return SlidingWindowFeature(
            aggregated_output / np.maximum(overlapping_chunk_count, 1e-12), frames
        )

    def __call__(self, file: AudioFile) -> Union[SlidingWindowFeature, np.ndarray]:
        """Run inference on a whole file

        Parameters
        ----------
        file : AudioFile
            Audio file.

        Returns
        -------
        output : SlidingWindowFeature or np.ndarray
            Model output, as `SlidingWindowFeature` if `window` is set to "sliding"
            and `np.ndarray` if is set to "whole".

        """

        waveform, sample_rate = self.model.audio(file)

        if self.window == "sliding":
            return self.slide(waveform, sample_rate)

        return self.infer(waveform[None])[0]

    def crop(
        self,
        file: AudioFile,
        chunk: Union[Segment, List[Segment]],
        fixed: Optional[float] = None,
    ) -> Union[SlidingWindowFeature, np.ndarray]:
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
        fixed : float, optional
            Enforce chunk duration (in seconds). This is a hack to avoid rounding
            errors that may result in a different number of audio samples for two
            chunks of the same duration.

        # TODO: document "fixed" better in pyannote.audio.core.io.Audio

        Returns
        -------
        output : SlidingWindowFeature or np.ndarray
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

        if self.window == "sliding":

            if not isinstance(chunk, Segment):
                start = min(c.start for c in chunk)
                end = max(c.end for c in chunk)
                chunk = Segment(start=start, end=end)

            waveform, sample_rate = self.model.audio.crop(file, chunk, fixed=fixed)
            output = self.slide(waveform, sample_rate)

            frames = output.sliding_window
            shifted_frames = SlidingWindow(
                start=chunk.start, duration=frames.duration, step=frames.step
            )
            return SlidingWindowFeature(output.data, shifted_frames)

        elif self.window == "whole":

            if isinstance(chunk, Segment):
                waveform, sample_rate = self.model.audio.crop(file, chunk, fixed=fixed)
            else:
                waveform = torch.cat(
                    [self.model.audio.crop(file, c)[0] for c in chunk], dim=1
                )

            return self.infer(waveform[None])[0]

        else:
            raise NotImplementedError(
                f"Unsupported window type '{self.window}': should be 'sliding' or 'whole'."
            )
