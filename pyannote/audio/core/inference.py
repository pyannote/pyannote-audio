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

import math
import warnings
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Text, Union

import numpy as np
import torch
from einops import rearrange
from pytorch_lightning.utilities.memory import is_oom_error

from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Resolution
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.progress import InferenceProgressHook
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature

TaskName = Union[Text, None]


class Inference:
    """Inference

    TODO: add support for model averaging (either at output or weight level)

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
        Step between consecutive chunks, in seconds. Defaults to 10% of duration.
        Has no effect when `window` is "whole".
    batch_size : int, optional
        Batch size. Larger values make inference faster. Defaults to 32.
    device : torch.device, optional
        Device used for inference. Defaults to `model.device`.
        In case `device` and `model.device` are different, model is sent to device.
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

    # TODO: add option to automatically find maximum batch size

    def __init__(
        self,
        model: Union[Model, Text, Path],
        window: Text = "sliding",
        skip_aggregation: bool = False,
        device: torch.device = None,
        duration: float = None,
        step: float = None,
        batch_size: int = 32,
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

        for task_name, specifications in self.model.specifications.items():
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

        self.model.eval()
        self.model.to(self.device)

        # chunk duration used during training. for multi-task,
        # we assume that the same duration was used for each task.
        if self.model.is_multi_task:
            _, specifications = next(iter(self.model.specifications.items()))
        else:
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

        # step between consecutive chunks
        if step is None:
            step = 0.1 * self.duration
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

    def infer(self, chunks: torch.Tensor) -> Dict[TaskName, np.ndarray]:
        """Forward pass

        Parameters
        ----------
        chunks : torch.Tensor
            Batch of audio chunks.

        Returns
        -------
        outputs : {task_name: np.ndarray} dict
            Model outputs.

        Notes
        -----
        If model is mono-task, `task_name` is set to None.
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

        if self.model.is_multi_task:
            return {
                task_name: output.cpu().numpy() for task_name, output in outputs.items()
            }

        return {None: outputs.cpu().numpy()}

    def slide(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> Union[SlidingWindowFeature, Dict[Text, SlidingWindowFeature]]:
        """Slide model on a waveform

        Parameters
        ----------
        waveform: torch.Tensor
            (num_channels, num_samples) waveform.
        sample_rate : int
            Sample rate.

        Returns
        -------
        output : SlidingWindowFeature
            Model output. Shape is (num_chunks, dimension) for chunk-level tasks,
            and (num_frames, dimension) for frame-level tasks.

        Notes
        -----
        If model has several outputs (multi-task), those will be returned as a
        {task_name: output} dictionary.
        """

        # prepare sliding audio chunks
        num_channels, num_samples = waveform.shape
        file_duration = num_samples / sample_rate
        window_size: int = round(self.duration * sample_rate)

        results: Dict[Text, SlidingWindowFeature] = dict()

        # corner case: waveform is shorter than chunk duration
        if num_samples < window_size:

            warnings.warn(
                f"Waveform is shorter than requested sliding window ({self.duration}s): "
                f"this might lead to inconsistant results."
            )

            one_output = self.infer(waveform[None, :])

            for task_name, specifications in self.model.specifications.items():
                if specifications.resolution == Resolution.CHUNK:
                    frames = SlidingWindow(
                        start=0.0, duration=self.duration, step=self.step
                    )
                    results[task_name] = SlidingWindowFeature(
                        one_output[task_name], frames
                    )

                else:
                    _, num_frames, dimension = one_output[task_name].shape
                    frames = SlidingWindow(
                        start=0,
                        duration=file_duration / num_frames,
                        step=file_duration / num_frames,
                    )
                    results[task_name] = SlidingWindowFeature(
                        one_output[task_name][0], frames
                    )

            if self.model.is_multi_task:
                return results
            else:
                return results.popitem()[1]

        # prepare (and count) sliding audio chunks
        step_size: int = round(self.step * sample_rate)
        chunks: torch.Tensor = rearrange(
            waveform.unfold(1, window_size, step_size),
            "channel chunk frame -> chunk channel frame",
        )
        num_chunks, _, _ = chunks.shape

        # prepare last (right-aligned) audio chunk
        last_step_size = (num_samples - window_size) % step_size
        if last_step_size > 0:
            last_start = num_samples - window_size
            last_chunk: torch.Tensor = waveform[:, last_start:]
            has_last_chunk = True
        else:
            has_last_chunk = False

        outputs: Dict[TaskName, Union[List[np.ndarray], np.ndarray]] = {
            task_name: list() for task_name, _ in self.model.specifications.items()
        }

        if self.progress_hook is not None:
            self.progress_hook(0, num_chunks + has_last_chunk)

        # slide over audio chunks in batch
        for c in np.arange(0, num_chunks, self.batch_size):

            batch: torch.Tensor = chunks[c : c + self.batch_size]

            output = self.infer(batch)
            for task_name, task_output in output.items():
                outputs[task_name].append(task_output)

            if self.progress_hook is not None:
                self.progress_hook(c + 1, num_chunks + has_last_chunk)

        outputs = {
            task_name: np.vstack(task_outputs)
            for task_name, task_outputs in outputs.items()
        }

        for task_name, specifications in self.model.specifications.items():
            # skip aggregation when requested
            # or when model outputs just one vector per chunk
            if self.skip_aggregation or specifications.resolution == Resolution.CHUNK:
                frames = SlidingWindow(
                    start=0.0, duration=self.duration, step=self.step
                )
                results[task_name] = SlidingWindowFeature(outputs[task_name], frames)
                continue

            # process orphan last chunk
            if has_last_chunk:
                last_output = {
                    task_name: output[0]
                    for task_name, output in self.infer(last_chunk[None]).items()
                }
                if self.progress_hook is not None:
                    self.progress_hook(
                        num_chunks + has_last_chunk, num_chunks + has_last_chunk
                    )

            # use model introspection to estimate the total number of frames
            introspection = self.model.introspection[task_name]
            num_frames, dimension = introspection(num_samples)
            num_frames_per_chunk, _ = introspection(window_size)
            num_frames_per_step, _ = introspection(step_size)
            if has_last_chunk:
                num_frames_last_step, _ = introspection(last_step_size)

            # Hamming window used for overlap-add aggregation
            hamming = np.hamming(num_frames_per_chunk).reshape(-1, 1)

            # aggregated_output[i] will be used to store the (hamming-weighted) sum
            # of all predictions for frame #i
            aggregated_output: np.ndarray = np.zeros(
                (num_frames, dimension), dtype=np.float32
            )

            # overlapping_chunk_count[i] will be used to store the (hamming-weighted)
            # number of chunks that overlap with frame #i
            overlapping_chunk_count: np.ndarray = np.zeros(
                (num_frames, 1), dtype=np.float32
            )

            if specifications.permutation_invariant:
                # previous outputs that overlap with current output by at least 50%
                maxlen = max(1, math.floor(0.5 * self.duration / self.step))
                previous_outputs: Deque[np.ndarray] = deque([], maxlen=maxlen)

            # loop on the outputs of sliding chunks
            for c, output in enumerate(outputs[task_name]):
                start_sample = c * step_size
                start_frame, _ = introspection(start_sample)

                if specifications.permutation_invariant:
                    if c > 0:
                        output = self.permutate(
                            np.stack(previous_outputs), output, num_frames_per_step
                        )
                    previous_outputs.append(output)

                aggregated_output[start_frame : start_frame + num_frames_per_chunk] += (
                    output * hamming
                )

                overlapping_chunk_count[
                    start_frame : start_frame + num_frames_per_chunk
                ] += hamming

            # process last (right-aligned) chunk separately
            if has_last_chunk:

                if specifications.permutation_invariant and previous_outputs:
                    # FIXME
                    last_output[task_name] = self.permutate(
                        previous_outputs[-1][np.newaxis],
                        last_output[task_name],
                        num_frames_last_step,
                    )

                aggregated_output[-num_frames_per_chunk:] += (
                    last_output[task_name] * hamming
                )
                overlapping_chunk_count[-num_frames_per_chunk:] += hamming

            aggregated_output /= np.maximum(overlapping_chunk_count, 1e-12)

            frames = SlidingWindow(
                start=0,
                duration=file_duration / num_frames,
                step=file_duration / num_frames,
            )

            results[task_name] = SlidingWindowFeature(aggregated_output, frames)

        if self.model.is_multi_task:
            return results
        else:
            return results.popitem()[1]

    def permutate(
        self, past_outputs: Deque[np.ndarray], output: np.ndarray, step_size: int
    ) -> np.ndarray:
        """Find optimal permutation between past outputs and current output

        Parameters
        ----------
        past_outputs : deque of(num_frames, num_classes) np.ndarray
            Previous output
        output : (num_frames, num_classes) np.ndarray
            Current output
        step_size : int
            Step between previous and current outputs.
            Should be smaller than num_frames.

        Returns
        -------
        perm_output : (num_frames, num_classes) np.ndarray
            Permutated current output.
        """

        num_frames, num_classes = output.shape

        permutations = []
        for o, past_output in enumerate(reversed(past_outputs)):
            permutation = permutate(
                past_output[np.newaxis, (o + 1) * step_size :],
                output[: num_frames - (o + 1) * step_size],
            )[1][0]
            permutations.append(permutation)

        # TODO: track regions where more than one permutation is selected
        # as those regions should probably not be trusted too much
        # TODO: be even smarter and re-initialize tracking at those regions

        # choose most frequent permutation
        ((permutation, _),) = Counter(permutations).most_common(1)
        return output[:, permutation]

    def __call__(
        self, file: AudioFile
    ) -> Union[
        SlidingWindowFeature,
        Dict[Text, SlidingWindowFeature],
        np.ndarray,
        Dict[Text, np.ndarray],
    ]:
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

        Notes
        -----
        If model has several outputs (multi-task), those will be returned as a
        {task_name: output} dictionary.
        """

        waveform, sample_rate = self.model.audio(file)

        if self.window == "sliding":
            return self.slide(waveform, sample_rate)

        outputs = {
            task_name: task_output[0]
            for task_name, task_output in self.infer(waveform[None]).items()
        }
        if self.model.is_multi_task:
            return outputs
        else:
            return outputs.popitem()[1]

    def crop(
        self,
        file: AudioFile,
        chunk: Union[Segment, List[Segment]],
        fixed: Optional[float] = None,
    ) -> Union[
        SlidingWindowFeature,
        Dict[Text, SlidingWindowFeature],
        np.ndarray,
        Dict[Text, np.ndarray],
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
        If model has several outputs (multi-task), those will be returned as a
        {task_name: output} dictionary.
        """

        if self.window == "sliding":

            if not isinstance(chunk, Segment):
                start = min(c.start for c in chunk)
                end = max(c.end for c in chunk)
                chunk = Segment(start=start, end=end)

            waveform, sample_rate = self.model.audio.crop(file, chunk, fixed=fixed)
            output = self.slide(waveform, sample_rate)

            if self.model.is_multi_task:
                shifted_output = dict()
                for task_name, task_output in output.items():
                    frames = task_output.sliding_window
                    shifted_frames = SlidingWindow(
                        start=chunk.start, duration=frames.duration, step=frames.step
                    )
                    shifted_output[task_name] = SlidingWindowFeature(
                        task_output.data, shifted_frames
                    )
                return shifted_output
            else:
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

            outputs = {
                task_name: task_output[0]
                for task_name, task_output in self.infer(waveform[None]).items()
            }

            if self.model.is_multi_task:
                return outputs
            else:
                return outputs.popitem()[1]

        else:
            raise NotImplementedError(
                f"Unsupported window type '{self.window}': should be 'sliding' or 'whole'."
            )

    # TODO: add a way to process a stream (to allow for online processing)
