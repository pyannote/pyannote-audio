# MIT License
#
# Copyright (c) 2020 CNRS
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
from functools import cached_property
from typing import Dict, List, Optional, Text, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from pytorch_lightning.utilities.memory import is_oom_error

from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.model import Model, ModelIntrospection
from pyannote.audio.core.task import Scale, TaskSpecification
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
    """

    def __init__(
        self,
        model: Model,
        window: Text = "sliding",
        device: torch.device = None,
        duration: float = None,
        step: float = None,
        batch_size: int = 32,
        slide: bool = True,
    ):

        self.model = model

        if window not in ["sliding", "whole"]:
            raise ValueError('`window` must be "sliding" or "whole".')

        for task_name, task_specifications in self.task_specifications:
            scale = task_specifications.scale
            if scale == Scale.FRAME and window == "whole":
                warnings.warn(
                    'Using "whole" `window` inference with a frame-based model might lead to bad results '
                    'and huge memory consumption: it is recommended to set `window` to "sliding".'
                )

        self.window = window

        if device is None:
            device = self.model.device
        self.device = device

        self.model.eval()
        self.model.to(self.device)

        # chunk duration used during training. for multi-task,
        #  we assume that the same duration was used for each task.
        training_duration = self.task_specifications[0][1].duration

        if duration is None:
            duration = training_duration
        elif training_duration != duration:
            warnings.warn(
                f"Model was trained with {training_duration:g}s chunks, and you requested "
                f"{duration:g}s chunks for inference: this might lead to suboptimal results."
            )
        self.duration = duration

        #  step between consecutive chunks
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

    @cached_property
    def is_multi_task(self) -> bool:
        return self.model.is_multi_task

    @cached_property
    def task_specifications(self) -> List[Tuple[TaskName, TaskSpecification]]:
        return list(self.model.hparams.task_specifications.items())

    @cached_property
    def model_introspection(self) -> List[Tuple[TaskName, ModelIntrospection]]:
        return list(self.model.hparams.model_introspection.items())

    def infer(self, chunks: torch.Tensor) -> Dict[TaskName, np.ndarray]:
        """Forward pass

        Parameters
        ----------
        chunks : torch.Tensor
            Batch of audio chunks.

        Returns
        -------
        outputs : {task_name: torch.Tensor} dict
            Model outputs.

        Notes
        -----
        If model is mono-task, `task_name` is set to None.
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

        if self.is_multi_task:
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
        output : np.ndarray
            Shape is (num_chunks, dimension) for chunk-scaled tasks,
            and (num_frames, dimension) for frame-scaled tasks.
        frames : pyannote.core.SlidingWindow
        """

        # prepare sliding audio chunks
        num_channels, num_samples = waveform.shape
        file_duration = num_samples / sample_rate
        window_size: int = round(self.duration * sample_rate)

        results: Dict[Text, SlidingWindowFeature] = dict()

        # corner case: waveform is shorter than chunk duration
        if num_samples <= window_size:

            warnings.warn(
                f"Waveform is shorter than requested sliding window ({self.duration}s): "
                f"this might lead to inconsistant results."
            )

            one_output = self.infer(waveform[None, :])

            for task_name, task_specifications in self.task_specifications:
                if task_specifications.scale == Scale.CHUNK:
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

            if self.is_multi_task:
                return results
            return results[None]

        # prepare (and count) sliding audio chunks
        step_size: int = round(self.step * sample_rate)
        chunks: torch.Tensor = rearrange(
            waveform.unfold(1, window_size, step_size),
            "channel chunk frame -> chunk channel frame",
        )
        num_chunks, _, _ = chunks.shape

        # prepare last (right-aligned) audio chunk
        if (num_samples - window_size) % step_size > 0:
            last_start = num_samples - window_size
            last_chunk: torch.Tensor = waveform[:, last_start:]
            has_last_chunk = True
        else:
            has_last_chunk = False

        outputs: Dict[TaskName, Union[List[np.ndarray], np.ndarray]] = {
            task_name: list() for task_name, _ in self.task_specifications
        }

        # slide over audio chunks in batch
        for c in np.arange(0, num_chunks, self.batch_size):
            batch: torch.Tensor = chunks[c : c + self.batch_size]

            output = self.infer(batch)
            for task_name, task_output in output.items():
                outputs[task_name].append(task_output)

        outputs = {
            task_name: np.vstack(task_outputs)
            for task_name, task_outputs in outputs.items()
        }

        for t, (task_name, task_specifications) in enumerate(self.task_specifications):
            # if model outputs just one vector per chunk, return the outputs as they are
            if task_specifications.scale == Scale.CHUNK:
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

            #  use model introspection to estimate the total number of frames
            _, model_introspection = self.model_introspection[t]
            num_frames, dimension = model_introspection(num_samples)
            num_frames_per_chunk, _ = model_introspection(window_size)

            # aggregated_output[i] will be used to store the sum of all predictions for frame #i
            aggregated_output: np.ndarray = np.zeros(
                (num_frames, dimension), dtype=np.float32
            )

            # overlapping_chunk_count[i] will be used to store the number of chunks that
            # overlap with frame #i
            overlapping_chunk_count: np.ndarray = np.zeros(
                (num_frames, 1), dtype=np.int32
            )

            # loop on the outputs of sliding chunks
            for c, output in enumerate(outputs[task_name]):
                start_sample = c * step_size
                start_frame, _ = model_introspection(start_sample)
                aggregated_output[
                    start_frame : start_frame + num_frames_per_chunk
                ] += output
                overlapping_chunk_count[
                    start_frame : start_frame + num_frames_per_chunk
                ] += 1

            # process last (right-aligned) chunk separately
            if has_last_chunk:
                aggregated_output[-num_frames_per_chunk:] += last_output[task_name]
                overlapping_chunk_count[-num_frames_per_chunk:] += 1

            aggregated_output /= np.maximum(overlapping_chunk_count, 1)

            frames = SlidingWindow(
                start=0,
                duration=file_duration / num_frames,
                step=file_duration / num_frames,
            )

            results[task_name] = SlidingWindowFeature(aggregated_output, frames)

        if self.is_multi_task:
            return results
        return results[None]

    def __call__(self, file: AudioFile) -> Union[np.ndarray, SlidingWindowFeature]:
        """Run inference on a whole file

        Parameters
        ----------
        file : AudioFile
            Audio file.

        Returns
        -------
        output : np.ndarray
            Output.
        frames : SlidingWindow, optional
            Only returned for "sliding" window.
        """

        waveform, sample_rate = self.model.audio(file)
        # TODO remove this conversion if/when we switch to torchaudio IO
        waveform = torch.tensor(waveform, requires_grad=False)

        if self.window == "sliding":
            return self.slide(waveform, sample_rate)

        outputs = {
            task_name: task_output[0]
            for task_name, task_output in self.infer(waveform[None]).items()
        }
        if self.is_multi_task:
            return outputs
        return outputs[None]

    def crop(
        self,
        file: AudioFile,
        chunk: Segment,
        fixed: Optional[float] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, SlidingWindow]]:
        """Run inference on a chunk

        Parameters
        ----------
        file : AudioFile
            Audio file.
        chunk : pyannote.core.Segment
            Chunk.
        fixed : float, optional
            Enforce chunk duration (in seconds). This is a hack to avoid rounding
            errors that may result in a different number of audio samples for two
            chunks of the same duration.

        # TODO: document "fixed" better in pyannote.audio.core.io.Audio

        Returns
        -------
        output : np.ndarray
            Output.
        frames : SlidingWindow, optional
            Only returned for "sliding" window.
        """

        waveform, sample_rate = self.model.audio.crop(file, chunk, fixed=fixed)
        waveform = waveform.requires_grad_(False)

        if self.window == "sliding":
            output = self.slide(waveform, sample_rate)

            if self.is_multi_task:
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

        outputs = {
            task_name: task_output[0]
            for task_name, task_output in self.infer(waveform[None]).items()
        }
        if self.is_multi_task:
            return outputs
        return outputs[None]

    # TODO: add a way to process a stream (to allow for online processing)
