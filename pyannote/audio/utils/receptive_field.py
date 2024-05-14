# MIT License
#
# Copyright (c) 2023 CNRS
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

from typing import List


def conv1d_num_frames(
    num_samples, kernel_size=5, stride=1, padding=0, dilation=1
) -> int:
    """Compute expected number of frames after 1D convolution

    Parameters
    ----------
    num_samples : int
        Number of samples in the input signal
    kernel_size : int
        Kernel size
    stride : int
        Stride
    padding : int
        Padding
    dilation : int
        Dilation

    Returns
    -------
    num_frames : int
        Number of frames in the output signal

    Source
    ------
    https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
    """
    return 1 + (num_samples + 2 * padding - dilation * (kernel_size - 1) - 1) // stride


def multi_conv_num_frames(
    num_samples: int,
    kernel_size: List[int] = None,
    stride: List[int] = None,
    padding: List[int] = None,
    dilation: List[int] = None,
) -> int:
    num_frames = num_samples
    for k, s, p, d in zip(kernel_size, stride, padding, dilation):
        num_frames = conv1d_num_frames(
            num_frames, kernel_size=k, stride=s, padding=p, dilation=d
        )

    return num_frames


def conv1d_receptive_field_size(
    num_frames=1, kernel_size=5, stride=1, padding=0, dilation=1
):
    """Compute size of receptive field

    Parameters
    ----------
    num_frames : int, optional
        Number of frames in the output signal
    kernel_size : int
        Kernel size
    stride : int
        Stride
    padding : int
        Padding
    dilation : int
        Dilation

    Returns
    -------
    size : int
        Receptive field size
    """

    effective_kernel_size = 1 + (kernel_size - 1) * dilation
    return effective_kernel_size + (num_frames - 1) * stride - 2 * padding


def multi_conv_receptive_field_size(
    num_frames: int,
    kernel_size: List[int] = None,
    stride: List[int] = None,
    padding: List[int] = None,
    dilation: List[int] = None,
) -> int:
    receptive_field_size = num_frames

    for k, s, p, d in reversed(list(zip(kernel_size, stride, padding, dilation))):
        receptive_field_size = conv1d_receptive_field_size(
            num_frames=receptive_field_size,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
        )
    return receptive_field_size


def conv1d_receptive_field_center(
    frame=0, kernel_size=5, stride=1, padding=0, dilation=1
) -> int:
    """Compute center of receptive field

    Parameters
    ----------
    frame : int
        Frame index
    kernel_size : int
        Kernel size
    stride : int
        Stride
    padding : int
        Padding
    dilation : int
        Dilation

    Returns
    -------
    center : int
        Index of receptive field center
    """

    effective_kernel_size = 1 + (kernel_size - 1) * dilation
    return frame * stride + (effective_kernel_size - 1) // 2 - padding


def multi_conv_receptive_field_center(
    frame: int,
    kernel_size: List[int] = None,
    stride: List[int] = None,
    padding: List[int] = None,
    dilation: List[int] = None,
) -> int:
    receptive_field_center = frame
    for k, s, p, d in reversed(list(zip(kernel_size, stride, padding, dilation))):
        receptive_field_center = conv1d_receptive_field_center(
            frame=receptive_field_center,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
        )

    return receptive_field_center
