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


from typing import Union

try:
    from IPython.display import Audio as IPythonAudio
    from IPython.display import Video as IPythonVideo

    IPYTHON_INSTALLED = True
except ImportError:
    IPYTHON_INSTALLED = False

import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from moviepy.editor import AudioClip, VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage

    MOVIEPY_INSTALLED = True
except ImportError:
    MOVIEPY_INSTALLED = False


from typing import Mapping

import torch
from pyannote.core import (
    Annotation,
    Segment,
    SlidingWindow,
    SlidingWindowFeature,
    Timeline,
    notebook,
)

from pyannote.audio.core.io import Audio, AudioFile
from pyannote.audio.core.model import Model
from pyannote.audio.utils.signal import Binarize


def listen(audio_file: AudioFile, segment: Segment = None) -> None:
    """listen to audio

    Allows playing of audio files. It will play the whole thing unless
    given a `Segment` to crop to.

    Parameters
    ----------
    audio_file : AudioFile
        A str, Path or ProtocolFile to be loaded.
    segment : Segment, optional
        The segment to crop the playback to.
        Defaults to playback the whole file.
    """
    if not IPYTHON_INSTALLED:
        warnings.warn("You need IPython installed to use this method")
        return

    if segment is None:
        waveform, sr = Audio(mono="downmix")(audio_file)
    else:
        waveform, sr = Audio(mono="downmix").crop(audio_file, segment)
    return IPythonAudio(waveform.flatten(), rate=sr)


def preview(
    audio_file: AudioFile,
    segment: Segment = None,
    zoom: float = 10.0,
    video_fps: int = 5,
    video_ext: str = "webm",
    display: bool = True,
    **views,
):
    """Preview

    Parameters
    ----------
    audio_file : AudioFile
        A str, Path or ProtocolFile to be previewed
    segment : Segment, optional
        The segment to crop the preview to.
        Defaults to preview the whole file.
    video_fps : int, optional
        Video frame rate. Defaults to 5. Higher frame rate leads
        to a smoother video but longer processing time.
    video_ext : str, optional
        One of {"webm", "mp4", "ogv"} according to what your
        browser supports. Defaults to "webm" as it seems to
        be supported by most browsers (see caniuse.com/webm)/
    display : bool, optional
        Wrap the video in a IPython.display.Video instance for
        visualization in notebooks (default). Set to False if
        you are only interested in saving the video preview to
        disk.
    **views : dict
        Additional views. See Usage section below.

    Returns
    -------
    * IPython.display.Video instance if `display` is True (default)
    * path to video preview file if `display` is False

    Usage
    -----
    >>> assert isinstance(annotation, pyannote.core.Annotation)
    >>> assert isinstance(scores, pyannote.core.SlidingWindowFeature)
    >>> assert isinstance(timeline, pyannote.core.Timeline)
    >>> preview("audio.wav", reference=annotation, speech_probability=scores, speech_regions=timeline)
    # will create a video with 4 views. from to bottom:
    # "reference", "speech probability", "speech regions", and "waveform")

    """

    if not MOVIEPY_INSTALLED:
        warnings.warn("You need MoviePy installed to use this method")
        return

    if display and not IPYTHON_INSTALLED:
        warnings.warn(
            "Since IPython is not installed, this method cannot be used "
            "with default display=True option. Either run this method in "
            "a notebook or use display=False to save video preview to disk."
        )

    if isinstance(audio_file, Mapping) and "uri" in audio_file:
        uri = audio_file["uri"]
    elif isinstance(audio_file, (str, Path)):
        uri = Path(audio_file).name
    else:
        raise ValueError("Unsupported 'audio_file' type.")

    temp_dir = tempfile.mkdtemp(prefix="pyannote-audio-preview")
    video_path = f"{temp_dir}/{uri}.{video_ext}"

    audio = Audio(sample_rate=16000, mono="downmix")

    if segment is None:
        duration = audio.get_duration(audio_file)
        segment = Segment(start=0.0, end=duration)

    # load waveform as SlidingWindowFeautre
    data, sample_rate = audio.crop(audio_file, segment)
    data = data.numpy().T
    samples = SlidingWindow(
        start=segment.start, duration=1 / sample_rate, step=1 / sample_rate
    )
    waveform = SlidingWindowFeature(data, samples)
    ylim_waveform = np.min(data), np.max(data)

    def make_audio_frame(T: float):
        if isinstance(T, np.ndarray):
            return np.take(data, (T * sample_rate).astype(np.int64))
        return data[round(T * sample_rate)]

    audio_clip = AudioClip(make_audio_frame, duration=segment.duration, fps=sample_rate)

    # reset notebook just once so that colors are coherent between views
    notebook.reset()

    # initialize subplots with one row per view + one view for waveform
    nrows = len(views) + 1
    fig, axes = plt.subplots(
        nrows=nrows, ncols=1, figsize=(10, 2 * nrows), squeeze=False
    )

    *ax_views, ax_wav = axes[:, 0]

    # TODO: be smarter based on all SlidingWindowFeature views
    ylim = (-0.1, 1.1)

    def make_frame(T: float):
        # make sure all subsequent calls to notebook.plot_*
        # will only display the region center on current time
        t = T + segment.start

        notebook.crop = Segment(t - 0.5 * zoom, t + 0.5 * zoom)

        ax_wav.clear()
        notebook.plot_feature(waveform, ax=ax_wav, time=True, ylim=ylim_waveform)

        # display time cursor
        ax_wav.plot([t, t], ylim_waveform, "k--")

        # display view name
        ax_wav.axes.get_yaxis().set_visible(True)
        ax_wav.axes.get_yaxis().set_ticks([])
        ax_wav.set_ylabel("waveform")

        for (name, view), ax_view in zip(views.items(), ax_views):
            ax_view.clear()

            if isinstance(view, Timeline):
                notebook.plot_timeline(view, ax=ax_view, time=False)

            elif isinstance(view, Annotation):
                notebook.plot_annotation(view, ax=ax_view, time=False, legend=True)

            elif isinstance(view, SlidingWindowFeature):
                # TODO: be smarter about ylim
                notebook.plot_feature(view, ax=ax_view, time=False, ylim=ylim)

            # display time cursor
            ax_view.plot([t, t], ylim, "k--")

            # display view name
            ax_view.axes.get_yaxis().set_visible(True)
            ax_view.axes.get_yaxis().set_ticks([])
            ax_view.set_ylabel(" ".join(name.split("_")))

        return mplfig_to_npimage(fig)

    video_clip = VideoClip(make_frame, duration=segment.duration)
    video_clip = video_clip.set_audio(audio_clip)

    video_clip.write_videofile(
        video_path,
        fps=video_fps,
        audio=True,
        audio_fps=sample_rate,
        preset="ultrafast",
        logger="bar",
    )

    plt.close(fig)

    if not display:
        return video_path

    return IPythonVideo(video_path, embed=True)


def BROKEN_preview_training_samples(
    model: Model,
    blank: float = 1.0,
    video_fps: int = 5,
    video_ext: str = "webm",
    display: bool = True,
) -> Union[IPythonVideo, str]:
    """Preview training samples of a given model

    Parameters
    ----------
    Model : Model
        Model, already setup for training (i.e. call model.setup(stage="fit") first).
    blank : float, optional
        Add blank of that many seconds between each sample. Defaults to 1.0
    video_fps : int, optional
        Video frame rate. Defaults to 5. Higher frame rate leads
        to a smoother video but longer processing time.
    video_ext : str, optional
        One of {"webm", "mp4", "ogv"} according to what your
        browser supports. Defaults to "webm" as it seems to
        be supported by most browsers (see caniuse.com/webm)/
    display : bool, optional
        Wrap the video in a IPython.display.Video instance for
        visualization in notebooks (default). Set to False if
        you are only interested in saving the video preview to
        disk.

    Returns
    -------
    * IPython.display.Video instance if `display` is True (default)
    * path to video preview file if `display` is False
    """

    batch = next(iter(model.train_dataloader()))
    batch_size, num_channels, num_samples = batch["X"].shape
    batch_size, num_frames, num_speakers = batch["y"].shape
    sample_rate = model.audio.sample_rate

    batch_num_samples = int(batch_size * (num_samples + blank * sample_rate))
    batch_num_frames = int(model.introspection(batch_num_samples)[0])

    waveform = torch.zeros((1, batch_num_samples))
    reference = torch.zeros((batch_num_frames, num_speakers))
    for b, (X, y) in enumerate(zip(batch["X"], batch["y"])):
        X_idx = int(b * (num_samples + blank * sample_rate))
        waveform[:, X_idx : X_idx + num_samples] = X
        y_idx, _ = model.introspection(X_idx)
        reference[y_idx : y_idx + num_frames, :] = y

    reference = Binarize()(SlidingWindowFeature(reference, model.introspection.frames))

    audio_file = {
        "waveform": waveform,
        "sample_rate": sample_rate,
        "uri": "TrainingSamples",
    }

    return preview(
        audio_file,
        video_fps=video_fps,
        video_ext=video_ext,
        display=display,
        reference=reference,
    )
