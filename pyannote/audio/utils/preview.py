try:
    from IPython.display import Audio as IPythonAudio

    IPYTHON_INSTALLED = True
except ImportError:
    IPYTHON_INSTALLED = False

import warnings

import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

from pyannote.audio.core.io import Audio, AudioFile
from pyannote.core import (
    Annotation,
    Segment,
    SlidingWindow,
    SlidingWindowFeature,
    Timeline,
    notebook,
)


def listen(audio_file: AudioFile, segment: Segment = None) -> None:
    """listen to audio

    Allows playing of audio files. It will play the whole thing unless
    given a `Segment` to crop to.


    Parameters
    ----------
    audio_file : AudioFile
        A str, Path or ProtocolFile to be loaded.
    segment : Segment, optional
        The segment to crop the playback too
    """
    if not IPYTHON_INSTALLED:
        warnings.warn("You need IPython installed to use this method")
        return

    if segment is None:
        waveform, sr = Audio()(audio_file)
    else:
        waveform, sr = Audio().crop(audio_file, segment)
    return IPythonAudio(waveform.flatten(), rate=sr)


def preview(
    audio_file: AudioFile, segment: Segment = None, zoom: float = 10.0, **views
):

    audio = Audio(sample_rate=16000, mono=True)

    if segment is None:
        duration = audio.get_duration(audio_file)
        segment = Segment(start=0.0, end=duration)

    # load waveform as SlidingWindowFeautre
    data, sample_rate = audio.crop(audio_file, segment)
    samples = SlidingWindow(
        start=segment.start, duration=1 / sample_rate, step=1 / sample_rate
    )

    waveform = SlidingWindowFeature(data.T, samples)

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

        notebook.plot_feature(waveform, ax=ax_wav, time=True, ylim=None)

        for (name, view), ax_view in zip(views.items(), ax_views):

            ax_view.clear()

            if isinstance(view, Timeline):
                notebook.plot_timeline(view, ax=ax_view, time=False)

            elif isinstance(view, Annotation):
                notebook.plot_annotation(view, ax=ax_view, time=False, legend=True)

            elif isinstance(view, SlidingWindowFeature):
                # TODO: be smarter about ylim
                notebook.plot_feature(view, ax=ax_view, time=True, ylim=ylim)

            # time cursor
            ax_view.plot([t, t], ylim, "k--")

            # ax_view.set_ylim(*ylim)

        return mplfig_to_npimage(fig)

    return VideoClip(make_frame, duration=segment.duration)
