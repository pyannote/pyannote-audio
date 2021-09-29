#!/usr/bin/env python
# encoding: utf-8

import base64
import io
import random
import warnings
from typing import Dict, Iterator, List, Text, Tuple

import librosa
import numpy as np
import scipy.io.wavfile
import soundfile as sf
import torch

# from librosa.util import valid_audio
# from librosa.util.exceptions import ParameterError
from soundfile import SoundFile

from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature

SAMPLE_RATE = 16000
Time = float


def normalizeT(waveform: torch.Tensor) -> torch.Tensor:
    """Normalize waveform for better display in Prodigy UI"""
    return waveform / (waveform.abs().max() + 1e-8)


def normalize(waveform: np.ndarray) -> np.ndarray:
    """Normalize waveform for better display in Prodigy UI"""
    return waveform / (np.max(np.abs(waveform)) + 1e-8)


def to_base64(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Text:
    """Convert waveform to base64 data"""
    with io.BytesIO() as content:
        scipy.io.wavfile.write(content, sample_rate, waveform)
        content.seek(0)
        b64 = base64.b64encode(content.read()).decode()
        b64 = f"data:audio/x-wav;base64,{b64}"
    return b64


def to_audio_spans(annotation: Annotation, focus: Segment = None) -> Dict:
    """Convert pyannote.core.Annotation to Prodigy's audio_spans
    Parameters
    ----------
    annotation : Annotation
        Annotation with t=0s time origin.
    focus : Segment, optional
        When provided, use its start time as audio_spans time origin.
    Returns
    -------
    audio_spans : list of dict
    """
    shift = 0.0 if focus is None else focus.start
    # label
    return [
        {"start": segment.start - shift, "end": segment.end - shift, "label": "Speech"}
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]


def remove_audio_before_db(examples: List[Dict]) -> List[Dict]:
    """Remove (potentially heavy) 'audio' key from examples
    Parameters
    ----------
    examples : list of dict
        Examples.
    Returns
    -------
    examples : list of dict
        Examples with 'audio' key removed.
    """
    for eg in examples:
        if "audio" in eg:
            del eg["audio"]

    return examples


def chunks(
    duration: float, chunk: float = 30, shuffle: bool = False
) -> Iterator[Segment]:
    """Partition [0, duration] time range into smaller chunks
    Parameters
    ----------
    duration : float
        Total duration, in seconds.
    chunk : float, optional
        Chunk duration, in seconds. Defaults to 30.
    shuffle : bool, optional
        Yield chunks in random order. Defaults to chronological order.
    Yields
    ------
    focus : Segment
    """

    sliding_window = SlidingWindow(start=0.0, step=chunk, duration=chunk)
    whole = Segment(0, duration)

    if shuffle:
        chunks_ = list(chunks(duration, chunk=chunk, shuffle=False))
        random.shuffle(chunks_)
        for chunk in chunks_:
            yield chunk

    else:
        for window in sliding_window(whole):
            yield window
        if window.end < duration:
            yield Segment(window.end, duration)


def time2index(
    constraints_time: List[Tuple[Time, Time]],
    window: SlidingWindow,
) -> List[Tuple[int, int]]:
    """Convert time-based constraints to index-based constraints
    Parameters
    ----------
    constraints_time : list of (float, float)
        Time-based constraints
    window : SlidingWindow
        Window used for embedding extraction
    Returns
    -------
    constraints : list of (int, int)
        Index-based constraints
    """

    constraints = []
    for t1, t2 in constraints_time:
        i1 = window.closest_frame(t1)
        i2 = window.closest_frame(t2)
        if i1 == i2:
            continue
        constraints.append((i1, i2))
    return constraints


def index2index(
    constraints: List[Tuple[int, int]],
    keep: np.ndarray,
    reverse=False,
    return_mapping=False,
) -> List[Tuple[int, int]]:
    """Map constraints from original to keep-only index base
    Parameters
    ----------
    constraints : list of pairs
        Constraints in original index base.
    keep : np.ndarray
        Boolean array indicating whether to keep observations.
    reverse : bool
        Set to True to go from keep-only to original index base.
    return_mapping : bool, optional
        Return mapping instead of mapped constraints.
    Returns
    -------
    shifted_constraints : list of index pairs
        Constraints in keep-only index base.
    """

    if reverse:
        mapping = np.arange(len(keep))[keep]
    else:
        mapping = np.cumsum(keep) - 1

    if return_mapping:
        return mapping

    return [
        (mapping[i1], mapping[i2]) for i1, i2 in constraints if keep[i1] and keep[i2]
    ]


# enlever
def get_audio_duration(current_file):
    """Return audio file duration
    Parameters
    ----------
    current_file : dict
        Dictionary given by pyannote.database.
    Returns
    -------
    duration : float
        Audio file duration.
    """

    with SoundFile(current_file["audio"], "r") as f:
        duration = float(f.frames) / f.samplerate

    return duration


# enlever
class RawAudio:
    """Raw audio with on-the-fly data augmentation
    Parameters
    ----------
    sample_rate: int, optional
        Target sampling rate. Defaults to using native sampling rate.
    mono : int, optional
        Convert multi-channel to mono. Defaults to True.
    augmentation : `pyannote.audio.augmentation.Augmentation`, optional
        Data augmentation.
    """

    def __init__(self, sample_rate=None, mono=True, augmentation=None):

        super().__init__()
        self.sample_rate = sample_rate
        self.mono = mono

        self.augmentation = augmentation

        if sample_rate is not None:
            self.sliding_window_ = SlidingWindow(
                start=-0.5 / sample_rate,
                duration=1.0 / sample_rate,
                step=1.0 / sample_rate,
            )

    @property
    def dimension(self):
        return 1

    @property
    def sliding_window(self):
        return self.sliding_window_

    def get_features(self, y, sample_rate):

        # convert to mono
        if self.mono:
            y = np.mean(y, axis=1, keepdims=True)

        # resample if sample rates mismatch
        if (self.sample_rate is not None) and (self.sample_rate != sample_rate):
            if y.shape[1] == 1:
                # librosa expects mono audio to be of shape (n,), but we have (n, 1).
                y = librosa.core.resample(y[:, 0], sample_rate, self.sample_rate)[
                    :, None
                ]
            else:
                y = librosa.core.resample(y.T, sample_rate, self.sample_rate).T
            sample_rate = self.sample_rate

        # augment data
        if self.augmentation is not None:
            y = self.augmentation(y, sample_rate)

        # TODO: how time consuming is this thing (needs profiling...)
        """
        try:
            valid = valid_audio(y[:, 0], mono=True)
        except ParameterError as e:
            msg = f"Something went wrong when augmenting waveform."
            raise ValueError(msg)

        return y
        """

    def __call__(self, current_file, return_sr=False):
        """Obtain waveform
        Parameters
        ----------
        current_file : dict
            `pyannote.database` files.
        return_sr : `bool`, optional
            Return sample rate. Defaults to False
        Returns
        -------
        waveform : `pyannote.core.SlidingWindowFeature`
            Waveform
        sample_rate : `int`
            Only when `return_sr` is set to True
        """

        if "waveform" in current_file:

            if self.sample_rate is None:
                msg = (
                    "`RawAudio` needs to be instantiated with an actual "
                    "`sample_rate` if one wants to use precomputed "
                    "waveform."
                )
                raise ValueError(msg)
            sample_rate = self.sample_rate

            y = current_file["waveform"]

            if len(y.shape) != 2:
                msg = (
                    "Precomputed waveform should be provided as a "
                    "(n_samples, n_channels) `np.ndarray`."
                )
                raise ValueError(msg)

        else:
            y, sample_rate = sf.read(
                current_file["audio"], dtype="float32", always_2d=True
            )

        # extract specific channel if requested
        channel = current_file.get("channel", None)
        if channel is not None:
            y = y[:, channel - 1 : channel]

        y = self.get_features(y, sample_rate)

        sliding_window = SlidingWindow(
            start=-0.5 / sample_rate, duration=1.0 / sample_rate, step=1.0 / sample_rate
        )

        if return_sr:
            return (
                SlidingWindowFeature(y, sliding_window),
                sample_rate if self.sample_rate is None else self.sample_rate,
            )

        return SlidingWindowFeature(y, sliding_window)

    def get_context_duration(self):
        return 0.0

    def crop(self, current_file, segment, mode="center", fixed=None):
        """Fast version of self(current_file).crop(segment, **kwargs)
        Parameters
        ----------
        current_file : dict
            `pyannote.database` file.
        segment : `pyannote.core.Segment`
            Segment from which to extract features.
        mode : {'loose', 'strict', 'center'}, optional
            In 'strict' mode, only frames fully included in 'segment' are
            returned. In 'loose' mode, any intersecting frames are returned. In
            'center' mode, first and last frames are chosen to be the ones
            whose centers are the closest to 'focus' start and end times.
            Defaults to 'center'.
        fixed : float, optional
            Overrides `Segment` 'focus' duration and ensures that the number of
            returned frames is fixed (which might otherwise not be the case
            because of rounding errors). Has no effect in 'strict' or 'loose'
            modes.
        Returns
        -------
        waveform : (n_samples, n_channels) numpy array
            Waveform
        See also
        --------
        `pyannote.core.SlidingWindowFeature.crop`
        """

        if self.sample_rate is None:
            msg = (
                "`RawAudio` needs to be instantiated with an actual "
                "`sample_rate` if one wants to use `crop`."
            )
            raise ValueError(msg)

        # find the start and end positions of the required segment
        ((start, end),) = self.sliding_window_.crop(
            segment, mode=mode, fixed=fixed, return_ranges=True
        )

        # this is expected number of samples.
        # this will be useful later in case of on-the-fly resampling
        # n_samples = end - start

        if "waveform" in current_file:

            y = current_file["waveform"]

            if len(y.shape) != 2:
                msg = (
                    "Precomputed waveform should be provided as a "
                    "(n_samples, n_channels) `np.ndarray`."
                )
                raise ValueError(msg)

            sample_rate = self.sample_rate
            data = y[start:end]

        else:
            # read file with SoundFile, which supports various fomats
            # including NIST sphere
            with SoundFile(current_file["audio"], "r") as audio_file:

                sample_rate = audio_file.samplerate

                # if the sample rates are mismatched,
                # recompute the start and end
                if sample_rate != self.sample_rate:

                    sliding_window = SlidingWindow(
                        start=-0.5 / sample_rate,
                        duration=1.0 / sample_rate,
                        step=1.0 / sample_rate,
                    )
                    ((start, end),) = sliding_window.crop(
                        segment, mode=mode, fixed=fixed, return_ranges=True
                    )

                try:
                    audio_file.seek(start)
                    data = audio_file.read(end - start, dtype="float32", always_2d=True)
                except RuntimeError as e:
                    msg = (
                        f"SoundFile failed to seek-and-read in "
                        f"{current_file['audio']}: loading the whole file..."
                    )
                    warnings.warn(msg)
                    warnings.warn(e)
                    return self(current_file).crop(segment, mode=mode, fixed=fixed)

        # extract specific channel if requested
        channel = current_file.get("channel", None)
        if channel is not None:
            data = data[:, channel - 1 : channel]

        return self.get_features(data, sample_rate)
