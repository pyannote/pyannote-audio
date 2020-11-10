try:
    from IPython.display import Audio as IPythonAudio

    IPYTHON_INSTALLED = True
except ImportError:
    IPYTHON_INSTALLED = False

import warnings
from typing import Union

from torch import Tensor

from pyannote.audio.core.io import Audio, AudioFile
from pyannote.core import Segment


def listen(
    audio_file: Union[Tensor, AudioFile], segment: Segment = None, sr=16000
) -> None:
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

    if isinstance(audio_file, Tensor):
        audio_file = {
            "waveform": audio_file,
            "sample_rate": sr,
        }
    if segment is None:
        waveform, sr = Audio()(audio_file)
    else:
        waveform, sr = Audio().crop(audio_file, segment)
    return IPythonAudio(waveform.flatten(), rate=sr)
