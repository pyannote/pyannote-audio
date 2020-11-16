try:
    from IPython.display import Audio as IPythonAudio

    IPYTHON_INSTALLED = True
except ImportError:
    IPYTHON_INSTALLED = False

import warnings

from pyannote.audio.core.io import Audio, AudioFile
from pyannote.core import Segment


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
