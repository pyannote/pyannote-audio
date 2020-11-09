import pytest
from IPython.display import Audio

from pyannote.audio.utils.preview import listen
from pyannote.core import Segment
from pyannote.database import FileFinder, get_protocol


def test_file():
    protocol = get_protocol(
        "Debug.SpeakerDiarization.Debug", preprocessors={"audio": FileFinder()}
    )
    return next(protocol.train())


def test_returns_audio_object():
    audio_file = test_file()
    ipython_audio = listen(audio_file)
    assert isinstance(ipython_audio, Audio)


def test_can_crop():
    audio_file = test_file()
    listen(audio_file, Segment(0, 1))


def test_fail_crop_too_large():
    with pytest.raises(ValueError):
        audio_file = test_file()
        duration = audio_file.duration
        listen(audio_file, Segment(0, duration * 2))
