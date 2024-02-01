import pytest
from pyannote.database import FileFinder, get_protocol

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import UnknownSpecificationsError
from pyannote.audio.tasks import SpeakerDiarization


@pytest.fixture()
def protocol():
    return get_protocol(
        "Debug.SpeakerDiarization.Debug", preprocessors={"audio": FileFinder()}
    )


def test_unknown_specifications_error_raised_on_non_setup_task(protocol):
    task = SpeakerDiarization(protocol=protocol)
    with pytest.raises(UnknownSpecificationsError):
        _ = task.specifications


def test_unknown_specifications_error_raised_on_non_setup_model_task(protocol):
    task = SpeakerDiarization(protocol=protocol)
    model = Model.from_pretrained("pyannote/ci-segmentation")
    model.task = task
    with pytest.raises(UnknownSpecificationsError):
        _ = model.specifications
