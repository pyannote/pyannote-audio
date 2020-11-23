import pytorch_lightning as pl
import torch

from pyannote.audio.models.debug import SimpleSegmentationModel
from pyannote.audio.tasks import SpeakerTracking, VoiceActivityDetection
from pyannote.database import FileFinder, get_protocol


def setup_tasks(task):
    protocol = get_protocol(
        "Debug.SpeakerDiarization.Debug", preprocessors={"audio": FileFinder()}
    )
    vad = task(protocol, duration=0.2, batch_size=32, num_workers=4)
    return protocol, vad


def create_dl(model, task):
    m = model(task=task)
    task.setup("fit")
    m.setup("fit")
    return task.train_dataloader()


def get_next5(dl):
    last5 = []
    it = iter(dl)
    for i in range(5):
        last5.append(next(it))
    return last5


def test_seeding_ensures_data_loaders():
    "Setting a global seed for the dataloaders ensures that we get data back in the same order"
    pl.seed_everything(1)

    for task in [VoiceActivityDetection, SpeakerTracking]:
        protocol, vad = setup_tasks(task)
        dl = create_dl(SimpleSegmentationModel, vad)
        last5a = get_next5(dl)

        protocol, vad = setup_tasks(task)
        dl = create_dl(SimpleSegmentationModel, vad)
        last5b = get_next5(dl)

        for i in range(len(last5b)):
            assert torch.equal(last5a[i]["X"], last5b[i]["X"])


def test_different_seeds():
    "Changing the global seed will change the order of the data that loads"

    for task in [VoiceActivityDetection, SpeakerTracking]:
        protocol, vad = setup_tasks(task)
        pl.seed_everything(4)
        dl = create_dl(SimpleSegmentationModel, vad)
        last5a = get_next5(dl)

        protocol, vad = setup_tasks(task)
        pl.seed_everything(5)
        dl = create_dl(SimpleSegmentationModel, vad)
        last5b = get_next5(dl)

        for i in range(5):
            assert not torch.equal(last5a[i]["X"], last5b[i]["X"])
