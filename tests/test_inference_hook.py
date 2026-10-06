import pytest
import torch

from pyannote.audio import Inference, Model
from pyannote.audio.core.task import Problem, Resolution, Specifications


class DummySegmentationModel(Model):
    """Tiny frame-level model outputting one frame every 160 samples"""

    def __init__(self):
        super().__init__()
        self.specifications = Specifications(
            problem=Problem.BINARY_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=1.0,
            classes=["speech"],
        )

    def receptive_field_size(self, num_frames: int = 1) -> int:
        return 160 * num_frames

    def receptive_field_center(self, frame: int = 0) -> int:
        return 80 + 160 * frame

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_samples = waveforms.shape
        return torch.zeros(batch_size, num_samples // 160, 1)


@pytest.mark.parametrize(
    "duration, batch_size",
    [
        (3.0, 32),  # 5 complete chunks, single incomplete batch
        (3.05, 32),  # 5 complete chunks + 1 padded last chunk
        (3.05, 4),  # two batches of complete chunks + 1 padded last chunk
        (10.0, 4),  # 19 complete chunks, last batch is incomplete
    ],
)
def test_sliding_inference_hook_progress(duration, batch_size):
    inference = Inference(
        DummySegmentationModel(), duration=1.0, step=0.5, batch_size=batch_size
    )
    file = {"waveform": torch.zeros(1, int(duration * 16000)), "sample_rate": 16000}

    progress = []

    def hook(completed: int, total: int):
        progress.append((int(completed), int(total)))

    inference(file, hook=hook)

    completed = [c for c, _ in progress]
    total = progress[0][1]

    # total number of chunks does not change
    assert all(t == total for _, t in progress)
    # progress never goes past the total number of chunks...
    assert all(c <= total for c in completed)
    # ... never goes backward...
    assert completed == sorted(completed)
    # ... and starts at 0 and ends at total
    assert completed[0] == 0
    assert completed[-1] == total
