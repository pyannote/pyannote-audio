"""Sampling should be reproducible without replaying an epoch on resume."""
from types import SimpleNamespace

import lightning.pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from pyannote.audio.utils.random import create_rng_for_worker


@pytest.mark.parametrize('worker_id', [None, 0, 1])
def test_sampling_seed_includes_restored_progress(monkeypatch, worker_id):
    monkeypatch.setenv('PL_GLOBAL_SEED', '0')
    worker = None if worker_id is None else SimpleNamespace(id=worker_id)
    monkeypatch.setattr(torch.utils.data, 'get_worker_info', lambda: worker)
    model = SimpleNamespace(local_rank=0, global_rank=0, current_epoch=1, global_step=0)
    def samples():
        rng = create_rng_for_worker(model)
        return [rng.random() for _ in range(16)]
    opening = samples()
    model.global_step = 345000
    resumed = samples()
    assert resumed != opening
    assert samples() == resumed
    model.current_epoch += 1
    assert samples() != resumed


class SamplingDataset(IterableDataset):
    def __init__(self, model):
        self.model = model

    def __iter__(self):
        rng = create_rng_for_worker(self.model)
        while True:
            yield torch.tensor([rng.random() for _ in range(4)])


class SamplingModel(pl.LightningModule):
    def __init__(self, workers):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.workers = workers
        self.samples = []
        self.sample_epochs = []

    def train_dataloader(self):
        kwargs = {'multiprocessing_context': 'fork'} if self.workers else {}
        return DataLoader(SamplingDataset(self), batch_size=2,
                          num_workers=self.workers, **kwargs)

    def training_step(self, batch, batch_idx):
        self.samples.append(batch.detach().clone())
        self.sample_epochs.append(self.current_epoch)
        return self.weight * batch.mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=.01)


@pytest.mark.parametrize('workers', [0, 2])
def test_lightning_mid_epoch_resume_does_not_replay_opening_samples(tmp_path, monkeypatch, workers):
    if workers:
        import multiprocessing
        if 'fork' not in multiprocessing.get_all_start_methods():
            pytest.skip('pyannote training workers require fork')
    monkeypatch.setenv('PL_GLOBAL_SEED', '0')

    checkpoint_path = tmp_path / 'mid-epoch.ckpt'

    class SaveMidEpoch(pl.Callback):
        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            if trainer.global_step == 3:
                trainer.save_checkpoint(checkpoint_path)

    def run(steps, checkpoint=None):
        model = SamplingModel(workers)
        trainer = pl.Trainer(accelerator='cpu', devices=1, max_steps=steps,
                             max_epochs=-1, limit_train_batches=1000,
                             logger=False, enable_checkpointing=False,
                             enable_progress_bar=False, enable_model_summary=False,
                             default_root_dir=tmp_path,
                             callbacks=[SaveMidEpoch()] if checkpoint is None else [])
        trainer.fit(model, ckpt_path=checkpoint)
        return model, trainer

    initial, trainer = run(3)
    assert torch.load(checkpoint_path, weights_only=False)['epoch'] == 0
    resumed, resumed_trainer = run(5, checkpoint_path)
    repeated, _ = run(5, checkpoint_path)
    assert resumed_trainer.global_step == 5
    assert initial.sample_epochs == [0, 0, 0]
    assert resumed.sample_epochs == [0, 0]
    assert len(resumed.samples) == len(repeated.samples) == 2
    assert not torch.equal(initial.samples[0], resumed.samples[0])
    for a, b in zip(resumed.samples, repeated.samples):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
