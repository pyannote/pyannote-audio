"""Regression test for the balancing path of SegmentationTask.train__iter__.

The values used for balancing are stored in ``prepared_data["metadata-values"]``
(a dict of lists, see ``Task.prepare_data``), but the balancing code read them
from ``prepared_data["metadata"]`` (a key that does not exist), raising
``KeyError: 'metadata'`` for any task using ``balance``.
"""

import types

import numpy as np
import pytest

from pyannote.audio.tasks.segmentation.mixins import SegmentationTask, Subsets


@pytest.fixture
def stub_task():
    task = object.__new__(SegmentationTask)
    task.balance = ["database"]
    task.duration = 2.0
    task.batch_size = 2
    task.num_chunks_per_file = 1
    task.model = types.SimpleNamespace(local_rank=0, global_rank=0, current_epoch=0)
    task.prepare_chunk = lambda file_id, start_time, duration: {
        "file_id": file_id,
        "start_time": start_time,
        "duration": duration,
    }

    n_files = 4
    n_regions_per_file = 3
    subset_idx = Subsets.index("train")

    regions = []
    regions_ids = []
    cursor = 0
    for i in range(n_files):
        start = cursor
        for _ in range(n_regions_per_file):
            regions.append((i, 10.0, float(cursor * 10)))
            cursor += 1
        regions_ids.append((start, cursor))

    task.prepared_data = {
        "audio-metadata": np.array(
            [(subset_idx, 0, i % 2) for i in range(n_files)],
            dtype=[("subset", "i"), ("scope", "i"), ("database", "i")],
        ),
        "audio-annotated": np.full(n_files, 30.0),
        "annotations-regions": np.array(
            regions, dtype=[("file_id", "i"), ("duration", "f"), ("start", "f")]
        ),
        "audio-regions-ids": np.array(
            regions_ids, dtype=[("start", "i"), ("end", "i")]
        ),
        "metadata-values": {
            "subset": [Subsets.index(s) for s in ("train", "development", "test")],
            "scope": [0, 1],
            "database": [0, 1],
        },
    }
    return task


def test_balance_uses_metadata_values(stub_task):
    """Balancing must read the per-key values from "metadata-values"."""
    chunks = [next(stub_task.train__iter__()) for _ in range(4)]
    assert len(chunks) == 4
    for chunk in chunks:
        assert set(chunk) == {"file_id", "start_time", "duration"}
