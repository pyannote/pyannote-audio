# MIT License
#
# Copyright (c) 2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import random

import torch


def create_rng_for_worker():
    """Create worker-specific random number generator

    This makes sure that
    1. training samples generation is reproducible
    2. all workers use a different seed (e.g. to avoid training samples duplication)

    """

    # create random number generator
    rng = random.Random()

    #  create seed as a combination of PL_GLOBAL_SEED (set by pl.seed_everything())
    #  and other PL multi-processing variables
    global_seed = int(os.environ.get("PL_GLOBAL_SEED", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    num_gpus = len(os.environ.get("PL_TRAINER_GPUS", "0").split(","))

    worker_info = torch.utils.data.get_worker_info()

    if worker_info is None:
        num_workers = 1
        worker_id = 0
    else:
        num_workers = worker_info.num_workers
        worker_id = worker_info.id

    seed = (
        global_seed
        + worker_id
        + local_rank * num_workers
        + node_rank * num_workers * num_gpus
    )

    rng.seed(seed)

    return rng
