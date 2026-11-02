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
import zlib
from random import Random

import torch


def create_rng_for_worker(model) -> Random:
    """Create worker-specific random number generator

    This makes sure that
    1. training samples generation is reproducible
    2. every (worker, epoch) uses a different seed

    Parameters
    ----------
    epoch : int
        Current epoch.
    """

    # create random number generator
    rng = Random()

    global_seed = os.environ.get("PL_GLOBAL_SEED", "unset")
    worker_info = torch.utils.data.get_worker_info()

    if worker_info is None:
        worker_id = None
    else:
        worker_id = worker_info.id

    seed_tuple = (
        global_seed,
        worker_id,
        model.local_rank,
        model.global_rank,
        model.current_epoch,
    )
    # use adler32 because python's `hash` is not deterministic.
    seed = zlib.adler32(str(seed_tuple).encode())
    rng.seed(seed)

    return rng
