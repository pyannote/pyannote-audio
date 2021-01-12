# MIT License
#
# Copyright (c) 2021 CNRS
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


from typing import Text

from tqdm import tqdm


class InferenceProgressHook:
    """Default inference progress bar"""

    def __init__(self, desc: Text = None):
        self.desc = desc

    def __call__(self, chunk_idx, num_chunks):

        if chunk_idx == 0:
            self.pbar = tqdm(desc=self.desc, total=num_chunks, unit="chunks")
            self.chunk_idx = chunk_idx

        self.pbar.update(chunk_idx - self.chunk_idx)
        self.chunk_idx = chunk_idx
        if self.chunk_idx == num_chunks:
            self.pbar.close()
