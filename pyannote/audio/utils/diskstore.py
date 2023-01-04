# MIT License
#
# Copyright (c) 2020-2021 CNRS
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
import binascii
import numpy as np


def generate_random_string(length=6):
    """
    Returns a random string of a specified length.
    >>> len(generate_random_string(length=25))
    25
    Test randomness. Try N times and observe no duplicaton
    >>> N = 100
    >>> len(set(generate_random_string(10) for i in range(N))) == N
    True
    """
    n = int(length / 2 + 1)
    x = binascii.hexlify(os.urandom(n))
    s = x[:length]
    return s.decode("utf-8")


class DiskArray:
    """
    Numpy array stored on the disk using memmap
    """

    def __init__(self, name, path, shape, dtype):
        self.name = name
        self.path = path
        self.shape = shape
        self.dtype = dtype
        self._detach_file = False

        self.data = np.memmap(path, mode="w+", shape=shape, dtype=dtype)

    def detach_file(self):
        """
        Detach the file from this object so that it
        is not removed from the disk when this object
        is destroyed (when garbage collected)
        """
        self._detach_file = True

    def __del__(self):
        del self.data
        if not self._detach_file:
            os.remove(self.path)


class DiskList(DiskArray):
    """
    Disk based list of numpy arrays abstraction
    designed specifically for storing outputs
    of inferencing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos = 0

    def append(self, batch):
        bx = batch.shape[0]
        new_pos = self.pos + bx

        if len(batch.shape) == 3:
            by, bz = batch.shape[1:]
            self.data[self.pos : new_pos, :by, :bz] = batch
        elif len(batch.shape) == 2:
            by = batch.shape[-1]
            self.data[self.pos : new_pos, :by] = batch
        else:
            raise Exception

        self.pos = new_pos


class DiskStore:
    """
    Abstraction to manage the storage of one or more
    numpy arrays on disk using memmap.
    """

    def __init__(self, path):
        self.path = path
        assert os.path.exists(self.path)

        self.stores = {}

    def _get_fpath(self, name):
        rnd_name = generate_random_string(length=10)
        fpath = os.path.join(self.path, "%s_%s" % (name, rnd_name))
        return fpath

    def get_array(self, name, shape, dtype):
        fpath = self._get_fpath(name)

        _store = self.stores.get("name", None)
        if _store:
            if _store.shape != shape or _store.dtype != dtype:
                raise Exception("DiskArray already exists with different options")

            return _store

        _store = DiskArray(name, fpath, shape, dtype)
        self.stores[name] = _store
        return _store

    def get_list(self, name, shape, dtype):
        fpath = self._get_fpath(name)

        _store = self.stores.get("name", None)
        if _store:
            if _store.shape != shape or _store.dtype != dtype:
                raise Exception("DiskList already exists with different options")

            return _store

        _store = DiskList(name, fpath, shape, dtype)
        self.stores[name] = _store
        return _store

    def cleanup(self):
        for name, data in list(self.stores.items()):
            del self.stores[name]
            del data

    def __del__(self):
        self.cleanup()
