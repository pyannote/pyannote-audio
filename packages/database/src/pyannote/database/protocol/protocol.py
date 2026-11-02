# The MIT License (MIT)

# Copyright (c) 2016-2020 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

"""
#########
Protocols
#########

"""

import collections
import itertools
import threading
import warnings
from collections.abc import Callable, Iterator
from typing import Any, Literal, Union

Subset = Literal["train", "development", "test"]
LEGACY_SUBSET_MAPPING = {"train": "trn", "development": "dev", "test": "tst"}
Scope = Literal["file", "database", "global"]

Preprocessor = Callable[["ProtocolFile"], Any]
Preprocessors = dict[str, Preprocessor]


class ProtocolFile(collections.abc.MutableMapping):
    """Protocol file with lazy preprocessors

    This is a dict-like data structure where some values may depend on other
    values, and are only computed if/when requested. Once computed, they are
    cached and never recomputed again.

    Parameters
    ----------
    precomputed : dict
        Regular dictionary with precomputed values
    lazy : dict, optional
        Dictionary describing how lazy value needs to be computed.
        Values are callable expecting a dictionary as input and returning the
        computed value.

    """

    def __init__(self, precomputed: Union[dict, "ProtocolFile"], lazy: dict = None):

        if lazy is None:
            lazy = dict()

        if isinstance(precomputed, ProtocolFile):
            # when 'precomputed' is a ProtocolFile, it may already contain lazy keys.

            # we use 'precomputed' precomputed keys as precomputed keys
            self._store: dict = abs(precomputed)

            # we handle the corner case where the intersection of 'precomputed' lazy keys
            # and 'lazy' keys is not empty. this is currently achieved by "unlazying" the
            # 'precomputed' one (which is probably not the most efficient solution).
            for key in set(precomputed.lazy) & set(lazy):
                self._store[key] = precomputed[key]

            # we use the union of 'precomputed' lazy keys and provided 'lazy' keys as lazy keys
            compound_lazy = dict(precomputed.lazy)
            compound_lazy.update(lazy)
            self.lazy: dict = compound_lazy

        else:
            # when 'precomputed' is a Dict, we use it directly as precomputed keys
            # and 'lazy' as lazy keys.
            self._store = dict(precomputed)
            self.lazy = dict(lazy)

        # re-entrant lock used below to make ProtocolFile thread-safe
        self.lock_ = threading.RLock()

        # this is needed to avoid infinite recursion
        # when a key is both in precomputed and lazy.
        # keys with evaluating_ > 0 are currently being evaluated
        # and therefore should be taken from precomputed
        self.evaluating_ = collections.Counter()

    # since RLock is not pickable, remove it before pickling...
    def __getstate__(self):
        d = dict(self.__dict__)
        del d["lock_"]
        return d

    # ... and add it back when unpickling
    def __setstate__(self, d):
        self.__dict__.update(d)
        self.lock_ = threading.RLock()

    def __abs__(self):
        with self.lock_:
            return dict(self._store)

    def __getitem__(self, key):
        with self.lock_:
            if key in self.lazy and self.evaluating_[key] == 0:
                # mark lazy key as being evaluated
                self.evaluating_.update([key])

                # apply preprocessor once and remove it
                value = self.lazy[key](self)
                del self.lazy[key]

                # warn the user when a precomputed key is modified
                if key in self._store and value != self._store[key]:
                    msg = 'Existing precomputed key "{key}" has been modified by a preprocessor.'
                    warnings.warn(msg.format(key=key))

                # store the output of the lazy computation
                # so that it is available for future access
                self._store[key] = value

                # lazy evaluation is finished for key
                self.evaluating_.subtract([key])

            return self._store[key]

    def __setitem__(self, key, value):
        with self.lock_:
            if key in self.lazy:
                del self.lazy[key]

            self._store[key] = value

    def __delitem__(self, key):
        with self.lock_:
            if key in self.lazy:
                del self.lazy[key]

            del self._store[key]

    def __iter__(self):
        with self.lock_:
            store_keys = list(self._store)
            for key in store_keys:
                yield key

            lazy_keys = list(self.lazy)
            for key in lazy_keys:
                if key in self._store:
                    continue
                yield key

    def __len__(self):
        with self.lock_:
            return len(set(self._store) | set(self.lazy))

    def files(self) -> Iterator["ProtocolFile"]:
        """Iterate over all files

        When `current_file` refers to only one file,
            yield it and return.
        When `current_file` refers to a list of file (i.e. 'uri' is a list),
            yield each file separately.

        Examples
        --------
        >>> current_file = ProtocolFile({
        ...     'uri': 'my_uri',
        ...     'database': 'my_database'})
        >>> for file in current_file.files():
        ...     print(file['uri'], file['database'])
        my_uri my_database

        >>> current_file = {
        ...     'uri': ['my_uri1', 'my_uri2', 'my_uri3'],
        ...     'database': 'my_database'}
        >>> for file in current_file.files():
        ...     print(file['uri'], file['database'])
        my_uri1 my_database
        my_uri2 my_database
        my_uri3 my_database

        """

        uris = self["uri"]
        if not isinstance(uris, list):
            yield self
            return

        n_uris = len(uris)

        # iterate over precomputed keys and make sure

        precomputed = {"uri": uris}
        for key, value in abs(self).items():
            if key == "uri":
                continue

            if not isinstance(value, list):
                precomputed[key] = itertools.repeat(value)

            else:
                if len(value) != n_uris:
                    msg = (
                        f'Mismatch between number of "uris" ({n_uris}) '
                        f'and number of "{key}" ({len(value)}).'
                    )
                    raise ValueError(msg)
                precomputed[key] = value

        keys = list(precomputed.keys())
        for values in zip(*precomputed.values()):
            precomputed_one = dict(zip(keys, values))
            yield ProtocolFile(precomputed_one, self.lazy)


class Protocol:
    """Experimental protocol

    An experimental protocol usually defines three subsets: a training subset,
    a development subset, and a test subset.

    An experimental protocol can be defined programmatically by creating a
    class that inherits from Protocol and implements at least
    one of `train_iter`, `development_iter` and `test_iter` methods:

        >>> class MyProtocol(Protocol):
        ...     def train_iter(self) -> Iterator[Dict]:
        ...         yield {"uri": "filename1", "any_other_key": "..."}
        ...         yield {"uri": "filename2", "any_other_key": "..."}

    `{subset}_iter` should return an iterator of dictionnaries with
        - "uri" key (mandatory) that provides a unique file identifier (usually
          the filename),
        - any other key that the protocol may provide.

    It can then be used in Python like this:

        >>> protocol = MyProtocol()
        >>> for file in protocol.train():
        ...    print(file["uri"])
        filename1
        filename2

    An experimental protocol can also be defined using `pyannote.database`
    configuration file, whose (configurable) path defaults to "~/database.yml".

    ~~~ Content of ~/database.yml ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Protocols:
      MyDatabase:
        Protocol:
          MyProtocol:
            train:
                uri: /path/to/collection.lst
                any_other_key: ... # see custom loader documentation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    where "/path/to/collection.lst" contains the list of identifiers of the
    files in the collection:

    ~~~ Content of "/path/to/collection.lst ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    filename1
    filename2
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    It can then be used in Python like this:

        >>> from pyannote.database import registry
        >>> protocol = registry.get_protocol('MyDatabase.Protocol.MyProtocol')
        >>> for file in protocol.train():
        ...    print(file["uri"])
        filename1
        filename2

    This class is usually inherited from, but can be used directly.

    Parameters
    ----------
    preprocessors : dict
        Preprocess protocol files so that `file[key] = preprocessors[key](file)`
        for each key in `preprocessors`. In case `preprocessors[key]` is not
        callable, it should be a string containing placeholders for `file` keys
        (e.g. {'audio': '/path/to/{uri}.wav'})
    """

    def __init__(self, preprocessors: Preprocessors | None = None):
        super().__init__()

        if preprocessors is None:
            preprocessors = dict()

        self.preprocessors = dict()
        for key, preprocessor in preprocessors.items():
            if callable(preprocessor):
                self.preprocessors[key] = preprocessor

            # when `preprocessor` is not callable, it should be a string
            # containing placeholder for item key (e.g. '/path/to/{uri}.wav')
            elif isinstance(preprocessor, str):
                preprocessor_copy = str(preprocessor)

                def func(current_file):
                    return preprocessor_copy.format(**current_file)

                self.preprocessors[key] = func

            else:
                msg = f'"{key}" preprocessor is neither a callable nor a string.'
                raise ValueError(msg)

    def preprocess(self, current_file: dict | ProtocolFile) -> ProtocolFile:
        return ProtocolFile(current_file, lazy=self.preprocessors)

    def __str__(self):
        return self.__doc__

    def train_iter(self) -> Iterator[dict | ProtocolFile]:
        """Iterate over files in the training subset"""
        raise NotImplementedError()

    def development_iter(self) -> Iterator[dict | ProtocolFile]:
        """Iterate over files in the development subset"""
        raise NotImplementedError()

    def test_iter(self) -> Iterator[dict | ProtocolFile]:
        """Iterate over files in the test subset"""
        raise NotImplementedError()

    def subset_helper(self, subset: Subset) -> Iterator[ProtocolFile]:

        try:
            files = getattr(self, f"{subset}_iter")()
        except (AttributeError, NotImplementedError):
            # previous pyannote.database versions used `trn_iter` instead of
            # `train_iter`, `dev_iter` instead of `development_iter`, and
            # `tst_iter` instead of `test_iter`. therefore, we use the legacy
            # version when it is available (and the new one is not).
            subset_legacy = LEGACY_SUBSET_MAPPING[subset]
            try:
                files = getattr(self, f"{subset_legacy}_iter")()
            except AttributeError:
                msg = f"Protocol does not implement a {subset} subset."
                raise NotImplementedError(msg)

        for file in files:
            yield self.preprocess(file)

    def train(self) -> Iterator[ProtocolFile]:
        return self.subset_helper("train")

    def development(self) -> Iterator[ProtocolFile]:
        return self.subset_helper("development")

    def test(self) -> Iterator[ProtocolFile]:
        return self.subset_helper("test")

    def files(self) -> Iterator[ProtocolFile]:
        """Iterate over all files in `protocol`"""

        # imported here to avoid circular imports
        from pyannote.database.util import get_unique_identifier

        yielded_uris = set()

        for method in [
            "development",
            "development_enrolment",
            "development_trial",
            "test",
            "test_enrolment",
            "test_trial",
            "train",
            "train_enrolment",
            "train_trial",
        ]:
            if not hasattr(self, method):
                continue

            def iterate():
                try:
                    yield from getattr(self, method)()
                except (AttributeError, NotImplementedError):
                    return

            for current_file in iterate():
                # skip "files" that do not contain a "uri" entry.
                # this happens for speaker verification trials that contain
                # two nested files "file1" and "file2"
                # see https://github.com/pyannote/pyannote-db-voxceleb/issues/4
                if "uri" not in current_file:
                    continue

                for current_file_ in current_file.files():
                    # corner case when the same file is yielded several times
                    uri = get_unique_identifier(current_file_)
                    if uri in yielded_uris:
                        continue

                    yield current_file_

                    yielded_uris.add(uri)
