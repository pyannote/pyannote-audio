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


from .protocol import Protocol


class SpeakerRecognitionProtocol(Protocol):
    """Speaker recognition protocol

    Parameters
    ----------
    preprocessors : dict or (key, preprocessor) iterable
        When provided, each protocol item (dictionary) are preprocessed, such
        that item[key] = preprocessor(item). In case 'preprocessor' is not
        callable, it should be a string containing placeholder for item keys
        (e.g. {'wav': '/path/to/{uri}.wav'})
    """

    def trn_iter(self):
        raise NotImplementedError(
            'Custom speaker recognition protocol should implement "trn_iter".'
        )

    def trn_enroll_iter(self):
        raise NotImplementedError(
            'Custom speaker recognition protocol should implement "trn_enroll_iter".'
        )

    def trn_test_iter(self):
        raise NotImplementedError(
            'Custom speaker recognition protocol should implement "trn_test_iter".'
        )

    def trn_keys(self):
        raise NotImplementedError(
            'Custom speaker recognition protocol should implement "trn_keys".'
        )

    def dev_enroll_iter(self):
        raise NotImplementedError(
            'Custom speaker recognition protocol should implement "dev_enroll_iter".'
        )

    def dev_test_iter(self):
        raise NotImplementedError(
            'Custom speaker recognition protocol should implement "dev_test_iter".'
        )

    def dev_keys(self):
        raise NotImplementedError(
            'Custom speaker recognition protocol should implement "dev_keys".'
        )

    def tst_enroll_iter(self):
        raise NotImplementedError(
            'Custom speaker recognition protocol should implement "tst_enroll_iter".'
        )

    def tst_test_iter(self):
        raise NotImplementedError(
            'Custom speaker recognition protocol should implement "tst_test_iter".'
        )

    def tst_keys(self):
        raise NotImplementedError(
            'Custom speaker recognition protocol should implement tst_keys".'
        )

    def train(self, yield_name=False):
        """Iterate over the training set

        This will yield dictionaries with the followings keys:

        * database: str
          unique database identifier
        * uri: str
          unique recording identifier
        * channel: int
          index of resource channel to use
        * speaker: str
          unique speaker identifier

        as well as keys coming from the provided preprocessors.

        Usage
        -----
        >>> for item in protocol.train():
        ...     uri = item['uri']
        ...     channel = item['channel']
        ...     speaker = item['speaker']
        """

        generator = self.trn_iter()

        for name, item in generator:
            if yield_name:
                yield name, self.preprocess(item)
            else:
                yield self.preprocess(item)

    def train_enroll(self, yield_name=True):
        """Iterate over the training set enrollments

        This will yield dictionaries with the followings keys:

        * database: str
          unique database identifier
        * uri: str
          uniform (or unique) resource identifier
        * channel: int
          index of resource channel to use

        as well as keys coming from the provided preprocessors.

        Usage
        -----
        >>> for item in protocol.train_enroll():
        ...     uri = item['uri']
        ...     channel = item['channel']
        """

        generator = self.trn_enroll_iter()

        for name, item in generator:
            if yield_name:
                yield name, self.preprocess(item)
            else:
                yield self.preprocess(item)

    def train_test(self, yield_name=True):
        """Iterate over the training set tests

        This will yield dictionaries with the followings keys:

        * database: str
          unique database identifier
        * uri: str
          uniform (or unique) resource identifier
        * channel: int
          index of resource channel to use

        as well as keys coming from the provided preprocessors.

        Usage
        -----
        >>> for item in protocol.train_test():
        ...     uri = item['uri']
        ...     channel = item['channel']
        """

        generator = self.trn_test_iter()

        for name, item in generator:
            if yield_name:
                yield name, self.preprocess(item)
            else:
                yield self.preprocess(item)

    def train_keys(self):
        return self.trn_keys()

    def development_enroll(self, yield_name=True):
        """Iterate over the development set enrollments

        This will yield dictionaries with the followings keys:

        * database: str
          unique database identifier
        * uri: str
          uniform (or unique) resource identifier
        * channel: int
          index of resource channel to use

        as well as keys coming from the provided preprocessors.

        Usage
        -----
        >>> for item in protocol.development_enroll():
        ...     uri = item['uri']
        ...     channel = item['channel']
        """

        generator = self.dev_enroll_iter()

        for name, item in generator:
            if yield_name:
                yield name, self.preprocess(item)
            else:
                yield self.preprocess(item)

    def development_test(self, yield_name=True):
        """Iterate over the development set tests

        This will yield dictionaries with the followings keys:

        * database: str
          unique database identifier
        * uri: str
          uniform (or unique) resource identifier
        * channel: int
          index of resource channel to use

        as well as keys coming from the provided preprocessors.

        Usage
        -----
        >>> for item in protocol.development_test():
        ...     uri = item['uri']
        ...     channel = item['channel']
        """

        generator = self.dev_test_iter()

        for name, item in generator:
            if yield_name:
                yield name, self.preprocess(item)
            else:
                yield self.preprocess(item)

    def development_keys(self):
        return self.dev_keys()

    def test_enroll(self, yield_name=True):
        """Iterate over the test set targets

        This will yield dictionaries with the followings keys:

        * database: str
          unique database identifier
        * uri: str
          uniform (or unique) resource identifier
        * channel: int
          index of resource channel to use
        * speaker: str
          unique speaker identifier

        as well as keys coming from the provided preprocessors.

        Usage
        -----
        >>> for item in protocol.test_enroll():
        ...     uri = item['uri']
        ...     channel = item['channel']
        ...     speaker = item['speaker']
        """

        generator = self.tst_enroll_iter()

        for name, item in generator:
            if yield_name:
                yield name, self.preprocess(item)
            else:
                yield self.preprocess(item)

    def test_test(self, yield_name=True):
        """Iterate over the test set tests

        This will yield dictionaries with the followings keys:

        * database: str
          unique database identifier
        * uri: str
          uniform (or unique) resource identifier
        * channel: int
          index of resource channel to use

        as well as keys coming from the provided preprocessors.

        Usage
        -----
        >>> for item in protocol.test_test():
        ...     uri = item['uri']
        ...     channel = item['channel']
        """

        generator = self.tst_test_iter()

        for name, item in generator:
            if yield_name:
                yield name, self.preprocess(item)
            else:
                yield self.preprocess(item)

    def test_keys(self):
        return self.tst_keys()
