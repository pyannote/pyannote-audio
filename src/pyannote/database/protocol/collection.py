# The MIT License (MIT)

# Copyright (c) 2018-2020 CNRS

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


from collections.abc import Iterator

from .protocol import Protocol


class CollectionProtocol(Protocol):
    """A collection of files with no train/dev/test split

    A collection can be defined programmatically by creating a class that
    inherits from CollectionProtocol and implements the `files_iter` method:

        >>> class MyCollection(CollectionProtocol):
        ...     def files_iter(self) -> Iterator[Dict]:
        ...         yield {"uri": "filename1", "any_other_key": "..."}
        ...         yield {"uri": "filename2", "any_other_key": "..."}
        ...         yield {"uri": "filename3", "any_other_key": "..."}

    `files_iter` should return an iterator of dictionnaries with
        - a mandatory "uri" key that provides a unique file identifier (usually
          the filename),
        - any other key that the collection may provide.

    It can then be used in Python like this:

        >>> collection = MyCollection()
        >>> for file in collection.files():
        ...    print(file["uri"])
        filename1
        filename2
        filename3

    A collection can also be defined using `pyannote.database` configuration
    file, whose (configurable) path defaults to "~/database.yml".

    ~~~ Content of ~/database.yml ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Protocols:
      MyDatabase:
        Collection:
          MyCollection:
            uri: /path/to/collection.lst
            any_other_key: ... # see custom loader documentation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    where "/path/to/collection.lst" contains the list of identifiers of the
    files in the collection:

    ~~~ Content of "/path/to/collection.lst ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    filename1
    filename2
    filename3
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    It can the be used in Python like this:

        >>> from pyannote.database import registry
        >>> collection = registry.get_protocol('MyDatabase.Collection.MyCollection')
        >>> for file in collection.files():
        ...    print(file["uri"])
        filename1
        filename2
        filename3
    """

    # this method should be overriden
    def files_iter(self) -> Iterator[dict]:
        raise NotImplementedError()

    # this allows Protocol.files() to iterate over the collection
    def train_iter(self) -> Iterator[dict]:
        return self.files_iter()
