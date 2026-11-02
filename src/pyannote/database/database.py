# The MIT License (MIT)

# Copyright (c) 2016 CNRS

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


import warnings

from .protocol.protocol import Preprocessors


class Database:
    """Base database

    This class should be inherited from, not used directly.

    """

    def __init__(self, preprocessors=None):

        if preprocessors is not None:
            database_name = self.__class__.__name__
            msg = (
                f"Ignoring deprecated 'preprocessors' argument in {database_name}.__init__. "
                f"Pass it to 'get_protocol' instead."
            )
            warnings.warn(msg)

        super().__init__()

    def register_protocol(self, task_name, protocol_name, protocol):
        if not hasattr(self, "protocols_"):
            self.protocols_ = {}
        if task_name not in self.protocols_:
            self.protocols_[task_name] = {}
        self.protocols_[task_name][protocol_name] = protocol
        # TODO / register globally.

    def _get_tasks(self):
        try:
            tasks = self.protocols_
        except AttributeError:
            message = "This database does not implement any protocol."
            raise AttributeError(message)
        return tasks

    def get_tasks(self):
        tasks = self._get_tasks()
        return sorted(tasks)

    def get_protocols(self, task):
        return sorted(self.protocols_[task].keys())

    def get_protocol(self, task, protocol, preprocessors: Preprocessors | None = None):
        return self.protocols_[task][protocol](preprocessors=preprocessors)

    def __str__(self):
        return self.__doc__
