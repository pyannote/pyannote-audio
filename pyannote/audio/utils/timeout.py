#!/usr/bin/env python
# encoding: utf-8

# Shamelessly stolen from
# https://gist.github.com/TySkby/143190ad1b88c6115597c45f996b030c

"""Easily put time restrictions on things

Note: Requires Python 3.x

Usage as a context manager:
```
with timeout(10):
    something_that_should_not_exceed_ten_seconds()
```

Usage as a decorator:
```
@timeout(10)
def something_that_should_not_exceed_ten_seconds():
    do_stuff_with_a_timeout()
```

Handle timeouts:
```
try:
   with timeout(10):
       something_that_should_not_exceed_ten_seconds()
   except TimeoutError:
       log('Got a timeout, couldn't finish')
```

Suppress TimeoutError and just die after expiration:
```
with timeout(10, suppress_timeout_errors=True):
    something_that_should_not_exceed_ten_seconds()

print('Maybe exceeded 10 seconds, but finished either way')
```
"""
import contextlib
import errno
import os
import signal


DEFAULT_TIMEOUT_MESSAGE = os.strerror(errno.ETIME)


class timeout(contextlib.ContextDecorator):
    def __init__(
        self,
        seconds,
        *,
        timeout_message=DEFAULT_TIMEOUT_MESSAGE,
        suppress_timeout_errors=False
    ):
        self.seconds = int(seconds)
        self.timeout_message = timeout_message
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)
        if self.suppress and exc_type is TimeoutError:
            return True
