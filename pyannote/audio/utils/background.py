#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019-2020 CNRS

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
Background generators
=====================

This module provides ways to send a data generator to one (or multiple)
background thread(s).

A typical use case is when training a neural network using mini-batches:
batches are produced on CPU and consumed (often faster) on GPU, leading to
sub-optimal usage of GPU resource (hence slow training).

`BackgroundGenerator` allows to send the CPU producer to a background thread so
that it can generate a new batch while the previous one is being consumed by
the GPU.

`AdaptiveBackgroundGenerator` goes one step further and uses a pool of
background threads whose size automatically (and continuously) adapts the
production rate to the consumption rate.
"""

import threading
import collections
import queue
import time
from typing import Iterator, Callable
import numpy as np


class BackgroundGenerator(threading.Thread):
    """Background generator with production/consumption time estimates

    Parameters
    ----------
    producer: generator function
        Generator function that takes no argument and yield (a possibly
        infinite number of) samples. This would typically be a BatchGenerator
        instance but can be any function that "yields" samples.
    prefetch: int, optional
        Maximum number of samples that can be prefetched and stored in a queue.
        Defaults to 1. In case the consumer is slower than the producer and the
        queue is full, the producer is paused until one sample is consumed.

    Usage
    -----
    >>> import time

    # a dummy producer that yield 'sample' string every 10ms.
    >>> def produce():
    ...     while True:
    ...        time.sleep(0.010)
    ...        yield 'sample'

    # a dummy consumer that takes 1ms to consume a sample
    >>> def consume(sample):
    ...     time.sleep(0.001)

    # create background generator from producer
    >>> generator = BackgroundGenerator(produce)

    # produce and consume 100 samples
    >>> for i in range(100):
    ...     sample = next(generator)
    ...     consume(sample)

    >>> p = generator.production_time
    >>> print(f'Production time estimate: {1000 * p:.0f}ms')
    # Production time estimate: 10ms

    >>> c = generator.consumption_time
    >>> print(f'Consumption time estimate: {1000 * c:.0f}ms')
    # Consumption time estimate: 1ms

    # kill background generator (and associated thread)
    >>> generator.deactivate()
    >>> sample = next(generator)
    # StopIteration: Background generator is no longer active.
    """

    def __init__(self, producer: Callable[[], Iterator], prefetch: int = 1):
        super().__init__(daemon=True)
        self.producer = producer
        self.prefetch = prefetch

        self.activated_ = True
        self.producer_ = producer()

        # used to keep track of how long it took to generate latest samples
        self.production_time_ = collections.deque([], max(10, 2 * self.prefetch))

        # used to keep track of how long it took to consume latest samples
        self.consumption_time_ = collections.deque([], max(10, 2 * self.prefetch))

        # used to keep track of last time
        self.last_ready_ = None

        # queue meant to store at most 'self.prefetch' prefetched samples
        self.queue_ = queue.Queue(self.prefetch)

        # start generator in a new thread
        self.start()

    def reset(self) -> None:
        """Reset production and consumption time estimators"""
        self.production_time_.clear()
        self.consumption_time_.clear()

    def deactivate(self) -> None:
        """Stop background generator"""
        self.activated_ = False
        # unlock queue stuck at line queue.put() in self.run()
        _ = self.queue_.get()

    @property
    def production_time(self) -> float:
        """Estimated time needed by the generator to yield a sample.

        This is computed as the median production time of the last few samples.

        Returns
        -------
        production_time : float or np.NAN
            Estimated time needed by the generator to yield a new sample, in
            seconds. Until enough samples have been yielded to accurately
            estimate production time, it is set to np.NAN.
        """

        if len(self.production_time_) < max(10, 2 * self.prefetch):
            return np.NAN
        return np.median(self.production_time_)

    @property
    def consumption_time(self) -> float:
        """Estimated time needed by the consumer to process a sample

        This is computed as the median consumption time of the last few samples.

        Returns
        -------
        consumption_time : float or np.NAN
            Estimated time needed by the consumer to process a sample, in
            seconds. Until enough samples have been consumed to accurately
            estimate consumption time, it is set to np.NAN.
        """
        if len(self.consumption_time_) < max(10, 2 * self.prefetch):
            return np.NAN
        return np.median(self.consumption_time_)

    def run(self) -> None:
        """Called by self.start(), should not be called directly."""

        # keep going until the background generator is deactivated
        while self.activated_:

            # produce a new sample
            _t = time.time()
            try:
                sample = next(self.producer_)
            except StopIteration:
                sample = None

            # keep track of how long it took to produce
            self.production_time_.append(time.time() - _t)

            # put the new sample into the queue for later consumption.
            # note that this line is blocking when the queue is full.
            # calling self.queue_.get() in self.__next__() or self.deactivate()
            # will eventually unblock it.
            self.queue_.put(sample)

    def __next__(self):
        """Produce new sample"""

        # raise a StopIteration once the generator has been deactivated
        if not self.activated_:
            msg = "Background generator is no longer active."
            raise StopIteration(msg)

        # keep track of how long it took to consume the last sample
        t = time.time()
        if self.last_ready_ is not None:
            self.consumption_time_.append(t - self.last_ready_)

        # get a new sample from the queue
        sample = self.queue_.get()

        # this happens when producer stopped yielding samples
        if sample is None:
            msg = "Producer stopped yielding samples."
            raise StopIteration(msg)

        # keep track of the last time a sample was taken from the queue
        self.last_ready_ = time.time()

        # actually return the new sample
        return sample

    def __iter__(self):
        return self


class AdaptiveBackgroundGenerator:
    """Adaptive pool of background generators

    The pool is initialized with only one background generator.

    Once production and consumption time estimates are available (after a short
    warm-up period of time), the pool will incrementally adapt the number of
    background generators to ensure that it produces samples fast enough for
    the consumer.

    Parameters
    ----------
    producer: generator function
        Generator function that takes no argument and yield (a possibly
        infinite number of) samples. This would typically be a BatchGenerator
        instance but can be any function that "yields" samples.
    n_jobs : int, optional
        Maximum number of background generators that can be created to keep up
        with consumer. Defaults to 4.
    prefetch : int, optional
        Maximum number of samples that can be prefetched by each background
        generator. See BackgroundGenerator documentation for more details.
        Defaults to 10.
    verbose : bool, optional
        Print a message when a background generator is added to (or removed
        from) the pool.

    Usage
    -----
    >>> import time

    # A producer that takes 5ms to produce a new sample
    >>> def producer():
    ...     while True:
    ...         time.sleep(0.005)
    ...         yield 'data'

    # A slow consumer that takes 5ms to consume a sample
    >>> def slow_consumer(data): time.sleep(0.005)

    # A fast consumer that takes 1ms to consume a sample
    >>> def fast_consumer(data): time.sleep(0.001)

    # send producer to the background and allows for at most 6 threads
    >>> generator = AdaptiveBackgroundGenerator(producer, n_jobs=6)

    >>> for _ in range(1000): fast_consumer(next(generator))
    >>> print(f'When consumer is fast, generator uses {len(generator)} thread(s).')
    # prints: "When consumer is fast, generator uses 4 thread(s)."

    >>> for _ in range(1000): slow_consumer(next(generator))
    >>> print(f'When consumer is slow, generator uses {len(generator)} thread(s).')
    # prints: "When consumer is slow, generator uses 1 thread(s)."

    # deactivate generator (and stop background threads)
    >>> generator.deactivate()
    >>> _ = next(generator)
    # raises: "StopIteration: Background generator is no longer active."
    """

    def __init__(
        self,
        producer: Callable[[], Iterator],
        n_jobs: int = 4,
        prefetch: int = 10,
        verbose: bool = False,
    ):

        self.producer = producer
        self.n_jobs = n_jobs
        self.prefetch = prefetch
        self.verbose = verbose

        # current pool of active background generators
        self.generators_ = []

        if self.verbose:
            msg = f"Starting with one producer."
            print(msg)

        # start by creating one background generator to the pool
        self._add_generator()

        # initialize main sample generator (used in __next__)
        self.samples_ = self._sample()

        # set to True once the maximum number of generators is reached.
        # (used to avoid repeating a message when verbose is True)
        self.reached_max_ = False

    def deactivate(self) -> None:
        """Stop background generator"""
        n_jobs = len(self.generators_)
        for _ in range(n_jobs):
            self._remove_generator()

    def _add_generator(self) -> None:
        """Add one more producer to the pool"""

        self.generators_.append(
            BackgroundGenerator(self.producer, prefetch=self.prefetch)
        )

        for g in self.generators_:
            g.reset()

    def _remove_generator(self, index: int = None) -> None:
        """Remove one producer from the pool

        Parameters
        ----------
        index : int, optional
            When provided, remove `index`th producer.
            Defaults to removing the last producer.
        """

        if index is None:
            n_jobs = len(self.generators_)
            index = n_jobs - 1

        g = self.generators_.pop(index)
        g.deactivate()

        for g in self.generators_:
            g.reset()

        self.reached_max_ = False

    def __len__(self):
        """Return current number of producers"""
        return len(self.generators_)

    @property
    def consumption_time(self) -> float:
        """Estimated time needed by the consumer to process a sample

        This is computed as the average of estimated consumption times of all
        currently active background generators.

        Returns
        -------
        consumption_time : float or np.NAN
            Estimated time needed by the consumer to process a sample, in
            seconds. Until enough samples have been consumed to accurately
            estimate consumption time, it is set to np.NAN.
        """

        # corner case when generator has been deactivated
        if not self.generators_:
            return np.NAN

        return np.mean([g.consumption_time for g in self.generators_])

    @property
    def production_time(self) -> float:
        """Estimated time needed by the generator to yield a sample.

        This is computed as the average estimated production time of all
        currently active background generators.

        Returns
        -------
        production_time : float or np.NAN
            Estimated time needed by the generator to yield a new sample, in
            seconds. Until enough samples have been yielded to accurately
            estimate production time, it is set to np.NAN.
        """

        # corner case when generator has been deactivated
        if not self.generators_:
            return np.NAN

        return np.mean([g.production_time for g in self.generators_])

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.samples_)

    def _sample(self) -> Iterator:
        """Iterate over (and manage) pool of generators"""

        # loop forever
        while True:

            if not self.generators_:
                msg = "Background generator is no longer active."
                raise StopIteration(msg)

            dead_generators = []
            for index, g in enumerate(self.generators_):

                try:
                    sample = next(g)
                except StopIteration:
                    # mark this generator as dead
                    dead_generators.append(index)
                    continue

                yield sample

            if self.verbose and dead_generators:
                msg = f"Replacing {len(dead_generators)} exhausted producers."
                print(msg)

            # replace dead generators by new ones
            for index in reversed(dead_generators):
                self._remove_generator(index=index)
            for _ in dead_generators:
                self._add_generator()

            consumption_time = self.consumption_time
            production_time = self.production_time

            if np.isnan(consumption_time) or np.isnan(production_time):
                continue

            n_jobs = len(self.generators_)
            ratio = production_time / consumption_time

            # consumption_time < production_time
            if ratio > 1:
                if n_jobs < self.n_jobs:

                    if self.verbose:
                        msg = (
                            f"Adding one producer because consumer is "
                            f"{ratio:.2f}x faster than current {n_jobs:d} "
                            f"producer(s)."
                        )
                        print(msg)

                    self._add_generator()

                else:
                    if not self.reached_max_ and self.verbose:
                        msg = (
                            f"Consumer is {ratio:.2f}x faster than the pool of "
                            f"{n_jobs:d} producer(s) but the maximum number of "
                            f"producers has been reached."
                        )
                        print(msg)

                    self.reached_max_ = True

            # production_time < consumption_time * (n_jobs - 1) / n_jobs
            elif (ratio < (n_jobs - 1) / n_jobs) and n_jobs > 1:

                if self.verbose:
                    msg = (
                        f"Removing one producer because consumer is "
                        f"{1 / ratio:.2f}x slower than current {n_jobs:d} "
                        f"producer(s)."
                    )
                    print(msg)

                self._remove_generator()
