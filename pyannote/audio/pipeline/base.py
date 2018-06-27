#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018 CNRS

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

import numpy as np
from pyannote.database import get_annotated
from abc import ABCMeta, abstractmethod
import chocolate


class Pipeline:
    """Base class for jointly optimized pipelines"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_tune_space(self):
        pass

    @abstractmethod
    def get_tune_metric(self):
        pass

    def with_params(self, **params):
        """Instantiate pipeline with given set of keyword parameters

        Must be overridden by sub-classes.
        """
        return self

    @abstractmethod
    def apply(self, current_file):
        """Apply pipeline on current file

        Must be overridden by sub-classes.
        """
        pass

    def objective(self, protocol, subset='development'):
        """Compute the value of the bjective function (the lower, the better)

        Parameters
        ----------
        protocol : pyannote.database.Protocol
            Protocol on which to compute the value of the objective function.
        subset : {'train', 'development', 'test'}, optional
            Subset on which to compute the value of the objective function.
            Defaults to 'development'.

        Returns
        -------
        metric : float
            Value of the objective function (the lower, the better).
        """
        metric = self.get_tune_metric()
        value, duration = [], []
        # NOTE -- embarrasingly parallel
        # TODO -- parallelize this
        for current_file in getattr(protocol, subset)():
            reference = current_file['annotation']
            uem = get_annotated(current_file)
            hypothesis = self.apply(current_file)
            if hypothesis is None:
                return 1.
            metric_value = metric(reference, hypothesis, uem=uem)
            value.append(metric_value)
            duration.append(uem.duration())

        # support for pyannote.metrics
        if hasattr(metric, '__abs__'):
            return abs(metric)
        # support for any other metric
        else:
            return np.average(value, weights=duration)

    def best(self, tune_db=None, connection=None):
        """Get (current) best set of hyper-parameters

        Parameters
        ----------
        connection : chocolate.SQLiteConnection, optional
            Existing connection to SQLite database.
        tune_db : str, optional
            Path to SQLite database where trial results will be stored. Has no
            effect when `connection` is provided.

        At least one of `tune_db` or `connection` must be provided.

        Returns
        -------
        status : dict
            ['loss'] (`float`) best loss so far
            ['params'] (`dict`) corresponding set of hyper-parameters
            ['n_trials'] (`int`) total number of trials
        """

        if connection is None:
            # start connection to SQLite database
            # (this is where trials are stored)
            connection = chocolate.SQLiteConnection(f'sqlite:///{tune_db}')

        # get current best set of hyper-parameter (and its loss)
        trials = connection.results_as_dataframe()
        best_params = dict(trials.iloc[trials['_loss'].idxmin()])
        best_loss = best_params.pop('_loss')
        best_params = {name: np.asscalar(value)
                       for name, value in best_params.items()}

        return {'loss': best_loss,
                'params': best_params,
                'n_trials': len(trials)}

    def tune(self, tune_db, protocol, subset='development', n_calls=1):
        """Tune pipeline

        Parameters
        ----------
        tune_db : str
            Path to SQLite database where trial results will be stored.
        protocol : pyannote.database.Protocol
            Protocol on which to tune the pipeline.
        subset : {'train', 'development', 'test'}, optional
            Subset on which to tune the pipeline. Defaults to 'development'.
        sampler : chocolate sampler, optional
            Defaults to chocolate.CMAES
        n_calls : int, optional
            Number of trials. Defaults to 1.
            Set `n_calls` to 0 to obtain best set of params.

        Returns
        -------
        best : dict
            ['loss'] (`float`) best loss so far
            ['params'] (`dict`) corresponding set of hyper-parameters
            ['n_trials'] (`int`) total number of trials
        """

        iterations = self.tune_iter(tune_db, protocol, subset=subset,
                                    sampler=sampler)

        for i in range(n_calls):
            _ = next(iterations)

        return self.best(tune_db=tune_db)


    def tune_iter(self, tune_db, protocol, subset='development',
                  sampler=None):
        """Tune pipeline forever

        Parameters
        ----------
        tune_db : str
            Path to SQLite database where trial results will be stored.
        protocol : pyannote.database.Protocol
            Protocol on which to tune the pipeline.
        subset : {'train', 'development', 'test'}, optional
            Subset on which to tune the pipeline. Defaults to 'development'.
        sampler : chocolate sampler, optional
            Defaults to chocolate.CMAES

        Yields
        ------
        status : dict
            ['latest']['loss'] (`float`) loss obtained by the latest trial
            ['latest']['params'] (`dict`) corresponding set of hyper-parameters
            ['latest']['n_trials'] (`int`) total number of trials in thes session
            ['new_best']['loss'] (`float`) best loss so far
            ['new_best']['params'] (`dict`) corresponding set of hyper-parameters
            ['new_best']['n_trials'] (`int`) total number of trials
        """

        # start connection to SQLite database
        # (this is where trials are stored)
        connection = chocolate.SQLiteConnection(f'sqlite:///{tune_db}')

        # get hyper-parameter space
        space = self.get_tune_space()

        # instantiate sampler
        if sampler is None:
            sampler = chocolate.CMAES
        sampler = sampler(connection, space)
        # TODO add option to use another sampler

        i = 0
        best = {'loss': np.inf}

        while True:
            i += 1

            # get next set of hyper-parameters to try
            token, params = sampler.next()

            # instantiate pipeline with this set of parameters
            # and compute the objective function
            loss = self.with_params(**params).objective(
                protocol, subset=subset)

            latest = {'loss': loss, 'params': params, 'n_trials': i}

            # tell the sampler what was the result
            sampler.update(token, loss)

            if loss < best['loss'] or i == 1:
                # if loss is better than previous known best
                # check in the database what is the current best
                best = self.best(connection=connection)
                yield {'latest': latest, 'new_best': best}
            else:
                yield {'latest': latest}
