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

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_tune_space(self):
        pass

    @abstractmethod
    def get_tune_metric(self):
        pass

    def with_params(self, **params):
        return self

    @abstractmethod
    def apply(self, current_file):
        pass

    def objective(self, protocol, subset='development'):
        metric = self.get_tune_metric()
        for current_file in getattr(protocol, subset)():
            reference = current_file['annotation']
            uem = get_annotated(current_file)
            hypothesis = self.apply(current_file)
            metric(reference, hypothesis, uem=uem, detailed=False)
        return abs(metric)

    def best(self, tune_db=None, connection=None):
        """

        Returns
        -------
        status : dict
             {'loss': <best loss so far>,
              'params': <corresponding set of hyper-parameters>,
              'n_trials': <overall total number of trials>}}
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
        """

        Parameters
        ----------
        tune_db : str
            Path to SQLite database.
        protocol : pyannote.database.Protocol
            Evaluation protocol.
        subset : {'train', 'development', 'test'}, optional
            Defaults to 'development'.
        n_calls : int, optional
            Number of trials. Defaults to 1.
            Set `n_calls` to 0 to obtain best set of params.

        Returns
        -------
        best : dict
            Best set of hyper-parameters.
             {'loss': <best loss so far>,
              'params': <corresponding set of hyper-parameters>,
              'n_trials': <overall total number of trials>}}

        """

        iterations = self.tune_iter(tune_db, protocol, subset=subset)

        for i in range(n_calls):
            _ = next(iterations)

        return self.best(tune_db=tune_db)


    def tune_iter(self, tune_db, protocol, subset='development'):
        """Tune pipeline

        Parameters
        ----------
        tune_db : str
            Path to SQLite database.
        protocol : pyannote.database.Protocol
            Evaluation protocol.
        subset : {'train', 'development', 'test'}, optional
            Defaults to 'development'.

        Yields
        ------
        status : dict
            {'latest': {'loss': <loss of the latest trial>,
                      'params': <corresponding set of hyper-parameters>,
                      'n_trials': <total number of trials in this session>},
             'new_best': {'loss': <best loss so far>,
                      'params': <corresponding set of hyper-parameters>,
                      'n_trials': <overall total number of trials>}}

        """

        # start connection to SQLite database
        # (this is where trials are stored)
        connection = chocolate.SQLiteConnection(f'sqlite:///{tune_db}')

        # get hyper-parameter space
        space = self.get_tune_space()

        # instantiate sampler
        sampler = chocolate.CMAES(connection, space)

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

            # tell the optimizer what was the result
            sampler.update(token, loss)

            if loss < best['loss'] or i == 1:
                # if loss is better than previous known best
                # check in the database what is the current best
                best = self.best(connection=connection)
                yield {'latest': latest, 'new_best': best}
            else:
                yield {'latest': latest}
