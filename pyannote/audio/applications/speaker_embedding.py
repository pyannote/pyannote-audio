#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2020 CNRS

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

import torch
import numpy as np
from typing import Optional

from .base import Application

from pyannote.core import Segment, Timeline, Annotation

from pyannote.database import get_protocol
from pyannote.database import get_annotated
from pyannote.database import get_unique_identifier
from pyannote.database import FileFinder
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.database.protocol import SpeakerVerificationProtocol

from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage

from pyannote.core.utils.distance import pdist
from pyannote.core.utils.distance import cdist
from pyannote.audio.features.precomputed import Precomputed

from pyannote.metrics.binary_classification import det_curve
from pyannote.metrics.diarization import DiarizationPurityCoverageFMeasure

from pyannote.audio.features import Pretrained


class SpeakerEmbedding(Application):

    @property
    def config_default_module(self):
        return 'pyannote.audio.embedding.approaches'

    def validation_criterion(self, protocol_name, purity=0.9, **kwargs):
        protocol = get_protocol(protocol_name)
        if isinstance(protocol, SpeakerVerificationProtocol):
            return f'equal_error_rate'
        elif isinstance(protocol, SpeakerDiarizationProtocol):
            return f'purity={100*purity:.0f}%'

    def validate_init(self, protocol_name,
                            subset='development'):

        protocol = get_protocol(protocol_name)

        if isinstance(protocol, (SpeakerVerificationProtocol,
                                 SpeakerDiarizationProtocol)):
            return

        msg = ('Only SpeakerVerification or SpeakerDiarization tasks are'
               'supported in "validation" mode.')
        raise ValueError(msg)

    def validate_epoch(self,
                       epoch,
                       validation_data,
                       protocol=None,
                       **kwargs):

        _protocol = get_protocol(protocol)

        if isinstance(_protocol, SpeakerVerificationProtocol):
            return self._validate_epoch_verification(
                epoch, validation_data, protocol=protocol, **kwargs)

        elif isinstance(_protocol, SpeakerDiarizationProtocol):
            return self._validate_epoch_diarization(
                epoch, validation_data, protocol=protocol, **kwargs)

        else:
            msg = ('Only SpeakerVerification or SpeakerDiarization tasks are'
                   'supported in "validation" mode.')
            raise ValueError(msg)

    @staticmethod
    def get_hash(file):
        hashable = []
        for f in file.files():
            hashable.append((f['uri'], tuple(f['try_with'])))
        return hash(tuple(sorted(hashable)))

    @staticmethod
    def get_embedding(file, pretrained):
        emb = []
        for f in file.files():
            if isinstance(f['try_with'], Segment):
                segments = [f['try_with']]
            else:
                segments = f['try_with']
            for segment in segments:
                emb.append(pretrained.crop(f, segment))

        return np.mean(np.vstack(emb), axis=0, keepdims=True)

    def _validate_epoch_verification(self,
                                     epoch,
                                     validation_data,
                                     protocol=None,
                                     subset='development',
                                     device: Optional[torch.device] = None,
                                     batch_size: int = 32,
                                     n_jobs: int = 1,
                                     duration: float = None,
                                     step : float = 0.25,
                                     metric : str = None,
                                     **kwargs):

        # initialize embedding extraction
        pretrained = Pretrained(validate_dir=self.validate_dir_,
                                epoch=epoch,
                                duration=duration,
                                step=step,
                                batch_size=batch_size,
                                device=device)

        _protocol = get_protocol(protocol, progress=False,
                                preprocessors=self.preprocessors_)

        y_true, y_pred, cache = [], [], {}

        for trial in getattr(_protocol, '{0}_trial'.format(subset))():

            # compute embedding for file1
            file1 = trial['file1']
            hash1 = self.get_hash(file1)
            if hash1 in cache:
                emb1 = cache[hash1]
            else:
                emb1 = self.get_embedding(file1, pretrained)
                cache[hash1] = emb1

            # compute embedding for file2
            file2 = trial['file2']
            hash2 = self.get_hash(file2)
            if hash2 in cache:
                emb2 = cache[hash2]
            else:
                emb2 = self.get_embedding(file2, pretrained)
                cache[hash2] = emb2

            # compare embeddings
            distance = cdist(emb1, emb2, metric=metric)[0, 0]
            y_pred.append(distance)

            y_true.append(trial['reference'])

        _, _, _, eer = det_curve(np.array(y_true), np.array(y_pred),
                                 distances=True)

        return {'metric': 'equal_error_rate',
                'minimize': True,
                'value': float(eer)}


    def _validate_epoch_diarization(self,
                                    epoch,
                                    validation_data,
                                    protocol=None,
                                    subset='development',
                                    device: Optional[torch.device] = None,
                                    batch_size: int = 32,
                                    n_jobs: int = 1,
                                    duration: float = None,
                                    step: float = 0.25,
                                    metric : str = None,
                                    purity : float = 0.9,
                                    **kwargs):

        # initialize embedding extraction
        pretrained = Pretrained(validate_dir=self.validate_dir_,
                                epoch=epoch,
                                duration=duration,
                                step=step,
                                batch_size=batch_size,
                                device=device)

        _protocol = get_protocol(protocol, progress=False,
                                preprocessors=self.preprocessors_)

        Z, t = dict(), dict()
        min_d, max_d = np.inf, -np.inf

        for current_file in getattr(_protocol, subset)():

            uri = get_unique_identifier(current_file)
            uem = get_annotated(current_file)
            reference = current_file['annotation']

            X_, t_ = [], []
            embedding = pretrained(current_file)
            for i, (turn, _) in enumerate(reference.itertracks()):

                # extract embedding for current speech turn. whenever possible,
                # only use those fully included in the speech turn ('strict')
                x_ = embedding.crop(turn, mode='strict')
                if len(x_) < 1:
                    x_ = embedding.crop(turn, mode='center')
                if len(x_) < 1:
                    x_ = embedding.crop(turn, mode='loose')
                if len(x_) < 1:
                    msg = (f'No embedding for {turn} in {uri:s}.')
                    raise ValueError(msg)

                # each speech turn is represented by its average embedding
                X_.append(np.mean(x_, axis=0))
                t_.append(turn)

            # apply hierarchical agglomerative clustering
            # all the way up to just one cluster (ie complete dendrogram)
            D = pdist(np.array(X_), metric=metric)
            min_d = min(np.min(D), min_d)
            max_d = max(np.max(D), max_d)

            Z[uri] = linkage(D, method='median')
            t[uri] = np.array(t_)

        def fun(threshold):

            _metric = DiarizationPurityCoverageFMeasure(weighted=False)

            for current_file in getattr(_protocol, subset)():

                uri = get_unique_identifier(current_file)
                uem = get_annotated(current_file)
                reference = current_file['annotation']

                clusters = fcluster(Z[uri], threshold, criterion='distance')

                hypothesis = Annotation(uri=uri)
                for (start_time, end_time), cluster in zip(t[uri], clusters):
                    hypothesis[Segment(start_time, end_time)] = cluster

                _ = _metric(reference, hypothesis, uem=uem)

            _purity, _coverage, _ = _metric.compute_metrics()

            return _purity, _coverage

        lower_threshold = min_d
        upper_threshold = max_d
        best_threshold = .5 * (lower_threshold + upper_threshold)
        best_coverage = 0.

        for _ in range(10):
            current_threshold = .5 * (lower_threshold + upper_threshold)
            _purity, _coverage = fun(current_threshold)

            if _purity < purity:
                upper_threshold = current_threshold
            else:
                lower_threshold = current_threshold
                if _coverage > best_coverage:
                    best_coverage = _coverage
                    best_threshold = current_threshold

        value = best_coverage if best_coverage else _purity - purity
        return {'metric': f'coverage@{purity:.2f}purity',
                'minimize': False,
                'value': float(value)}
