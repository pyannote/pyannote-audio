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
from pathlib import Path
from typing import Optional

from pyannote.core import Annotation
from pyannote.pipeline import Pipeline
from pyannote.audio.features import Precomputed
from pyannote.pipeline.blocks.clustering import \
    HierarchicalAgglomerativeClustering
from pyannote.pipeline.blocks.clustering import AffinityPropagationClustering
from .utils import assert_string_labels


class SpeechTurnClustering(Pipeline):
    """Speech turn clustering

    Parameters
    ----------
    embedding : `Path`
        Path to precomputed embeddings.
    metric : {'euclidean', 'cosine', 'angular'}, optional
        Metric used for comparing embeddings. Defaults to 'cosine'.
    method : {'pool', 'affinity_propagation'}
    """

    def __init__(self, embedding: Optional[Path],
                       metric: Optional[str] = 'cosine',
                       method: Optional[str] = 'pool'):
        super().__init__()

        self.embedding = embedding
        self.precomputed_ = Precomputed(self.embedding)

        self.metric = metric
        self.method = method

        if self.method == 'affinity_propagation':
            self.clustering = AffinityPropagationClustering(
                metric=self.metric)

            # sklearn documentation: Preferences for each point - points with
            # larger values of preferences are more likely to be chosen as
            # exemplars. The number of exemplars, ie of clusters, is influenced by
            # the input preferences value. If the preferences are not passed as
            # arguments, they will be set to the median of the input similarities.

            # NOTE one could set the preference value of each speech turn
            # according to their duration. longer speech turns are expected to
            # have more accurate embeddings, therefore should be prefered for
            # exemplars

        else:
            self.clustering = HierarchicalAgglomerativeClustering(
                method=self.method, metric=self.metric)

    def __call__(self, current_file: dict,
                       speech_turns: Annotation) -> Annotation:
        """Apply speech turn clustering

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol.
        speech_turns : `Annotation`
            Speech turns. Should only contain `str` labels.

        Returns
        -------
        speech_turns : `pyannote.core.Annotation`
            Clustered speech turns.
        """

        assert_string_labels(speech_turns, 'speech_turns')

        embedding = self.precomputed_(current_file)

        labels = speech_turns.labels()
        X, clustered_labels, skipped_labels = [], [], []
        for l, label in enumerate(labels):

            timeline = speech_turns.label_timeline(label, copy=False)

            # be more and more permissive until we have
            # at least one embedding for current speech turn
            for mode in ['strict', 'center', 'loose']:
                x = embedding.crop(timeline, mode=mode)
                if len(x) > 0:
                    break

            # skip labels so small we don't have any embedding for it
            if len(x) < 1:
                skipped_labels.append(label)
                continue

            clustered_labels.append(label)
            X.append(np.mean(x, axis=0))

        # apply clustering of label embeddings
        clusters = self.clustering(np.vstack(X))

        # map each clustered label to its cluster (between 1 and N_CLUSTERS)
        mapping = {label: k for label, k in zip(clustered_labels, clusters)}

        # map each skipped label to its own cluster
        # (between -1 and -N_SKIPPED_LABELS)
        for l, label in enumerate(skipped_labels):
            mapping[skipped_labels] = -(l + 1)

        # do the actual mapping
        return speech_turns.rename_labels(mapping=mapping)
