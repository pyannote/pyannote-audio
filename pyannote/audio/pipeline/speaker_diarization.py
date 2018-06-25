#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2018 CNRS

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
import sklearn.cluster
from pathlib import Path
from scipy.spatial.distance import squareform
from pyannote.audio.embedding.utils import l2_normalize, pdist

from pyannote.audio.features import Precomputed
from pyannote.audio.signal import Binarize, Peak

from pyannote.core import Annotation
from pyannote.core import SlidingWindowFeature
from pyannote.database import get_annotated

from pyannote.audio.embedding.clustering import HierarchicalPoolingClustering
from pyannote.metrics.diarization import GreedyDiarizationErrorRate

from .base import Pipeline
import chocolate


class NeuralSegmentation(Pipeline):

    def __init__(self, sad=None, scd=None, **kwargs):
        super().__init__()
        self.sad = Path(sad).expanduser().resolve(strict=True)
        self.scd = Path(scd).expanduser().resolve(strict=True)
        self.with_params(**kwargs)

    def get_tune_space(self):
        return {
            'sad_onset': chocolate.uniform(0., 1.),
            'sad_offset': chocolate.uniform(0., 1.),
            'scd_alpha': chocolate.uniform(0., 1.),
            'scd_min_duration': chocolate.uniform(0., 5.),
        }

    def get_tune_metric(self):
        raise NotImplementedError()

    def with_params(self, sad_onset=0.7, sad_offset=0.7,
                    scd_alpha=0.5, scd_min_duration=1.):

        # initialize speech activity detection
        self.sad_ = Precomputed(self.sad)
        self.sad_onset = sad_onset
        self.sad_offset = sad_offset
        self.sad_binarize_ = Binarize(onset=sad_onset, offset=sad_offset)

        # initialize speaker change detection
        self.scd_ = Precomputed(self.scd)
        self.scd_alpha = scd_alpha
        self.scd_min_duration = scd_min_duration
        self.scd_peak_ = Peak(alpha=scd_alpha, min_duration=scd_min_duration)

        return self

    def apply(self, current_file):

        # Speech Activity Detection

        # get raw SAD scores
        soft_sad = self.sad_(current_file)

        # check once and for all whether SAD scores are log-scaled
        if not hasattr(self, 'sad_log_scale_'):
            if np.nanmean(soft_sad.data) < 0:
                self.sad_log_scale_ = True
            else:
                self.sad_log_scale_ = False

        # get SAD probability
        prob_sad = np.exp(soft_sad.data) if self.sad_log_scale_ \
                   else soft_sad.data

        # support both non-speech/speech & non-speech/single/overlap
        prob_sad = 1. - prob_sad[:, 0]
        prob_sad = SlidingWindowFeature(prob_sad, soft_sad.sliding_window)

        # binarization
        hard_sad = self.sad_binarize_.apply(prob_sad)

        # Speaker Change Detection

        # get raw SCD scores
        soft_scd = self.scd_(current_file)

        # check once and for all whether SCD scores are log-scaled
        if not hasattr(self, 'scd_log_scale_'):
            if np.nanmean(soft_scd.data) < 0:
                self.scd_log_scale_ = True
            else:
                self.scd_log_scale_ = False

        # get SCD probability
        prob_scd = np.exp(soft_scd.data) if self.scd_log_scale_ \
                   else soft_scd.data

        # take the final dimension
        # (in order to support both classification and regression scores)
        prob_scd = prob_scd[:, -1]
        prob_scd = SlidingWindowFeature(prob_scd, soft_scd.sliding_window)

        # peak detection
        hard_scd = self.scd_peak_.apply(prob_scd)

        speech_turns = hard_scd.crop(hard_sad)

        # only process the annotated part
        speech_turns = speech_turns.crop(get_annotated(current_file))

        return speech_turns


class Yin2018(NeuralSegmentation):
    """Speaker diarization pipeline introduced in Yin et al., 2018

        Ruiqing Yin, Hervé Bredin, and Claude Barras. "Neural speech turn
        segmentation and affinity propagation for speaker diarization".
        Interspeech 2018.

    Parameters
    ----------
    sad : str
        Path to precomputed speech activity detection scores.
    scd : str
        Path to precomputed speaker change detection scores.
    emb : str
        Path to precomputed embeddings.
    metric : {'euclidean', 'cosine', 'angular'}, optional
        Metric used for comparing embeddings. Defaults to 'angular'.
    """

    def __init__(self, sad=None, scd=None, emb=None, metric='angular', **kwargs):
        super().__init__(sad=sad, scd=scd)
        self.emb = Path(emb).expanduser().resolve(strict=True)
        self.metric = metric
        self.with_params(**kwargs)

    def get_tune_space(self):

        base_space = super().get_tune_space()
        space = {
            'cls_damping': chocolate.uniform(0.5, 1.),
            # FIXME: be smarter about this parameter
            'cls_preference': chocolate.uniform(-8.0, 0.0)}
        base_space.update(space)
        return base_space

    def get_tune_metric(self):

        def func(reference, hypothesis, uem=None):

            # heuristic to avoid wasting time computing DER
            # when the proposed solution is obviously wrong
            r_labels = reference.crop(uem).labels()
            h_labels = hypothesis.crop(uem).labels()
            if len(h_labels) > 100 * len(r_labels):
                return 1.

            metric = GreedyDiarizationErrorRate()
            return metric(reference, hypothesis, uem=uem)

        return func

    def with_params(self, sad_onset=0.7, sad_offset=0.7,
                    scd_alpha=0.5, scd_min_duration=1.,
                    cls_preference=-7.0,
                    cls_damping=0.8):

        # initialize speech activity detection and speaker change detection
        super().with_params(sad_onset=sad_onset, sad_offset=sad_offset,
                            scd_alpha=scd_alpha,
                            scd_min_duration=scd_min_duration)

        # initialize speech turn embedding
        self.emb_ = Precomputed(self.emb)

        # initialize clustering module
        self.cls_damping = cls_damping

        self.cls_preference = cls_preference
        # NOTE cls_preference could be a multiplicative factor of a default
        # affinity value (e.g. median affinity value)
        self.cls_ = sklearn.cluster.AffinityPropagation(
            damping=cls_damping, preference=cls_preference,
            affinity='precomputed', max_iter=200, convergence_iter=15)

        # sklearn documentation: Preferences for each point - points with
        # larger values of preferences are more likely to be chosen as
        # exemplars. The number of exemplars, ie of clusters, is influenced by
        # the input preferences value. If the preferences are not passed as
        # arguments, they will be set to the median of the input similarities.

        # NOTE one could set the preference value of each speech turn
        # according to their duration. longer speech turns are expected to
        # have more accurate embeddings, therefore should be prefered for
        # exemplars

        return self

    def apply(self, current_file):

        # initial segmentation
        speech_turns = super().apply(current_file)

        # initialize the hypothesized annotation
        hypothesis = Annotation(uri=current_file['uri'])
        if len(speech_turns) < 1:
            return hypothesis

        # get raw (sliding window) embeddings
        emb = self.emb_(current_file)

        # get one embedding per speech turn
        # FIXME don't l2_normalize for any metric
        fX = l2_normalize(np.vstack(
            [np.sum(emb.crop(t, mode='loose'), axis=0) for t in speech_turns]))

        # compute affinity matrix
        affinity = -squareform(pdist(fX, metric=self.metric))

        # apply clustering
        clusters = self.cls_.fit_predict(affinity)

        for speech_turn, cluster in zip(speech_turns, clusters):
            # HACK find why fit_predict returns NaN sometimes and fix it.
            cluster = -1 if np.isnan(cluster) else cluster
            hypothesis[speech_turn] = cluster

        return hypothesis


class HierachicalEmbeddingPooling(NeuralSegmentation):
    """

    Parameters
    ----------
    sad : str
        Path to precomputed speech activity detection scores.
    scd : str
        Path to precomputed speaker change detection scores.
    emb : str
        Path to precomputed embeddings.
    metric : {'euclidean', 'cosine', 'angular'}, optional
        Metric used for comparing embeddings. Defaults to 'angular'.
    """

    def __init__(self, sad=None, scd=None, emb=None, metric='angular', **kwargs):
        super().__init__(sad=sad, scd=scd)
        self.emb = Path(emb).expanduser().resolve(strict=True)
        self.metric = metric
        self.with_params(**kwargs)

    def get_tune_space(self):

        base_space = super().get_tune_space()
        space = {
            'cls_threshold': chocolate.uniform(0, 2.),
        base_space.update(space)
        return base_space

    def get_tune_metric(self):

        def func(reference, hypothesis, uem=None):

            # heuristic to avoid wasting time computing DER
            # when the proposed solution is obviously wrong
            r_labels = reference.crop(uem).labels()
            h_labels = hypothesis.crop(uem).labels()
            if len(h_labels) > 100 * len(r_labels):
                return 1.

            metric = GreedyDiarizationErrorRate()
            return metric(reference, hypothesis, uem=uem)

        return func

    def with_params(self, sad_onset=0.7, sad_offset=0.7,
                    scd_alpha=0.5, scd_min_duration=1.,
                    cls_threshold=0.8):

        # initialize speech activity detection and speaker change detection
        super().with_params(sad_onset=sad_onset, sad_offset=sad_offset,
                            scd_alpha=scd_alpha,
                            scd_min_duration=scd_min_duration)

        # initialize speech turn embedding
        self.emb_ = Precomputed(self.emb)

        # initialize clustering module
        self.cls_threshold = cls_threshold

        self.cls_ = HierarchicalPoolingClustering(metric=self.metric)

        return self

    def apply(self, current_file):

        # (inherited) initial segmentation
        seg = super().apply(current_file)

        # load raw embeddings
        emb = self.emb_(current_file)

        # hierarchical pooling clustering
        return self.cls_.fit(seg, emb).apply(self.cls_threshold)
