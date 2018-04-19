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
from scipy.spatial.distance import squareform
from pyannote.audio.embedding.utils import l2_normalize, pdist

from pyannote.audio.features import Precomputed
from pyannote.audio.signal import Binarize, Peak

from pyannote.core import Annotation
from pyannote.database import get_annotated

from pyannote.metrics.diarization import GreedyDiarizationErrorRate

from .base import Pipeline
import chocolate


# class SpeakerDiarization(object):
#     """Speaker diarization pipeline
#
#     1. Speech activity detection
#     2. Speaker change detection
#     3. Speech turn clustering
#     TODO. 4. Resegmentation
#
#     Parameters
#     ----------
#     sad : pyannote.audio.features.Precomputed
#     scd : pyannote.audio.features.Precomputed
#     emb : pyannote.audio.features.Precomputed
#     sad_onset, sad__offset : float, optional
#         See pyannote.audio.signal.Binarize. Default to 0.7 and 0.7.
#     sad__dimension : int, optional
#         See pyannote.audio.signal.Binarize.apply(). Defaults to 1.
#     scd_alpha, scd_min_duration : float, optional
#         See pyannote.audio.signal.Peak. Default to 0.5 and 1.
#     scd__dimension : int, optional
#         See pyannote.audio.signal.Peak.apply(). Defaults to 1.
#     cls__min_cluster_size, cls__min_samples, cls__metric : optional
#         See pyannote.audio.embedding.clustering.Clustering
#     normalize : boolean, optional
#         Normalize embeddings before clustering. Defaults to not normalize.
#     long_first : boolean, optional
#         Start by clustering longer speech turns, then associate each smaller
#         speech turns to the closest cluster. Defaults to cluster all speech
#         turns at once.
#     """
#
#     def __init__(self, sad, scd, emb,
#                  sad_onset=0.7, sad__offset=0.7, sad__dimension=1,
#                  scd_alpha=0.5, scd_min_duration=1., scd__dimension=1,
#                  cls__min_cluster_size=5, cls__min_samples=None,
#                  cls__metric='cosine', long_first=False, normalize=False):
#
#         super(SpeakerDiarization, self).__init__()
#
#         # speech activity detection hyper-parameters
#         self.sad = sad
#         self.sad_onset = sad_onset
#         self.sad__offset = sad__offset
#         self.sad__dimension = sad__dimension
#
#         # speaker change detection hyper-parameters
#         self.scd = scd
#         self.scd_alpha = scd_alpha
#         self.scd_min_duration = scd_min_duration
#         self.scd__dimension = scd__dimension
#
#         # embedding hyper-parameters
#         self.emb = emb
#
#         # clustering hyper-parameters
#         self.cls__min_cluster_size = cls__min_cluster_size
#         self.cls__min_samples = cls__min_samples
#         self.cls__metric = cls__metric
#         self.long_first = long_first
#         self.normalize = normalize
#
#         # initialize speech activity detection module
#         self.sad_binarize_ = Binarize(onset=self.sad_onset,
#                                       offset=self.sad__offset)
#
#         # initialize speaker change detection module
#         self.scd_peak_ = Peak(alpha=self.scd_alpha,
#                               min_duration=self.scd_min_duration,
#                               percentile=False)
#
#         # initialize clustering module
#         self.cls_ = Clustering(metric=self.cls__metric,
#                                min_cluster_size=self.cls__min_cluster_size,
#                                min_samples=self.cls__min_samples)
#
#     def __call__(self, current_file, annotated=False):
#         """Process current file
#
#         Parameters
#         ----------
#         current_file : dict
#         annotated : boolean, optional
#             Only process annotated region.
#
#         Return
#         ------
#         result : Annotation
#             Speaker diarization result.
#         """
#
#         # speech activity detection
#         soft_sad = self.sad(current_file)
#         hard_sad = self.sad_binarize_.apply(
#             soft_sad, dimension=self.sad__dimension)
#
#         # speaker change detection
#         soft_scd = self.scd(current_file)
#         hard_scd = self.scd_peak_.apply(
#             soft_scd, dimension=self.scd__dimension)
#
#         # speech turns
#         speech_turns = hard_scd.crop(hard_sad)
#
#         if annotated:
#             speech_turns = speech_turns.crop(
#                 get_annotated(current_file))
#
#         hypothesis = Annotation(uri=current_file['uri'])
#         if not speech_turns:
#             return hypothesis
#
#         # speech turns embedding
#         emb = self.emb(current_file)
#
#         # normalize speech turns embeddings
#         if self.normalize:
#             emb.data = l2_normalize(emb.data)
#
#         # long speech turns = those with at least one embedding fully included
#         # shrt speech turns = all the other speech turns
#         long_speech_turns, long_fX = [], []
#         shrt_speech_turns, shrt_fX = [], []
#
#         # when long_first == False (i.e. mode == 'center'), there can only be
#         # long speech turns (according to the above definition)
#         mode = 'strict' if self.long_first else 'center'
#         for speech_turn in speech_turns:
#             fX = emb.crop(speech_turn, mode=mode)
#             if len(fX):
#                 long_speech_turns.append(speech_turn)
#                 long_fX.append(np.sum(fX, axis=0))
#             else:
#                 shrt_speech_turns.append(speech_turn)
#                 fX = emb.crop(speech_turn, mode='center')
#                 shrt_fX.append(np.sum(fX, axis=0))
#
#         # stack long (and short) speech turns embedding into numpy arrays
#         long_fX = np.vstack(long_fX)
#         if shrt_speech_turns:
#             shrt_fX = np.vstack(shrt_fX)
#
#         # cluster long speech turns based on their normalized embeddings
#         # TODO: make sure it makes sense for embeddings trained with the
#         # euclidean distance and not normalized
#         long_clusters = self.cls_.apply(l2_normalize(long_fX))
#
#         # get a unique embedding for each cluster by summing the embedding of
#         # all the included speech turns
#         cluster_fX = np.vstack(
#             [np.sum(long_fX[long_clusters == i], axis=0)
#              for i in np.unique(long_clusters)])
#
#         # associate each short speech turn to the closest cluster
#         shrt_clusters = []
#         if shrt_speech_turns:
#             distances = cdist(l2_normalize(cluster_fX),
#                               l2_normalize(shrt_fX),
#                               metric=self.cls__metric)
#             shrt_clusters = np.argmin(distances, axis=0)
#
#         # build hypothesis from clustering results
#         speech_turns = long_speech_turns + shrt_speech_turns
#         clusters = list(long_clusters) + list(shrt_clusters)
#         for speech_turn, cluster in zip(speech_turns, clusters):
#             hypothesis[speech_turn] = cluster
#
#         return hypothesis


class SpeakerDiarization(Pipeline):
    """

    Parameters
    ----------
    sad : str
        Path to precomputed speech activity detection scores.
    scd : str
        Path to precomputed speaker change detection scores.
    emb : str
        Path to precomputed embeddings.
    metric :


    """

    def __init__(self, sad=None, scd=None, emb=None, metric='angular', **kwargs):
        super(SpeakerDiarization, self).__init__()
        self.sad = sad
        self.scd = scd
        self.emb = emb
        self.metric = metric
        self.with_params(**kwargs)

    def get_tune_space(self):

        return {
            'sad_onset': chocolate.uniform(0., 1.),
            'sad_offset': chocolate.uniform(0., 1.),
            'scd_alpha': chocolate.uniform(0., 1.),
            'scd_min_duration': chocolate.uniform(0., 5.),
            'cls_damping': chocolate.uniform(0.5, 1.),
            # FIXME: be smarter about this parameter
            'cls_preference': chocolate.uniform(-8.0, 0.0),
        }

    def get_tune_metric(self):
        return GreedyDiarizationErrorRate()

    def with_params(self, sad_onset=0.7, sad_offset=0.7,
                    scd_alpha=0.5, scd_min_duration=1.,
                    cls_preference=-7.0,
                    cls_damping=0.8):

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

    def __call__(self, current_file, annotated=False):

        # Speech Activity Detection

        # get raw SAD scores
        soft_sad = self.sad_(current_file)

        # check once and for all whether SAD scores are log-scaled
        if not hasattr(self, "sad_log_scale_"):
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
        if not hasattr(self, "scd_log_scale_"):
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

        # Initial Segmentation

        speech_turns = hard_scd.crop(hard_sad)

        # only process the annotated part
        if annotated:
            speech_turns = speech_turns.crop(get_annotated(current_file))

        # initialize the hypothesized annotation
        hypothesis = Annotation(uri=current_file['uri'])
        if len(speech_turns) < 1:
            return hypothesis

        # Speech Turn Clustering

        # get raw (sliding window) embeddings
        emb = self.emb_(current_file)

        # get one embedding per speech turn
        fX = l2_normalize(np.vstack(
            [np.sum(emb.crop(t, mode='loose'), axis=0) for t in speech_turns]))

        # compute affinity matrix
        affinity = -squareform(pdist(fX, metric=self.metric))

        # apply clustering
        clusters = self.cls_.fit_predict(affinity)

        for speech_turn, cluster in zip(speech_turns, clusters):
            hypothesis[speech_turn] = cluster

        return hypothesis

# class SpeakerDiarization(object):
#     """Speaker diarization pipeline
#
#     1. Speech activity detection
#     2. Speaker change detection
#     3. Speech turn clustering
#
#     Parameters
#     ----------
#     sad : pyannote.audio.features.Precomputed
#     scd : pyannote.audio.features.Precomputed
#     emb : pyannote.audio.features.Precomputed
#     sad_onset, sad__offset : float, optional
#         See pyannote.audio.signal.Binarize. Default to 0.7 and 0.7.
#     sad__dimension : int, optional
#         See pyannote.audio.signal.Binarize.apply(). Defaults to 1.
#     scd_alpha, scd_min_duration : float, optional
#         See pyannote.audio.signal.Peak. Default to 0.5 and 1.
#     scd__dimension : int, optional
#         See pyannote.audio.signal.Peak.apply(). Defaults to 1.
#     cls__metric, cls__threshold: optional
#         See pyannote.audio.embedding.clustering.EmbeddingClustering
#     """
#
#     def __init__(self, sad, scd, emb,
#                  sad_onset=0.7, sad__offset=0.7, sad__dimension=1,
#                  scd_alpha=0.5, scd_min_duration=1., scd__dimension=1,
#                  cls__metric='angular', cls__threshold=1.0):
#
#         super(SpeakerDiarization, self).__init__()
#
#         # speech activity detection hyper-parameters
#         self.sad = sad
#         self.sad_onset = sad_onset
#         self.sad__offset = sad__offset
#         self.sad__dimension = sad__dimension
#
#         # speaker change detection hyper-parameters
#         self.scd = scd
#         self.scd_alpha = scd_alpha
#         self.scd_min_duration = scd_min_duration
#         self.scd__dimension = scd__dimension
#
#         # embedding hyper-parameters
#         self.emb = emb
#
#         # clustering hyper-parameters
#         self.cls__metric = cls__metric
#         self.cls__threshold = cls__threshold
#
#         # initialize speech activity detection module
#         self.sad_binarize_ = Binarize(onset=self.sad_onset,
#                                       offset=self.sad__offset)
#
#         # initialize speaker change detection module
#         self.scd_peak_ = Peak(alpha=self.scd_alpha,
#                               min_duration=self.scd_min_duration,
#                               percentile=False)
#
#         # initialize clustering module
#         self.cls_ = EmbeddingClustering(metric=self.cls__metric)
#
#     def __call__(self, current_file, annotated=False):
#         """Process current file
#
#         Parameters
#         ----------
#         current_file : dict
#         annotated : boolean, optional
#             Only process annotated region.
#
#         Return
#         ------
#         result : Annotation
#             Speaker diarization result.
#         """
#
#         # speech activity detection
#         soft_sad = self.sad(current_file)
#         hard_sad = self.sad_binarize_.apply(
#             soft_sad, dimension=self.sad__dimension)
#
#         # speaker change detection
#         soft_scd = self.scd(current_file)
#         hard_scd = self.scd_peak_.apply(
#             soft_scd, dimension=self.scd__dimension)
#
#         # segmentation
#         speech_turns = hard_scd.crop(hard_sad)
#         segmentation = Annotation()
#         from s, speech_turn in enumerate(speech_turns):
#             segmentation[speech_turn] = s
#
#         if annotated:
#             segmentation = segmentation.crop(
#                 get_annotated(current_file))
#
#         # speech turns embedding
#         emb = self.emb(current_file)
#
#         # clustering
#         clustering = self.cls_.fit(segmentation, emb)
#         return clustering.apply(self.cls__threshold)
