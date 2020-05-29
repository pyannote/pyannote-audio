#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 CNRS

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


from typing import Text, Union, Tuple, List
from pathlib import Path
from pyannote.core import (
    Segment,
    SlidingWindow,
    Timeline,
    Annotation,
    SlidingWindowFeature,
)
from pyannote.database.protocol.protocol import ProtocolFile

import numpy as np

from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Uniform

from pyannote.audio.features.wrapper import Wrapper
from pyannote.audio.features.utils import get_audio_duration

from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.detection import DetectionErrorRate

from pyannote.audio.utils.signal import Binarize

from pyannote.core.utils.hierarchy import pool
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist


from .utils import time2index
from .utils import index2index

# Hyper-parameters tuned to minimize diarization error rate
# on one third of DIHARD II training set
PRETRAINED_PARAMS = {
    "emb_duration": 1.7657045140297274,
    "emb_step_ratio": 0.20414598809353782,
    "emb_threshold": 0.5274911675340328,
    "sad_min_duration_off": 0.13583405625051126,
    "sad_min_duration_on": 0.0014190874731107286,
    "sad_threshold_off": 0.7878607185085043,
    "sad_threshold_on": 0.5940560764213958,
}


class InteractiveDiarization(Pipeline):
    """Interactive diarization pipeline

    Parameters
    ----------
    sad : str or Path, optional
        Pretrained speech activity detection model. Defaults to "sad".
    emb : str or Path, optional
        Pretrained speaker embedding model. Defaults to "emb".
    batch_size : int, optional
        Batch size.
    only_sad : bool, optional
        Set to True if you only care about speech activity detection.

    Hyper-parameters
    ----------------
    sad_threshold_on, sad_threshold_off : float
        Onset/offset speech activity detection thresholds.
    sad_min_duration_on, sad_min_duration_off : float
        Minimum duration of speech/non-speech regions.
    emb_duration, emb_step_ratio : float
        Sliding window used for embedding extraction.
    emb_threshold : float
        Distance threshold used as stopping criterion for hierarchical
        agglomerative clustering.
    """

    def __init__(
        self,
        sad: Union[Text, Path] = {"sad": {"duration": 2.0, "step": 0.1}},
        emb: Union[Text, Path] = "emb",
        batch_size: int = None,
        only_sad: bool = False,
    ):

        super().__init__()

        self.sad = Wrapper(sad)
        if batch_size is not None:
            self.sad.batch_size = batch_size
        self.sad_speech_index_ = self.sad.classes.index("speech")

        self.sad_threshold_on = Uniform(0.0, 1.0)
        self.sad_threshold_off = Uniform(0.0, 1.0)
        self.sad_min_duration_on = Uniform(0.0, 0.5)
        self.sad_min_duration_off = Uniform(0.0, 0.5)

        self.only_sad = only_sad
        if self.only_sad:
            return

        self.emb = Wrapper(emb)
        if batch_size is not None:
            self.emb.batch_size = batch_size

        max_duration = self.emb.duration
        min_duration = getattr(self.emb, "min_duration", 0.25 * max_duration)
        self.emb_duration = Uniform(min_duration, max_duration)
        self.emb_step_ratio = Uniform(0.1, 1.0)
        self.emb_threshold = Uniform(0.0, 2.0)

    def initialize(self):
        """Initialize pipeline internals with current hyper-parameter values"""

        self.sad_binarize_ = Binarize(
            onset=self.sad_threshold_on,
            offset=self.sad_threshold_off,
            min_duration_on=self.sad_min_duration_on,
            min_duration_off=self.sad_min_duration_off,
        )

        if not self.only_sad:
            # embeddings will be extracted with a sliding window
            # of "emb_duration" duration and "emb_step_ratio x emb_duration" step.
            self.emb.duration = self.emb_duration
            self.emb.step = self.emb_step_ratio

    def compute_speech(self, current_file: ProtocolFile) -> Timeline:
        """Apply speech activity detection

        Parameters
        ----------
        current_file : ProtocolFile
            Protocol file.

        Returns
        -------
        speech : Timeline
            Speech activity detection result.
        """

        # speech activity detection
        if "sad_scores" in current_file:
            sad_scores: SlidingWindowFeature = current_file["sad_scores"]
        else:
            sad_scores = self.sad(current_file)
            if np.nanmean(sad_scores) < 0:
                sad_scores = np.exp(sad_scores)
            current_file["sad_scores"] = sad_scores

        speech: Timeline = self.sad_binarize_.apply(
            sad_scores, dimension=self.sad_speech_index_
        )

        return speech

    def compute_embedding(self, current_file: ProtocolFile) -> SlidingWindowFeature:
        """Extract speaker embedding

        Parameters
        ----------
        current_file : ProtocolFile
            Protocol file

        Returns
        -------
        embedding : SlidingWindowFeature
            Speaker embedding.
        """

        return self.emb(current_file)

    def get_segment_assignment(
        self, embedding: SlidingWindowFeature, speech: Timeline
    ) -> np.ndarray:
        """Get segment assignment

        Parameters
        ----------
        embedding : SlidingWindowFeature
            Embeddings.
        speech : Timeline
            Speech regions.

        Returns
        -------
        assignment : (num_embedding, ) np.ndarray
            * assignment[i] = s with s > 0 means that ith embedding is strictly
            contained in (1-based) sth segment.
            * assignment[i] = s with s < 0 means that more than half of ith
            embedding is part of (1-based) sth segment.
            * assignment[i] = 0 means that none of the above is true.
        """

        assignment: np.ndarray = np.zeros((len(embedding),), dtype=np.int32)

        for s, segment in enumerate(speech):
            indices = embedding.sliding_window.crop(segment, mode="strict")
            if len(indices) > 0:
                strict = 1
            else:
                strict = -1
                indices = embedding.sliding_window.crop(segment, mode="center")
            for i in indices:
                if i < 0 or i >= len(embedding):
                    continue
                assignment[i] = strict * (s + 1)

        return assignment

    def __call__(
        self,
        current_file: ProtocolFile,
        cannot_link: List[Tuple[float, float]] = None,
        must_link: List[Tuple[float, float]] = None,
    ) -> Annotation:
        """Apply speaker diarization

        Parameters
        ----------
        current_file : ProtocolFile
            Protocol file.
        cannot_link :
            List of time-based "cannot link" constraints.
        must_link :
            List of time-based "must link" constraints.

        Returns
        -------
        diarization : Annotation
            Speaker diarization result.
        """

        if cannot_link is None:
            cannot_link = []
        if must_link is None:
            must_link = []

        if "duration" not in current_file:
            current_file["duration"] = get_audio_duration(current_file)

        # in "interactive annotation" mode, there is no need to recompute speech
        # regions every time a file is processed: they can be passed with the
        # file directly
        if "speech" in current_file:
            speech: Timeline = current_file["speech"]

        # in "pipeline optimization" mode, pipeline hyper-parameters are different
        # every time a file is processed: speech regions must be recomputed
        else:
            speech = self.compute_speech(current_file)

        if self.only_sad:
            return speech.to_annotation(generator=iter(lambda: "SPEECH", None))

        # in "interactive annotation" mode, pipeline hyper-parameters are fixed.
        # therefore, there is no need to recompute embeddings every time a file
        # is processed: they can be passed with the file directly.
        if "embedding" in current_file:
            embedding: SlidingWindowFeature = current_file["embedding"]

        # in "pipeline optimization" mode, pipeline hyper-parameters are different
        # every time a file is processed: embeddings must be recomputed
        else:
            embedding = self.compute_embedding(current_file)

        window: SlidingWindow = embedding.sliding_window

        # segment_assignment[i] = s with s > 0 means that ith embedding is
        #       strictly contained in (1-based) sth segment.
        # segment_assignment[i] = s with s < 0 means that more than half of ith
        #       embedding is part of (1-based) sth segment.
        # segment_assignment[i] = 0 means that none of the above is true.
        segment_assignment: np.ndarray = self.get_segment_assignment(embedding, speech)

        # cluster_assignment[i] = k (k > 0) means that the ith embedding belongs
        #                           to kth cluster
        # cluster_assignment[i] = 0 when segment_assignment[i] = 0
        cluster_assignment: np.ndarray = np.zeros((len(embedding),), dtype=np.int32)

        clean = segment_assignment > 0
        noisy = segment_assignment < 0
        clean_indices = np.where(clean)[0]
        if len(clean_indices) < 2:
            cluster_assignment[clean_indices] = 1

        else:

            # convert time-based constraints to index-based constraints
            cannot_link = index2index(time2index(cannot_link, window), clean)
            must_link = index2index(time2index(must_link, window), clean)

            dendrogram = pool(
                embedding[clean_indices],
                metric="cosine",
                cannot_link=cannot_link,
                must_link=must_link,
                must_link_method="propagate",
            )
            clusters = fcluster(dendrogram, self.emb_threshold, criterion="distance")
            for i, k in zip(clean_indices, clusters):
                cluster_assignment[i] = k

        loose_indices = np.where(noisy)[0]
        if len(clean_indices) == 0:
            if len(loose_indices) < 2:
                clusters = [1] * len(loose_indices)
            else:
                dendrogram = pool(embedding[loose_indices], metric="cosine")
                clusters = fcluster(
                    dendrogram, self.emb_threshold, criterion="distance"
                )
            for i, k in zip(loose_indices, clusters):
                cluster_assignment[i] = k

        else:
            # NEAREST NEIGHBOR
            distance = cdist(
                embedding[clean_indices], embedding[loose_indices], metric="cosine"
            )
            nearest_neighbor = np.argmin(distance, axis=0)
            for loose_index, nn in zip(loose_indices, nearest_neighbor):
                strict_index = clean_indices[nn]
                cluster_assignment[loose_index] = cluster_assignment[strict_index]

            # # NEAREST CLUSTER
            # centroid = np.vstack(
            #     [
            #         np.mean(embedding[cluster_assignment == k], axis=0)
            #         for k in np.unique(clusters)
            #     ]
            # )
            # distance = cdist(centroid, embedding[loose_indices], metric="cosine")
            # cluster_assignment[loose_indices] = np.argmin(distance, axis=0) + 1

        # convert cluster assignment to pyannote.core.Annotation
        # (make sure to keep speech regions unchanged)
        hypothesis = Annotation(uri=current_file.get("uri", None))
        for s, segment in enumerate(speech):

            indices = np.where(segment_assignment == s + 1)[0]
            if len(indices) == 0:
                indices = np.where(segment_assignment == -(s + 1))[0]
                if len(indices) == 0:
                    continue

            clusters = cluster_assignment[indices]

            start, k = segment.start, clusters[0]
            change_point = np.diff(clusters) != 0
            for i, new_k in zip(indices[1:][change_point], clusters[1:][change_point]):
                end = window[i].middle + 0.5 * window.step
                hypothesis[Segment(start, end)] = k
                start = end
                k = new_k
            hypothesis[Segment(start, segment.end)] = k

        return hypothesis.support()

    def get_metric(self) -> Union[DetectionErrorRate, DiarizationErrorRate]:
        if self.only_sad:
            return DetectionErrorRate(collar=0.0)
        else:
            return DiarizationErrorRate(collar=0.0, skip_overlap=False)
