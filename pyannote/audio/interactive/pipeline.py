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


from typing import Text, Union, Tuple, List, Iterator, Dict
from pathlib import Path

import numpy as np
from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Uniform

from pyannote.audio.features.wrapper import Wrapper
from pyannote.audio.features import RawAudio
from pyannote.audio.features.utils import get_audio_duration

from pyannote.database.protocol.protocol import ProtocolFile
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Segment
from pyannote.core import Timeline
from pyannote.core import Annotation
from pyannote.core import SlidingWindow
from pyannote.core import SlidingWindowFeature
from pyannote.audio.utils.signal import Binarize
from pyannote.core.utils.hierarchy import pool
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform



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
            sad_scores: SlidingWindowFeature = self.sad(current_file)
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

    def __call__(self, current_file: ProtocolFile) -> Annotation:
        """Apply speaker diarization

        Parameters
        ----------
        current_file : ProtocolFile
            Protocol file.

        Returns
        -------
        diarization : Annotation
            Speaker diarization result.
        """

        if "duration" not in current_file:
            current_file["duration"] = get_audio_duration(current_file)

        # in "interactive annotation" mode, pipeline hyper-parameters are fixed.
        # therefore, there is no need to recompute embeddings every time a file
        # is processed: they can be passed with the file directly.
        if "embedding" in current_file:
            embedding: SlidingWindowFeature = current_file["embedding"]

        # in "pipeline optimization" mode, pipeline hyper-parameters are different
        # every time a file is processed: embeddings must be recomputed
        else:
            embedding: SlidingWindowFeature = self.compute_embedding(current_file)

        # in "interactive annotation" mode, there is no need to recompute speech
        # regions every time a file is processed: they can be passed with the
        # file directly
        if "speech" in current_file:
            speech: Timeline = current_file["speech"]

        # in "pipeline optimization" mode, pipeline hyper-parameters are different
        # every time a file is processed: speech regions must be recomputed
        else:
            speech: Timeline = self.compute_speech(current_file)

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

        strict_indices = np.where(segment_assignment > 0)[0]
        if len(strict_indices) < 2:
            cluster_assignment[strict_indices] = 1

        else:
            dendrogram = pool(embedding[strict_indices], metric="cosine")
            clusters = fcluster(dendrogram, self.emb_threshold, criterion="distance")
            for i, k in zip(strict_indices, clusters):
                cluster_assignment[i] = k

        loose_indices = np.where(segment_assignment < 0)[0]
        if len(strict_indices) == 0:
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
            # TODO. try distance to average instead
            distance = cdist(
                embedding[strict_indices], embedding[loose_indices], metric="cosine"
            )
            nearest_neighbor = np.argmin(distance, axis=0)
            for loose_index, nn in zip(loose_indices, nearest_neighbor):
                strict_index = strict_indices[nn]
                cluster_assignment[loose_index] = cluster_assignment[strict_index]

        # convert cluster assignment to pyannote.core.Annotation
        # (make sure to keep speech regions unchanged)
        hypothesis = Annotation(uri=current_file.get("uri", None))
        for s, segment in enumerate(speech):

            indices = np.where(segment_assignment == s + 1)[0]
            if len(indices) == 0:
                indices = np.where(segment_assignment == - (s + 1))[0]
                if len(indices) == 0:
                    continue

            clusters = cluster_assignment[indices]

            start, k = segment.start, clusters[0]
            change_point = np.diff(clusters) != 0
            for i, new_k in zip(indices[1:][change_point], clusters[1:][change_point]):
                end = embedding.sliding_window[i].middle
                hypothesis[Segment(start, end)] = k
                start = end
                k = new_k
            hypothesis[Segment(start, segment.end)] = k

        return hypothesis.support()

    def get_metric(self) -> DiarizationErrorRate:
        return DiarizationErrorRate(collar=0.0, skip_overlap=False)
