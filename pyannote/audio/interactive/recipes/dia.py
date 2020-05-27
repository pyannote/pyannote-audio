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

# types
from pathlib import Path
from typing import Text, Dict, List, Tuple, Iterable
from pyannote.core import Segment

# pyannote.audio
from ..pipeline import InteractiveDiarization
from ..pipeline import PRETRAINED_PARAMS
from ..pipeline import time2index
from ..pipeline import index2index
from pyannote.audio.features import RawAudio
from pyannote.audio.features.utils import get_audio_duration

# prodigy
import prodigy
from prodigy.components.loaders import Audio
from prodigy.components.db import connect
from .utils import SAMPLE_RATE
from .utils import remove_audio_before_db
from .utils import normalize
from .utils import to_base64
from .utils import to_audio_spans

# clustering
CONSTRAINT = Tuple[float, float]
CONSTRAINTS = List[CONSTRAINT]
from pyannote.core.utils.hierarchy import propagate_constraints
from pyannote.core.utils.hierarchy import pool
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

# others
import numpy as np
from copy import deepcopy
from .sad import load_sad_manual
from itertools import filterfalse


class DiaRecipeHelper:
    def __init__(self, pipeline: InteractiveDiarization, dataset: Text, source: Path):
        self.pipeline = pipeline
        self.dataset = dataset
        self.source = source

    def load_dia_binary(self, path: Text):
        """Load existing examples as constraints for diarization

        This will set (or overwrite) the following attributes and return them
            * cannot_link_time
            * must_link_time
            * dont_know_time

        Parameters
        ----------
        path : Text
            Only load examples for this file.
        """

        db = connect()

        examples = [
            eg
            for eg in db.get_dataset(self.dataset)
            if eg["recipe"] == "pyannote.dia.binary" and eg["path"] == path
        ]

        cannot_link: CONSTRAINTS = [
            (eg["t1"], eg["t2"]) for eg in examples if eg["answer"] == "reject"
        ]
        must_link: CONSTRAINTS = [
            (eg["t1"], eg["t2"]) for eg in examples if eg["answer"] == "accept"
        ]
        dont_know: CONSTRAINTS = [
            (eg["t1"], eg["t2"])
            for eg in examples
            if eg["answer"] not in ["accept", "reject"]
        ]

        if len(cannot_link) > 0:
            prodigy.log(
                f"RECIPE: {path}: init: {len(cannot_link)} cannot link constraints"
            )
        if len(must_link) > 0:
            prodigy.log(f"RECIPE: {path}: init: {len(must_link)} must link constraints")

        # expand list of "cannot link" constraints thanks to the following rule
        # (u != v) & (v == w) ==> u != w
        cannot_link = propagate_constraints(cannot_link, must_link)

        self.cannot_link_time = cannot_link
        self.must_link_time = must_link
        self.dont_know_time = dont_know

    def dia_binary_stream(self) -> Iterable[Dict]:

        raw_audio = RawAudio(sample_rate=SAMPLE_RATE, mono=True)

        for audio_source in Audio(self.source):

            path = audio_source["path"]
            text = audio_source["text"]

            # load speech/non-speech annotations (from pyannote.sad.manual recipe)
            file = load_sad_manual(self.dataset, path)
            speech = file["speech"]

            if not speech:
                prodigy.log(f"RECIPE: {path}: skip: no annotated speech")
                continue

            # load existing same/different annotations (from this very recipe)
            self.load_dia_binary(path)

            # extract speaker embedding
            embedding = self.pipeline.compute_embedding(file)

            # number of consecutive steps with overlap
            window = embedding.sliding_window
            n_steps = int(np.ceil(window.duration / window.step))

            # find clean embedding (i.e. those fully included in speech regions)
            assignment = self.pipeline.get_segment_assignment(embedding, speech)
            clean = assignment > 0
            clean_embedding = embedding[clean]
            if len(clean_embedding) < 2:
                prodigy.log(f"RECIPE: {path}: skip: not enough speech")
                continue

            # conversion from "clean-only" index base to "all" index base (used later)
            clean2all = index2index(None, clean, reverse=True, return_mapping=True)

            done_with_current_file = False
            while not done_with_current_file:

                # IMPROVE do not recompute if no new constraint since last time

                # filter and convert time-based constraints in whole file referential
                # to index-based constraints in clean-only embeddings referential
                cannot_link = index2index(
                    time2index(self.cannot_link_time, window), clean
                )
                must_link = index2index(time2index(self.must_link_time, window), clean)

                dendrogram = pool(
                    clean_embedding,
                    metric="cosine",
                    cannot_link=cannot_link if cannot_link else None,
                    must_link=must_link if must_link else None,
                    must_link_method="propagate",
                )

                # # iterate from dendrogram top to bottom
                # iterations = iter(range(len(dendrogram) - 1, 0, -1))

                # iterate from merging step whose distance is the most similar
                # to the "optimal" threshold and progressively wander away from it
                iterations = filterfalse(
                    lambda i: i < 1,
                    iter(
                        np.argsort(
                            np.abs(self.pipeline.emb_threshold - dendrogram[:, 2])
                        )
                    ),
                )

                # IDEA stop annotating early once the current distance is much
                # smaller/greater than the "optimal" and we can be sure that all
                # further iterations are easy to decide.

                while True:

                    try:
                        i = next(iterations)
                    except StopIteration as e:
                        done_with_current_file = True
                        break

                    distance = dendrogram[i, 2]

                    # if distance is infinite, this is a fake clustering step
                    # prevented by a "cannot link" constraint.
                    # see pyannote.core.hierarchy.pool for details
                    if distance == np.infty:
                        prodigy.log(f"RECIPE: {path}: depth {i}: skip: cannot link")
                        continue

                    # find clusters k1 and k2 that were merged at iteration i
                    current = fcluster(
                        dendrogram, dendrogram[i, 2], criterion="distance"
                    )
                    previous = fcluster(
                        dendrogram, dendrogram[i - 1, 2], criterion="distance",
                    )
                    n_current, n_previous = max(current), max(previous)

                    # TODO handle these corner cases better
                    if n_current >= n_previous or n_previous - n_current > 1:
                        prodigy.log(f"RECIPE: {path}: depth {i}: skip: corner case")
                        continue
                    C = np.zeros((n_current, n_previous))
                    for k_current, k_previous in zip(current, previous):
                        C[k_current - 1, k_previous - 1] += 1
                    k1, k2 = (
                        np.where(C[int(np.where(np.sum(C > 0, axis=1) == 2)[0])] > 0)[0]
                        + 1
                    )

                    # find indices of embeddings fully included in clusters k1 and k2
                    neighbors1 = np.convolve(previous == k1, [1] * n_steps, mode="same")
                    indices1 = np.where(neighbors1 == n_steps)[0]
                    # if indices1.size == 0:
                    #     indices1 = np.where(neighbors1 == np.max(neighbors1))[0]

                    neighbors2 = np.convolve(previous == k2, [1] * n_steps, mode="same")
                    indices2 = np.where(neighbors2 == n_steps)[0]
                    # if indices2.size == 0:
                    #     indices2 = np.where(neighbors2 == np.max(neighbors2))[0]

                    if indices1.size == 0 or indices2.size == 0:
                        prodigy.log(
                            f"RECIPE: {path}: depth {i}: skip: too short segments"
                        )
                        continue

                    # find centroids of clusters k1 and k2
                    i1 = indices1[
                        np.argmin(
                            np.mean(
                                squareform(
                                    pdist(clean_embedding[indices1], metric="cosine")
                                ),
                                axis=1,
                            )
                        )
                    ]

                    i2 = indices2[
                        np.argmin(
                            np.mean(
                                squareform(
                                    pdist(clean_embedding[indices2], metric="cosine")
                                ),
                                axis=1,
                            )
                        )
                    ]

                    i1, i2 = sorted([i1, i2])
                    distance = cdist(
                        clean_embedding[np.newaxis, i1],
                        clean_embedding[np.newaxis, i2],
                        metric="cosine",
                    )[0, 0]

                    segment1 = window[clean2all[i1]]
                    t1 = segment1.middle
                    segment2 = window[clean2all[i2]]
                    t2 = segment2.middle

                    # did the human in the loop already provide feedback on this pair of segments?
                    pair = (t1, t2)

                    if (
                        pair in self.cannot_link_time
                        or pair in self.must_link_time
                        or pair in self.dont_know_time
                    ):
                        # do not annotate the same pair twice
                        prodigy.log(f"RECIPE: {path}: depth {i}: skip: exists")
                        continue

                    prodigy.log(f"RECIPE: {path}: depth {i}: annotate")

                    task_text = f"{text} t={t1:.1f}s vs. t={t2:.1f}s"

                    waveform1 = normalize(raw_audio.crop(file, segment1))
                    waveform2 = normalize(raw_audio.crop(file, segment2))
                    task_audio = to_base64(
                        np.vstack([waveform1, waveform2]), sample_rate=SAMPLE_RATE
                    )

                    task_audio_spans = [
                        {"start": 0.0, "end": segment1.duration, "label": "SPEAKER",},
                        {
                            "start": segment1.duration,
                            "end": segment1.duration + segment2.duration,
                            "label": "SAME_SPEAKER",
                        },
                    ]

                    yield {
                        "path": path,
                        "text": task_text,
                        "audio": task_audio,
                        "audio_spans": task_audio_spans,
                        "t1": t1,
                        "t2": t2,
                        "meta": {
                            "t1": f"{t1:.1f}s",
                            "t2": f"{t2:.1f}s",
                            "file": text,
                            "distance": f"{distance:.2f}",
                        },
                        "recipe": "pyannote.dia.binary",
                    }

                    # at that point, "dia_binary_update" is called. hence,
                    # we exit the loop because the dendrogram needs to be updated
                    break

    def dia_binary_update(self, examples: List[Dict]) -> List[Dict]:

        needs_update = False

        for eg in examples:

            t1, t2 = eg["t1"], eg["t2"]

            if eg["answer"] == "accept":
                self.must_link_time.append((t1, t2))
                needs_update = True
                prodigy.log(f"RECIPE: new constraint: +1 must link")

            elif eg["answer"] == "reject":
                self.cannot_link_time.append((t1, t2))
                needs_update = True
                prodigy.log(f"RECIPE: new constraint: +1 cannot link")

            else:
                self.dont_know_time.append((t1, t2))
                prodigy.log(f"RECIPE: new constraint: skip")

        # expand list of "cannot link" constraints thanks to the following rule
        # (u != v) & (v == w) ==> u != w
        if needs_update:
            num_cannot = len(self.cannot_link_time)
            self.cannot_link_time = propagate_constraints(
                self.cannot_link_time, self.must_link_time
            )
            new_num_cannot = len(self.cannot_link_time)
            if new_num_cannot > num_cannot:
                prodigy.log(
                    f"RECIPE: propagate constraint: +{new_num_cannot - num_cannot} cannot link"
                )

    def dia_manual_stream(self) -> Iterable[Dict]:

        for audio_source in Audio(self.source):

            path = audio_source["path"]
            text = audio_source["text"]

            # load speech/non-speech annotations (from pyannote.sad.manual recipe)
            file = load_sad_manual(self.dataset, path)
            manual_speech = file["speech"]
            annotated = file["annotated"]

            # use manual speech/non-speech annotation where available,
            # and automatic speech/non-speech else where
            duration = get_audio_duration(file)
            file_extent = Segment(0, duration)
            non_annotated = annotated.gaps(file_extent)
            if non_annotated:
                automatic_speech = self.pipeline.compute_speech(file)
                file["speech"] = automatic_speech.crop(non_annotated).update(
                    manual_speech
                )

            # load existing same/different annotations (from pyannote.dia.binary recipe)
            self.load_dia_binary(path)

            # apply speaker diarization pipeline using same/different speaker
            # binary annotation as must link/cannot link constraints
            hypothesis = self.pipeline(
                file, cannot_link=self.cannot_link_time, must_link=self.must_link_time
            )

            # rename 9 most talkative speakers to {SPEAKER_1, ..., SPEAKER_9}
            # and remaining speakers as OTHER
            mapping = {
                label: f"SPEAKER_{s+1}" if s < 9 else "OTHER"
                for s, (label, duration) in enumerate(hypothesis.chart())
            }
            hypothesis = hypothesis.rename_labels(mapping=mapping)

            audio_spans = to_audio_spans(hypothesis)
            audio_source["audio_spans"] = audio_spans
            audio_source["audio_spans_original"] = deepcopy(audio_spans)
            audio_source["recipe"] = "pyannote.dia.manual"

            yield audio_source


@prodigy.recipe(
    "pyannote.dia.binary",
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=("Directory containing audio files to annotate", "positional", None, Path),
)
def dia_binary(dataset: Text, source: Path) -> Dict:

    pipeline = InteractiveDiarization().instantiate(PRETRAINED_PARAMS)
    helper = DiaRecipeHelper(pipeline, dataset, source)

    return {
        "dataset": dataset,
        "view_id": "audio",
        "stream": helper.dia_binary_stream(),
        "update": helper.dia_binary_update,
        "before_db": remove_audio_before_db,
        "config": {
            "audio_autoplay": True,
            "audio_loop": True,
            "show_audio_minimap": False,
            "audio_bar_width": 3,
            "audio_bar_height": 1,
            "labels": ["SPEAKER", "SAME_SPEAKER"],
            "batch_size": 1,
            "instant_submit": True,
        },
    }


@prodigy.recipe(
    "pyannote.dia.manual",
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=("Directory containing audio files to annotate", "positional", None, Path),
)
def dia_manual(dataset: Text, source: Path) -> Dict:

    pipeline = InteractiveDiarization().instantiate(PRETRAINED_PARAMS)
    helper = DiaRecipeHelper(pipeline, dataset, source)

    return {
        "dataset": dataset,
        "view_id": "audio_manual",
        "stream": helper.dia_manual_stream(),
        "before_db": remove_audio_before_db,
        "config": {
            "audio_autoplay": True,
            "audio_loop": True,
            "show_audio_minimap": True,
            "audio_bar_width": 3,
            "audio_bar_height": 1,
            "labels": [f"SPEAKER_{i+1}" for i in range(10)] + ["OTHER"],
            "batch_size": 1,
        },
    }
