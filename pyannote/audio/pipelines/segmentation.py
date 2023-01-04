# MIT License
#
# Copyright (c) 2018-2022 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Speaker segmentation pipeline"""

import math
from functools import partial
from typing import Callable, Optional, Text, Union

import networkx as nx
import numpy as np
from pyannote.core import SlidingWindowFeature
from pyannote.pipeline.parameter import Uniform

from pyannote.audio.core.inference import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.model import Model
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.utils import (
    PipelineModel,
    SpeakerDiarizationMixin,
    get_model,
)
from pyannote.audio.utils.metric import (
    DiscreteDiarizationErrorRate,
    SlidingDiarizationErrorRate,
)
from pyannote.audio.utils.permutation import mae_cost_func, permutate
from pyannote.audio.utils.signal import binarize


class SpeakerSegmentation(SpeakerDiarizationMixin, Pipeline):
    """Speaker segmentation

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation model. Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    skip_conversion : bool, optional
        Skip final conversion to pyannote.core.Annotation. Defaults to False.
    skip_stitching : bool, optional
        Skip stitching step. Defaults to False
    use_auth_token : str, optional
        When loading private huggingface.co models, set `use_auth_token`
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running `huggingface-cli login`

    Hyper-parameters
    ----------------
    onset : float
        Onset speaker activation threshold
    offset : float
        Offset speaker activation threshold
    stitch_threshold : float
        Initial stitching threshold.
    min_duration_on : float
        Remove speaker turn shorter than that many seconds.
    min_duration_off : float
        Fill same-speaker gaps shorter than that many seconds.

    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        skip_conversion: bool = False,
        skip_stitching: bool = False,
        use_auth_token: Union[Text, None] = None,
    ):
        super().__init__()

        self.segmentation = segmentation
        self.skip_stitching = skip_stitching
        self.skip_conversion = skip_conversion

        model: Model = get_model(segmentation, use_auth_token=use_auth_token)
        self._segmentation = Inference(model)
        self._frames = self._segmentation.model.introspection.frames

        self._audio = model.audio

        # number of speakers in output of segmentation model
        self._num_speakers = len(model.specifications.classes)

        #  hyper-parameters used for hysteresis thresholding
        self.onset = Uniform(0.0, 1.0)
        self.offset = Uniform(0.0, 1.0)

        if not self.skip_stitching:
            self.stitch_threshold = Uniform(0.0, 1.0)

        if not (self.skip_stitching or self.skip_conversion):
            self.min_duration_on = Uniform(0.0, 1.0)
            self.min_duration_off = Uniform(0.0, 1.0)

        self.warm_up = 0.05

    def default_parameters(self):
        # parameters optimized on DIHARD 3 development set

        if self.segmentation == "pyannote/segmentation":

            parameters = {
                "onset": 0.84,
                "offset": 0.46,
            }

            if not self.skip_stitching:
                parameters["stitch_threshold"] = 0.39

            if not (self.skip_stitching or self.skip_conversion):
                parameters.update(
                    {
                        "min_duration_on": 0.0,
                        "min_duration_off": 0.0,
                    }
                )

            return parameters

        raise NotImplementedError()

    CACHED_SEGMENTATION = "cache/segmentation/inference"

    @staticmethod
    def get_stitching_graph(
        segmentations: SlidingWindowFeature, onset: float = 0.5
    ) -> nx.Graph:
        """Build stitching graph

        Parameters
        ----------
        segmentations : (num_chunks, num_frames, local_num_speakers)-shaped SlidingWindowFeature
            Raw output of segmentation model.
        onset : float, optional
            Onset speaker activation threshold. Defaults to 0.5

        Returns
        -------
        stitching_graph : nx.Graph
            Nodes are (chunk_idx, speaker_idx) tuples.
            An edge between two nodes indicate that those are likely to be the same speaker
            (the lower the value of "cost" attribute, the more likely).
        """

        chunks = segmentations.sliding_window
        num_chunks, num_frames, _ = segmentations.data.shape
        max_lookahead = math.floor(chunks.duration / chunks.step - 1)
        lookahead = 2 * (max_lookahead,)

        stitching_graph = nx.Graph()

        for C, (chunk, segmentation) in enumerate(segmentations):
            for c in range(
                max(0, C - lookahead[0]), min(num_chunks, C + lookahead[1] + 1)
            ):

                if c == C:
                    continue

                # extract common temporal support
                shift = round((C - c) * num_frames * chunks.step / chunks.duration)

                if shift < 0:
                    shift = -shift
                    this_segmentations = segmentation[shift:]
                    that_segmentations = segmentations[c, : num_frames - shift]
                else:
                    this_segmentations = segmentation[: num_frames - shift]
                    that_segmentations = segmentations[c, shift:]

                # find the optimal one-to-one mapping
                _, (permutation,), (cost,) = permutate(
                    this_segmentations[np.newaxis],
                    that_segmentations,
                    cost_func=mae_cost_func,
                    return_cost=True,
                )

                for this, that in enumerate(permutation):

                    this_is_active = np.any(this_segmentations[:, this] > onset)
                    that_is_active = np.any(that_segmentations[:, that] > onset)

                    if this_is_active:
                        stitching_graph.add_node((C, this))

                    if that_is_active:
                        stitching_graph.add_node((c, that))

                    if this_is_active and that_is_active:
                        stitching_graph.add_edge(
                            (C, this), (c, that), cost=cost[this, that]
                        )

        return stitching_graph

    @staticmethod
    def stitchable_components(
        stitching_graph: nx.Graph, stitch_threshold: float, factor: float = 0.5
    ):
        """Find stitchable connected components

        A component is 'stitchable' if it contains at most one node per chunk

        Parameters
        ----------
        stitching_graph : nx.Graph
            Stitching graph.
        stitch_threshold : float

        Yields
        ------
        component : list of (chunk_idx, speaker_idx) tuple

        """

        f = stitching_graph.copy()
        while f:
            f.remove_edges_from(
                [
                    (n1, n2)
                    for n1, n2, cost in f.edges(data="cost")
                    if cost > stitch_threshold
                ]
            )
            for component in list(nx.connected_components(f)):
                if len(set(c for c, _ in component)) == len(component):
                    yield component
                    f.remove_nodes_from(component)
            stitch_threshold *= factor

    def apply(
        self, file: AudioFile, hook: Optional[Callable] = None
    ) -> SlidingWindowFeature:
        """Apply speaker segmentation

        Parameters
        ----------
        file : AudioFile
            Processed file.
        hook : callable, optional
            Callback called after each major steps of the pipeline as follows:
                hook(step_name,      # human-readable name of current step
                     step_artefact,  # artifact generated by current step
                     file=file)      # file being processed
            Time-consuming steps call `hook` multiple times with the same `step_name`
            and additional `completed` and `total` keyword arguments usable to track
            progress of current step.

        Returns
        -------
        segmentation : Annotation
            Speaker segmentation
        """

        hook = self.setup_hook(file, hook=hook)

        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, local_num_speakers)
        if self.training:
            if self.CACHED_SEGMENTATION in file:
                segmentations = file[self.CACHED_SEGMENTATION]
            else:
                segmentations = self._segmentation(
                    file, hook=partial(hook, "segmentation", None)
                )
                file[self.CACHED_SEGMENTATION] = segmentations
        else:
            segmentations: SlidingWindowFeature = self._segmentation(
                file, hook=partial(hook, "segmentation", None)
            )

        hook("segmentation", segmentations)

        if self.skip_stitching:
            return binarize(
                segmentations, onset=self.onset, offset=self.offset, initial_state=False
            )

        # estimate frame-level number of instantaneous speakers
        count = self.speaker_count(
            segmentations,
            onset=self.onset,
            offset=self.offset,
            warm_up=(self.warm_up, self.warm_up),
            frames=self._frames,
        )
        hook("speaker_counting", count)

        # trim warm-up regions
        segmentations = Inference.trim(
            segmentations, warm_up=(self.warm_up, self.warm_up)
        )

        # build stitching graph
        stitching_graph = self.get_stitching_graph(segmentations, onset=self.onset)

        # find (constraint-compliant) connected components
        components = list(
            self.stitchable_components(
                stitching_graph, stitch_threshold=self.stitch_threshold
            )
        )

        num_stitches = len(components)
        num_chunks, num_frames, _ = segmentations.data.shape

        stitched_segmentations = np.NAN * np.zeros(
            (num_chunks, num_frames, num_stitches)
        )

        for k, component in enumerate(components):
            for c, s in component:
                stitched_segmentations[c, :, k] = segmentations.data[c, :, s]

        stitched_segmentations = SlidingWindowFeature(
            stitched_segmentations, segmentations.sliding_window
        )

        hook("stitching", stitched_segmentations)

        # build discrete diarization
        discrete_diarization = self.to_diarization(stitched_segmentations, count)

        # remove inactive speakers
        discrete_diarization.data = discrete_diarization.data[
            :, np.nonzero(np.sum(discrete_diarization.data, axis=0))[0]
        ]

        if self.skip_conversion:
            return discrete_diarization

        hook("aggregation", discrete_diarization)

        # convert to continuous diarization
        diarization = self.to_annotation(
            discrete_diarization,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )

        diarization.uri = file["uri"]

        return diarization.rename_labels(
            {
                label: expected_label
                for label, expected_label in zip(diarization.labels(), self.classes())
            }
        )

    def get_metric(self):

        if self.skip_stitching:
            return DiscreteDiarizationErrorRate()

        if self.skip_conversion:
            raise NotImplementedError(
                "Cannot optimize segmentation pipeline when `skip_conversion` is True."
            )

        return SlidingDiarizationErrorRate(window=2.0 * self._segmentation.duration)
