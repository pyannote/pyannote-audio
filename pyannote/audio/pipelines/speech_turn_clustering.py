# The MIT License (MIT)
#
# Copyright (c) 2018-2020 CNRS
#
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

from typing import Optional, Text, Union

from pyannote.audio.core.inference import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.core import Annotation
from pyannote.pipeline import Pipeline
from pyannote.pipeline.blocks.clustering import HierarchicalAgglomerativeClustering

from .utils import assert_string_labels, gather_label_embeddings


class SpeechTurnClustering(Pipeline):
    """Speech turn clustering

    Parameters
    ----------
    embeddings : Inference or str, optional
        `Inference` instance used to extract speaker embeddings. When `str`,
        assumes that file already contains a corresponding key with precomputed
        embeddings. Defaults to "emb".
    metric : {'euclidean', 'cosine', 'angular'}, optional
        Metric used for comparing embeddings. Defaults to 'cosine'.
    """

    def __init__(
        self,
        embeddings: Union[Inference, Text] = "emb",
        metric: Optional[str] = "cosine",
    ):
        super().__init__()

        self.embeddings = embeddings
        self.metric = metric

        self.clustering = HierarchicalAgglomerativeClustering(
            metric=self.metric, use_threshold=True
        )

    def __call__(self, file: AudioFile, speech_turns: Annotation) -> Annotation:
        """Apply speech turn clustering

        Parameters
        ----------
        file : AudioFile
            Processed file
        speech_turns : Annotation
            Speech turns. Should only contain `str` labels.

        Returns
        -------
        speech_turns : Annotation
            Clustered speech turns
        """

        assert_string_labels(speech_turns, "speech_turns")

        if isinstance(self.embeddings, Inference):
            if self.embeddings.window == "sliding":
                # we precompute embeddings using a sliding window
                embeddings = self.embeddings(file)

            elif self.embeddings.window == "whole":
                raise NotImplementedError(
                    "Inference with 'whole' window is not supported yet."
                )
                # TODO: warning about speed issue when Inference is on CPU

        else:
            # we load precomputed embeddings
            embeddings = file[self.embeddings]

        X, clustered_labels, skipped_labels = gather_label_embeddings(
            speech_turns, embeddings
        )

        # apply clustering of label embeddings
        clusters = self.clustering(X)

        # map each clustered label to its cluster (between 1 and N_CLUSTERS)
        mapping = {label: k for label, k in zip(clustered_labels, clusters)}

        # map each skipped label to its own cluster
        # (between -1 and -N_SKIPPED_LABELS)
        for i, label in enumerate(skipped_labels):
            mapping[label] = -(i + 1)

        # do the actual mapping
        return speech_turns.rename_labels(mapping=mapping)
