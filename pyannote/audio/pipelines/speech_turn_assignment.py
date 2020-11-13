# The MIT License (MIT)
#
# Copyright (c) 2017-2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
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
from pyannote.pipeline.blocks.classification import ClosestAssignment

from .utils import assert_int_labels, assert_string_labels, gather_label_embeddings


class SpeechTurnClosestAssignment(Pipeline):
    """Assign speech turn to closest cluster

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

        self.closest_assignment = ClosestAssignment(metric=self.metric)

    def __call__(
        self, file: AudioFile, speech_turns: Annotation, targets: Annotation
    ) -> Annotation:
        """Assign each speech turn to closest target (if close enough)

        Parameters
        ----------
        file : AudioFile
            Processed file
        speech_turns : Annotation
            Speech turns. Should only contain `int` labels.
        targets : Annotation
            Targets. Should only contain `str` labels.

        Returns
        -------
        assigned : Annotation
            Assigned speech turns.
        """

        assert_string_labels(targets, "targets")
        assert_int_labels(speech_turns, "speech_turns")

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

        # gather targets embedding
        X_targets, targets_labels, _ = gather_label_embeddings(targets, embeddings)

        # gather speech turns embedding
        X, assigned_labels, skipped_labels = gather_label_embeddings(
            speech_turns, embeddings
        )

        # assign speech turns to closest class
        assignments = self.closest_assignment(X_targets, X)
        mapping = {
            label: targets_labels[k]
            for label, k in zip(assigned_labels, assignments)
            if not k < 0
        }
        return speech_turns.rename_labels(mapping=mapping)
