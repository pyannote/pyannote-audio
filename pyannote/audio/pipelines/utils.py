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

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Union

import numpy as np

from pyannote.audio.core.inference import Inference
from pyannote.core import Annotation, SlidingWindowFeature


def assert_string_labels(annotation: Annotation, name: str):
    """Check that annotation only contains string labels

    Parameters
    ----------
    annotation : Annotation
        Annotation.
    name : str
        Name of the annotation (used for user feedback in case of failure)
    """

    if any(not isinstance(label, str) for label in annotation.labels()):
        msg = f"{name} must contain `str` labels only."
        raise ValueError(msg)


def assert_int_labels(annotation: Annotation, name: str):
    """Check that annotation only contains integer labels

    Parameters
    ----------
    annotation : Annotation
        Annotation.
    name : str
        Name of the annotation (used for user feedback in case of failure)
    """

    if any(not isinstance(label, int) for label in annotation.labels()):
        msg = f"{name} must contain `int` labels only."
        raise ValueError(msg)


def gather_label_embeddings(
    annotation: Annotation, embeddings: Union[SlidingWindowFeature, Inference]
):
    """Extract one embedding per label

    Parameters
    ----------
    annotation : Annotation
        Annotation
    embeddings : SlidingWindowFeature or Inference
        Embeddings, either precomputed on a sliding window (SlidingWindowFeature)
        or to be computed on the fly (Inference).

    Returns
    -------
    embeddings : ((len(embedded_labels), embedding_dimension) np.ndarray
        Embeddings.
    embedded_labels : list of labels
        Labels for which an embedding has been computed.
    skipped_labels : list of labels
        Labels for which no embedding could be computed.
    """

    X, embedded_labels, skipped_labels = [], [], []

    labels = annotation.labels()
    for label in labels:

        label_support = annotation.label_timeline(label, copy=False)

        if isinstance(embeddings, SlidingWindowFeature):

            # be more and more permissive until we have
            # at least one embedding for current speech turn
            for mode in ["strict", "center", "loose"]:
                x = embeddings.crop(label_support, mode=mode)
                if len(x) > 0:
                    break

            # skip labels so small we don't have any embedding for it
            if len(x) < 1:
                skipped_labels.append(label)
                continue

            embedded_labels.append(label)
            X.append(np.mean(x, axis=0))

        elif isinstance(embeddings, Inference):
            # TODO: add "Timeline" chunk support to Inference.crop
            raise NotImplementedError()

    return np.vstack(X), embedded_labels, skipped_labels
