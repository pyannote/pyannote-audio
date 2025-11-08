# MIT License
#
# Copyright (c) 2024-2025 CNRS
# Copyright (c) 2025- pyannoteAI
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

import os
from pathlib import Path
from typing import Optional

import numpy as np
from pyannote.audio.utils.hf_hub import download_from_hf_hub
from pyannote.audio.utils.vbx import vbx_setup


class PLDA:
    """PLDA"""

    def __init__(
        self, transform_npz: str | Path, plda_npz: str | Path, lda_dimension: int = 128
    ):
        self._xvec_tf, self._plda_tf, self._plda_psi = vbx_setup(
            transform_npz, plda_npz
        )

        self.lda_dimension = lda_dimension

    @property
    def phi(self):
        """Between-class covariance in the PLDA space."""
        return self._plda_psi[: self.lda_dimension]

    def __call__(self, embeddings: np.ndarray):
        """

        Parameters
        ----------
        embeddings : (num_embeddings, embedding_dimension) ndarray
            Embeddings to be transformed into the PLDA space.

        Returns
        -------
        fea : (num_embeddings, lda_dimension) ndarray
            Embeddings transformed into the PLDA space.
        """
        return self._plda_tf(self._xvec_tf(embeddings), lda_dim=self.lda_dimension)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: Path | str,
        subfolder: str | None = None,
        revision: str | None = None,
        token: str | None = None,
        cache_dir: str | Path | None = None,
        **kwargs,
    ) -> Optional["PLDA"]:
        """Load PLDA from disk or Huggingface Hub

        Parameters
        ----------
        checkpoint : Path or str
            Path to checkpoint or a model identifier from the hf.co model hub.
        subfolder : str, optional
            Folder inside the hf.co model repo.
        revision : str, optional
            Revision when loading from the huggingface.co model hub.
        token : str, optional
            When loading a private hf.co model, set `token`
            to True or to a string containing your hugginface.co authentication
            token that can be obtained by running `huggingface-cli login`
        cache_dir: Path or str, optional
            Path to model cache directory.
        """

        # if checkpoint is a directory, look for the PLDA checkpoint
        # inside this directory (or inside a subfolder if specified)
        if os.path.isdir(checkpoint):
            if revision is not None:
                raise ValueError("Revisions cannot be used with local checkpoints.")

            if subfolder:
                path_to_transform = Path(checkpoint) / subfolder / "xvec_transform.npz"
                path_to_plda = Path(checkpoint) / subfolder / "plda.npz"
            else:
                path_to_transform = Path(checkpoint) / "xvec_transform.npz"
                path_to_plda = Path(checkpoint) / "plda.npz"

        # otherwise, assume that the checkpoint is hosted on HF model hub
        else:
            checkpoint = str(checkpoint)
            if "@" in checkpoint:
                raise ValueError(
                    "Revisions must be passed with `revision` keyword argument."
                )

            path_to_transform = download_from_hf_hub(
                checkpoint,
                "xvec_transform.npz",
                subfolder=subfolder,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
            )

            path_to_plda = download_from_hf_hub(
                checkpoint,
                "plda.npz",
                subfolder=subfolder,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
            )

            if path_to_transform is None or path_to_plda is None:
                return None

        return cls(path_to_transform, path_to_plda)
