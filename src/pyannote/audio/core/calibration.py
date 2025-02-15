# MIT License
#
# Copyright (c) 2024- CNRS
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
from typing import Optional, Text, Union

import numpy as np
import safetensors.numpy
import scipy.interpolate
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.validation import NotFittedError, check_is_fitted

from pyannote.audio.utils.hf_hub import AssetFileName, download_from_hf_hub


class Calibration(IsotonicRegression):
    """Logit/distance calibration"""

    def __init__(self):
        super().__init__(y_min=0.0, y_max=1.0, increasing="auto", out_of_bounds="clip")

    def save(self, path: str):
        """Save fitted calibration to disk

        Parameters
        ----------
        path : str
            Path to the file where the calibration should be saved

        Raises
        ------
        NotFittedError
            If the calibration is not fitted yet

        """
        try:
            check_is_fitted(self)
        except NotFittedError:
            raise NotFittedError("Cannot save an unfitted model.")

        tensor_dict = {
            "X_min_": self.X_min_,
            "X_max_": self.X_max_,
            "X_thresholds_": self.X_thresholds_,
            "y_thresholds_": self.y_thresholds_,
            "increasing_": self.increasing_,
        }

        safetensors.numpy.save_file(tensor_dict, path)

    @classmethod
    def from_file(cls, path: str) -> "Calibration":
        """Load calibration from disk

        Parameters
        ----------
        path : str
            Path to the file where the calibration is saved

        Returns
        -------
        calibration : Calibration
            Fitted calibration
        """
        calibration = cls()

        tensor_dict = safetensors.numpy.load_file(path)
        for key, value in tensor_dict.items():
            setattr(calibration, key, value)

        calibration.f_ = scipy.interpolate.interp1d(
            np.hstack(
                [
                    [np.min(calibration.X_thresholds_) - 1.0],
                    calibration.X_thresholds_,
                    [np.max(calibration.X_thresholds_) + 1.0],
                ]
            ),
            np.hstack(
                [
                    [1.0 - calibration.increasing_],
                    calibration.y_thresholds_,
                    [1.0 * calibration.increasing_],
                ]
            ),
            kind="linear",
            bounds_error=False,
        )

        return calibration

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: Union[Path, str],
        subfolder: Optional[str] = None,
        token: Optional[Text] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Optional["Calibration"]:
        """Load calibration from disk or Huggingface Hub

        Parameters
        ----------
        checkpoint : Path or str
            Path to checkpoint or a model identifier from the hf.co model hub.
        subfolder : str, optional
            Folder inside the hf.co model repo.
        token : str, optional
            When loading a private hf.co model, set `token`
            to True or to a string containing your hugginface.co authentication
            token that can be obtained by running `huggingface-cli login`
        cache_dir: Path or str, optional
            Path to model cache directory.
        """

        # if checkpoint is a directory, look for the calibration checkpoint
        # inside this directory (or inside a subfolder if specified)
        if os.path.isdir(checkpoint):
            if subfolder:
                path_to_calibration_checkpoint = (
                    Path(checkpoint) / subfolder / AssetFileName.Calibration.value
                )
            else:
                path_to_calibration_checkpoint = (
                    Path(checkpoint) / AssetFileName.Calibration.value
                )

        # if checkpoint is a file, use it as is
        elif os.path.isfile(checkpoint):
            path_to_calibration_checkpoint = checkpoint

        # otherwise, assume that the checkpoint is hosted on HF model hub
        else:
            _, _, path_to_calibration_checkpoint = download_from_hf_hub(
                str(checkpoint),
                AssetFileName.Calibration,
                subfolder=subfolder,
                cache_dir=cache_dir,
                token=token,
            )

            if path_to_calibration_checkpoint is None:
                return None

        return cls.from_file(path_to_calibration_checkpoint)
