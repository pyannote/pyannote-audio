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

from enum import Enum
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from pyannote.audio import __version__


# Correspondence between asset_file type
# and their filename on Huggingface
class AssetFileName(Enum):
    Calibration = "calibration.safetensors"
    Model = "pytorch_model.bin"
    Pipeline = "config.yaml"

    def __str__(self) -> str:
        return self.value


def download_from_hf_hub(
    model_id: str,
    asset_file: AssetFileName | str,
    subfolder: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: bool | str | None = None,
) -> str | None:
    """Download file from Huggingface Hub

    Parameters
    ----------
    model_id : str
        Model identifier from the hf.co model hub.
    asset_file : AssetFileName
        Type of asset file to download.
    subfolder : str, optional
        Folder inside the model repo.
    revision : str, optional
        Revision when loading from the huggingface.co model hub.
    token : str or bool, optional
        Huggingface token to be used for downloading from Huggingface hub.
    cache_dir: Path or str, optional
        Path to the folder where files downloaded from Huggingface hub are stored.

    See also
    --------
    `huggingface_hub.hf_hub_download`
    """

    # if provided token does not start with 'hf_', it is likely a pyannoteAI API key
    # and therefore should not be passed to Huggingface Hub.
    if isinstance(token, str) and not token.startswith("hf_"):
        token = None

    try:
        return hf_hub_download(
            model_id,
            asset_file.value if isinstance(asset_file, AssetFileName) else asset_file,
            subfolder=subfolder,
            repo_type="model",
            revision=revision,
            library_name="pyannote",
            library_version=__version__,
            cache_dir=cache_dir,
            token=token,
        )
    except HfHubHTTPError:
        asset_file_name: str = asset_file.name if isinstance(asset_file, AssetFileName) else asset_file
        print(
            f"""
Could not download {asset_file_name} from {model_id}.
It might be because the repository is private or gated:

* visit https://hf.co/{model_id} to accept user conditions
* visit https://hf.co/settings/tokens to create an authentication token
* load the {asset_file_name} with the `token` argument:
    >>> {asset_file_name}.from_pretrained('{model_id}', token='hf_....')
"""
        )
        raise
