import glob
import os
import tarfile
import urllib
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Union

import yaml
from torchaudio.datasets.utils import tqdm, validate_file

from pyannote.audio.utils.functional import apply_to_array
from pyannote.database.util import get_database_yml


def _get_yaml():
    """
    This method retrieves the url of the pyannote database.yml
    file location. If it doesn't exist, then the one is created
    where the program is run.
    """
    try:
        yml_path = get_database_yml()
    except FileNotFoundError:
        yml_path = Path("database.yml")
        yml_path.touch()
        with open(yml_path, "w") as f:
            yaml.dump({"Protocols": {}, "Databases": {}}, f, Dumper=yaml.SafeDumper)
    return yml_path


def _extract_file(path: Path, dest: Path, extract=True):
    """
    Extracts the given tar or zip into the dest directory
    """
    parent = path.parent
    fname = str(path)
    if fname.endswith(".gz") or fname.endswith(".tar"):
        rm = ":gz" if fname[-2] == "gz" else ""
        tar = tarfile.open(path, "r" + rm)
        return [parent / t.name for t in tar.getmembers() if not t.isdir()]
    if fname.endswith(".zip"):
        Zip = zipfile.ZipFile(fname)
        return [
            parent / f.filename for f in Zip.filelist if not f.filename.endswith("/")
        ]
    raise ValueError("Unsupported file extension")


@apply_to_array
def extract(archive: Union[str, Path]):
    """
    Wraps the _extract_file and checks if the folder has
    already been extracted
    """
    p = Path(archive)
    d = p.stem
    dest = Path(p).parent / d
    if dest.is_dir():
        return glob.glob(str(dest) + "**/**", recursive=True)
    return _extract_file(p, dest, archive is not None)


def stream_url(
    url: str,
    start_byte: Optional[int] = None,
    block_size: int = 32 * 1024,
    progress_bar: bool = True,
    headers: dict = {},
) -> Iterable:
    """Stream url by chunk

    Method tweaked from torchaudio.datasets.utils

    Args:
        url (str): Url.
        start_byte (int, optional): Start streaming at that point (Default: ``None``).
        block_size (int, optional): Size of chunks to stream (Default: ``32 * 1024``).
        progress_bar (bool, optional): Display a progress bar (Default: ``True``).
    """

    # If we already have the whole file, there is no need to download it again
    req = urllib.request.Request(url, method="HEAD", headers=headers)
    url_size = int(urllib.request.urlopen(req).info().get("Content-Length", -1))
    if url_size == start_byte:
        return

    req = urllib.request.Request(url, headers=headers)
    if start_byte:
        req.headers["Range"] = "bytes={}-".format(start_byte)
    else:
        start_byte = 0

    with urllib.request.urlopen(req) as upointer, tqdm(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        total=url_size - start_byte,
        disable=not progress_bar,
    ) as pbar:

        num_bytes = 0
        while True:
            chunk = upointer.read(block_size)
            if not chunk:
                break
            yield chunk
            num_bytes += len(chunk)
            pbar.update(len(chunk))


def download_url_with_headers(
    url: str,
    download_folder: str,
    filename: Optional[str] = None,
    hash_value: Optional[str] = None,
    hash_type: str = "sha256",
    progress_bar: bool = True,
    resume: bool = True,
    headers: dict = {},
) -> None:
    """Download file to disk.

    Method tweaked from torchaudio.datasets.utils.download_url

    Args:
        url (str): Url.
        download_folder (str): Folder to download file.  filename (str, optional): Name of downloaded file. If None, it is inferred from the url (Default: ``None``).
        hash_value (str, optional): Hash for url (Default: ``None``).
        hash_type (str, optional): Hash type, among "sha256" and "md5" (Default: ``"sha256"``).
        progress_bar (bool, optional): Display a progress bar (Default: ``True``).
        resume (bool, optional): Enable resuming download (Default: ``False``).
        headers (doct, optional): Headers to add to the file
    """
    req = urllib.request.Request(url, method="HEAD", headers=headers)
    req_info = urllib.request.urlopen(req).info()

    # Detect filename
    filename = filename or req_info.get_filename() or os.path.basename(url)
    filepath = os.path.join(download_folder, filename)

    if resume and os.path.exists(filepath):
        mode = "ab"
        local_size: Optional[int] = os.path.getsize(filepath)

    elif not resume and os.path.exists(filepath):
        raise RuntimeError(
            "{} already exists. Delete the file manually and retry.".format(filepath)
        )
    else:
        mode = "wb"
        local_size = None

    if hash_value and local_size == int(req_info.get("Content-Length", -1)):
        with open(filepath, "rb") as file_obj:
            if validate_file(file_obj, hash_value, hash_type):
                return filepath
        raise RuntimeError(
            "The hash of {} does not match. Delete the file manually and retry.".format(
                filepath
            )
        )

    with open(filepath, mode) as fpointer:
        for chunk in stream_url(
            url, start_byte=local_size, progress_bar=progress_bar, headers=headers
        ):
            fpointer.write(chunk)

    with open(filepath, "rb") as file_obj:
        if hash_value and not validate_file(file_obj, hash_value, hash_type):
            raise RuntimeError(
                "The hash of {} does not match. Delete the file manually and retry.".format(
                    filepath
                )
            )
    return filepath


@apply_to_array
def download(url: str, dest_dir=Path("."), headers={}):
    """Wrapper for the download_url_with_headers that sets defaults up.

    Parameters
    ----------
    url : str
        The url to download
    dest_dir: Path
        The directory to save downloaded files in
    headers: dict
        Headers to add to requests e.g. for authentication
    Returns
    -------
    downloaded_file_uri: Path
        Path of the file that was downloaded

    Usage
    -----
    >>> audio = Audio(sample_rate=16000, mono=True)
    >>> waveform, sample_rate = audio({"audio": "/path/to/audio.wav"})
    >>> assert sample_rate == 16000

    >>> two_seconds_stereo = np.random.rand(44100 * 2, 2, dtype=np.float32)
    >>> waveform, sample_rate = audio({"waveform": two_seconds_stereo, "sample_rate": 44100})
    >>> assert sample_rate == 16000
    >>> assert waveform.shape[1] == 1
    """
    return Path(download_url_with_headers(url, dest_dir, headers=headers))


@apply_to_array
def cat(file: Union[str, Path], dest: Union[str, Path]):
    """Appends one file to the dest

    Parameters
    ----------
    file : Path
        The file that is to be appended
    dest : str
        Where to append the file

    Returns
    -------
    The created file
    """
    with open(file, "rb") as part, open(dest, "ab") as dest:
        dest.write(part.read())
    return dest


@apply_to_array
def rm(file: Path):
    """Removes a file

    Parameters
    ----------
    file : str, Path
        The file to remove
    """
    file.unlink()
