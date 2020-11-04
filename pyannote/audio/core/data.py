import subprocess
import tarfile
import urllib.request
import zipfile
from functools import cached_property, wraps
from pathlib import Path
from typing import Union

import yaml

from pyannote.database import DATABASES, Database, FileFinder
from pyannote.database.config import get_database_yml
from pyannote.database.custom import create_protocol, get_init


def apply_to_array(func):
    """
    Returns the same function but it will iterate through an Iterable
    and apply the function to each item, returning an array"""

    @wraps(func)
    def _inner(arg, **kwargs):
        if isinstance(arg, list):
            return [func(a, **kwargs) for a in arg]
        return func(arg, **kwargs)

    return _inner


def _get_yaml():
    """
    Configure the default location to set yaml
    even if it is not found anywhere
    """
    try:
        yml_path = get_database_yml()
    except FileNotFoundError:
        yml_path = Path("database.yml")
        yml_path.touch()
        with open(yml_path, "w") as f:
            yaml.dump({"Protocols": {}, "Databases": {}}, f, Dumper=yaml.SafeDumper)
    return yml_path


def chain(funcs, x):
    """
    Chains methods of a function to be used one after other on the outputs

    Usage
    =====
    chain([g,f],x) -> f(g(x))
    """

    current = x
    for f in funcs:
        current = f(current)
    return current


def apply_to_array(func):
    """
    Returns the same function but it will iterate through an Iterable
    and apply the function to each item, returning an array
    """

    @wraps(func)
    def _inner(arg, **kwargs):
        if isinstance(arg, list):
            return [func(a, **kwargs) for a in arg]
        return func(arg, **kwargs)

    return _inner


def curl_download(url, dest):
    subprocess.call(["curl", "-C", "-", url, "-o", dest])


@apply_to_array
def download(url: str, dest_dir=Path("."), download_func=curl_download):
    dest = url.split("/")[-1]
    dest = dest_dir / Path(dest)

    # If file doesn't exist download
    downloaded = False
    dest_size = -1

    content_length = urllib.request.urlopen(url).length
    # Force the directory
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file():  # or force_download
        dest_size = dest.stat().st_size

    # If size not correct download
    if dest_size < content_length:
        download_func(url, dest)
        downloaded = True
    return {
        "path": dest,
        "downloaded": downloaded,
    }


def _extract_file(path, dest, extract=True):
    parent = path.parent
    fname = str(path)
    if fname.endswith(".gz") or fname.endswith(".tar"):
        rm = ":gz" if fname[-2] == "gz" else ""
        tar = tarfile.open(path, "r" + rm)
        if extract:
            tar.extractall(dest)
        return [parent / t.name for t in tar.getmembers() if not t.isdir()]
    if fname.endswith(".zip"):
        Zip = zipfile.ZipFile(fname)
        if extract:
            Zip.extractall(dest)
        return [
            parent / f.filename for f in Zip.filelist if not f.filename.endswith("/")
        ]
    raise ValueError("Unsupported file extension")


@apply_to_array
def extract(archive):
    if isinstance(archive, (str, Path)):
        p, downloaded = Path(archive), True
    else:
        p, downloaded = archive["path"], archive["downloaded"]
    dest = Path(p).parent
    return _extract_file(p, dest, downloaded)


class DownloadableProtocol:
    def __init__(
        self,
        task: str,
        database: str = None,
        data_dir: Union[str, Path] = None,
        force_download=False,
        force_valid=False,
        protocol_name=None,
        database_glob=None,
        protocol_entries=dict(),
    ):
        super().__init__()
        # Configure whether to force download and validate data
        self.force_download = force_download
        self.force_valid = force_valid

        # These are used to create the protocol class
        self.database = self.__class__.__name__ if database is None else database
        self.task = task
        self.protocol_name = protocol_name
        self.database_glob = database_glob
        self.protocol_entries = protocol_entries

        # Write these to the configured yaml environment
        self._write_to_yaml()
        data_dir = _get_yaml() if data_dir is None else data_dir
        self.data_dir = Path(data_dir).parent / self.database

    def prepare():
        raise NotImplementedError("DownloadableProtocol.prepare() must be implemented")

    def _write_to_yaml(self):
        yaml_path = _get_yaml()

        with open(yaml_path) as f:
            yml = yaml.load(f, Loader=yaml.SafeLoader)
        if yml is None:
            yml = {}

        Protocols = yml["Protocols"]
        if Protocols is None:
            Protocols = {}

        if self.database not in Protocols:
            Protocols[self.database] = {
                f"{self.task}": {f"{self.protocol_name}": self.protocol_entries}
            }

        if yml["Databases"] is None:
            yml["Databases"] = {}

        if self.database not in yml["Databases"]:
            if yml["Databases"] is None:
                yml["Databases"] = {}
            yml["Databases"][self.database] = self.database_glob

        with open(yaml_path, "w") as f:
            yaml.dump(yml, f, Dumper=yaml.SafeDumper)

    def train(self):
        return self.protocol.train()

    def development(self):
        return self.protocol.development()

    def test(self):
        return self.protocol.test()

    @cached_property
    def protocol(self):
        protocol = create_protocol(
            self.database,
            self.task,
            self.protocol_name,
            self.protocol_entries,
            self.data_dir,
        )
        DATABASES[self.database] = type(
            self.database,
            (Database,),
            {"__init__": get_init([self.task, self.protocol_name, protocol])},
        )
        return protocol(preprocessors={"audio": FileFinder()})
