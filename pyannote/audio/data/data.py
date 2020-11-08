from functools import cached_property
from pathlib import Path
from typing import Union

import yaml

from pyannote.database import DATABASES, Database, FileFinder, create_protocol, get_init

from .util import _get_yaml


class DownloadableProtocol:
    """A class to assist in the download and creation of Protocols
    within a pyannote.database yml file.

    Parameters
    ----------
    task : str
        The task that this belongs too, i.e. SpeakerDiarisation
    database : str
        The name of the database that this protocol belongs too,
        if none, derived from the class name
    data_dir : str, Path
        The parent directory of the pyannote.database file
    protocol_name : str
        A string naming the particular protocol
    database_glob : str
        A glob to help pyannote.database find files for the database
    protocol_entries : dict
        A dictionary where the splits for the dataset can be found.
    """

    def __init__(
        self,
        task: str,
        database: str = None,
        data_dir: Union[str, Path] = None,
        protocol_name: str = None,
        database_glob: str = None,
        protocol_entries: dict = dict(),
    ):
        # These are used to create the protocol class
        self.database = self.__class__.__name__ if database is None else database
        self.task = task
        self.protocol_name = protocol_name
        self.database_glob = database_glob
        self.protocol_entries = protocol_entries

        # Write these to the configured yaml environment
        self._write_to_yaml()
        data_dir = _get_yaml().parent if data_dir is None else data_dir
        self.data_dir = Path(data_dir) / self.database
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _write_to_yaml(self):
        "Writes the Protocol into the database.yml file"
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
        yml["Databases"][self.database] = self.database_glob

        with open(yaml_path, "w") as f:
            yaml.dump(yml, f, Dumper=yaml.SafeDumper)

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

    def train(self):
        return self.protocol.train()

    def development(self):
        return self.protocol.development()

    def test(self):
        return self.protocol.test()
