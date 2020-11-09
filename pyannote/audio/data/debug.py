from pyannote.audio.data.data import DownloadableProtocol
from pyannote.audio.data.util import download


class Debug(DownloadableProtocol):
    """"""

    def __init__(self, *args, **kwargs):
        protocol_name = "Debug"
        task = "SpeakerDiarisation"
        database_glob = "./" + self.__class__.__name__ + "{uri}.wav"
        protocol_entries = {
            "train": {
                "uri": "debug.train.lst",
                "annotation": "debug.train.rttm",
                "annotated": "debug.train.uem",
            },
            "development": {
                "uri": "debug.development.lst",
                "annotation": "debug.development.rttm",
                "annotated": "debug.development.uem",
            },
            "test": {
                "uri": "debug.test.lst",
                "annotation": "debug.test.rttm",
                "annotated": "debug.test.uem",
            },
        }
        super().__init__(
            task,
            database_glob=database_glob,
            protocol_entries=protocol_entries,
            protocol_name=protocol_name,
        )

    def prepare(self):
        root_url = (
            "https://raw.githubusercontent.com/pyannote/pyannote-data/master/audio/"
        )
        files = []
        files += [f"dev0{x}.wav" for x in range(1)]
        files += [f"trn0{x}.wav" for x in range(9)]
        files += [
            f"debug.{mode}.{ext}"
            for mode in ["development", "train", "test"]
            for ext in ["lst", "rttm", "uem"]
        ]

        files = [root_url + f for f in files]
        download(files)
