from pyannote.audio.data.data import DownloadableProtocol
from pyannote.audio.data.util import download, extract
from pyannote.audio.utils.functional import chain


class MUSAN(DownloadableProtocol):
    """
    A collection of sounds from real life to help
    with sound classification
    """

    def __init__(self, *args, **kwargs):
        protocol_name = "MUSAN"
        task = "Collection"
        database_glob = "./" + self.__class__.__name__ + "{uri}.wav"
        protocol_entries = {
            "BackgroundNoise": {"uri": "./MUSAN/background_noise.txt"},
            "Noise": {"uri": "./MUSAN/noise.txt"},
            "Music": {"uri": "./MUSAN/music.txt"},
            "Speech": {"uri": "./MUSAN/speech.txt"},
        }
        super().__init__(
            task,
            protocol_name=protocol_name,
            database_glob=database_glob,
            protocol_entries=protocol_entries,
        )

    def prepare(self):
        # Get the lists
        url = "https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/data_preparation/MUSAN/{lst}.txt"
        lsts = [
            url.format(lst=lst)
            for lst in ["background_noise", "music", "noise", "speech"]
        ]
        download(lsts)

        # Download and extract the tar file
        url = "http://www.openslr.org/resources/17/musan.tar.gz"
        chain([download, extract], url)
