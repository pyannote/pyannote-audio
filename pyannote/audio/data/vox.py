import os
import subprocess
from functools import partial

from pyannote.audio.core.data import DownloadableProtocol, download


def authorised_curl(url, dest, username=None, password=None):
    subprocess.call(
        ["curl", "-C", "-", url, "-o", dest, "-u", f"{username}:{password}"]
    )


class VoxDataset(DownloadableProtocol):
    def setup_authentication(self):
        if self.user is None:
            self.user = os.environ["VOX_USER"]
        if self.password is None:
            self.password = os.environ["VOX_PASSWORD"]
        if None in [self.user, self.password]:
            msg = """You must set the VOX_USER AND VOX_PASSWORD to download the {self.__class__.__name} datset
            """
            raise ValueError(msg)

        curl = partial(authorised_curl, username=self.user, password=self.password)
        self.dl = partial(download, download_func=curl)


class VoxCeleb1(VoxDataset):
    def prepare(self):
        root_url = "http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_part{part}"
        parts = [
            root_url.format(part=p) for p in ["a", "b", "c", "d", "e", "f", "g", "h"]
        ]

        self.setup_authentication()
        download(parts)


class VoxCeleb2(VoxDataset):
    pass


class VoxConverse(VoxDataset):
    pass
