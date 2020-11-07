import os
import subprocess
from functools import partial

from pyannote.audio.core.data import (
    DownloadableProtocol,
    apply_to_array,
    download,
    extract,
)


def authorised_curl(url, dest, username=None, password=None):
    subprocess.call(
        ["curl", "-C", "-", url, "-o", dest, "-u", f"{username}:{password}"]
    )


@apply_to_array
def cat(file, dest):
    subprocess.call(["cat", file, ">>", str(dest)])


class VoxDataset(DownloadableProtocol):
    def setup_authentication(self):
        if not hasattr(self, "user"):
            self.user = os.environ["VOX_USER"]
        if not hasattr(self, "password"):
            self.password = os.environ["VOX_PASSWORD"]
        if None in [self.user, self.password]:
            msg = """You must set the VOX_USER AND VOX_PASSWORD to download the {self.__class__.__name} datset
            """
            raise ValueError(msg)

        curl = partial(authorised_curl, username=self.user, password=self.password)
        self.dl = partial(download, download_func=curl)


class VoxCeleb1(VoxDataset):
    def __init__(self, *args, **kwargs):
        super().__init__("SpeakerIdentification", *args, **kwargs)

    def prepare(self):
        dev_url = "http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_parta{part}"
        parts = [dev_url.format(part=p) for p in ["a", "b", "c", "d"]]

        self.setup_authentication()
        zip_uri = self.data_dir / "vox_aac.zip"
        cat(self.dl(parts), dest=zip_uri)
        extract(zip_uri)
        extract(
            self.dl(
                "http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip"
            )
        )

        list_url = (
            "http://www.robots.ox.ac.uk/~vgg/data/voxceleb/data/vox1_{mode}_txt.zip"
        )

        # List Files
        extract(self.dl([list_url.format(mode=mode) for mode in ["dev", "test"]]))

        # Contains splits
        splits = "http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt"
        download(splits)


class VoxCeleb2(VoxDataset):
    def __init__(self, *args, **kwargs):
        super().__init__("SpeakerIdentification", *args, **kwargs)

    def prepare(self):
        self.setup_authentication()

        # Dev
        dev_url = "http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_part{part}"
        parts = [
            dev_url.format(part=p) for p in ["a", "b", "c", "d", "e", "f", "g", "h"]
        ]
        zip_uri = self.data_dir / "vox2_aac.zip"
        cat(self.dl(parts), dest=zip_uri)
        extract(zip_uri)

        extract(
            self.dl(
                "http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_test_aac.zip"
            )
        )

        # Test
        list_url = (
            "http://www.robots.ox.ac.uk/~vgg/data/voxceleb/data/vox2_{mode}_txt.zip"
        )

        # List Files
        extract(self.dl([list_url.format(mode=mode) for mode in ["dev", "test"]]))

        # Contains splits
        download("http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox2_meta.csv")


class VoxConverse(VoxDataset):
    def prepare(self):
        # Dev

        urls = [
            "http://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip",
            "http://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_challengetest_wav.zip",
            "https://github.com/joonson/voxconverse/archive/master.zip",
        ]

        extract(self.dl(urls))
