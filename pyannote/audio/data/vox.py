import os
from base64 import b64encode
from functools import partial

from pyannote.audio.data.data import DownloadableProtocol
from pyannote.audio.data.util import cat, download, extract, rm


class VoxDataset(DownloadableProtocol):
    """
    Base for the others
    """

    def setup_authentication(self):
        if not hasattr(self, "user"):
            self.user = os.environ["VOX_USER"]
        if not hasattr(self, "password"):
            self.password = os.environ["VOX_PASSWORD"]
        if None in [self.user, self.password]:
            msg = """
            You must set the VOX_USER AND VOX_PASSWORD to download
            the {self.__class__.__name} dataset
            """
            raise ValueError(msg)

        authorization = "Basic " + b64encode(
            f"{self.user}:{self.password}".encode("utf-8")
        ).decode("utf-8")
        self.dl = partial(
            download, dest_dir=self.data_dir, headers={"authorization": authorization}
        )


class VoxCeleb1(VoxDataset):
    """
    VoxCeleb1 contains over 100000 utterances for 1251 celebrities,
    extracted from videos uploaded to YouTube.

    http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html

    Citation
    --------
    @InProceedings{Nagrani17,
        author       = "Nagrani, A. and Chung, J.~S. and Zisserman, A.",
        title        = "VoxCeleb: a large-scale speaker identification dataset",
        booktitle    = "INTERSPEECH",
        year         = "2017",
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__("SpeakerIdentification", *args, **kwargs)

    def prepare(self):
        dev_url = "http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_parta{part}"

        parts = [dev_url.format(part=p) for p in ["a", "b", "c", "d"]]

        self.setup_authentication()

        zip_uri = self.data_dir / "vox_aac.zip"

        if not zip_uri.is_file() and not (self.data_dir / "vox_aac").is_dir():
            partspaths = self.dl(parts)
            cat(partspaths, dest=zip_uri)
            rm(partspaths)

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
    """
    VoxCeleb2 contains over 1 million utterances for 6112 celebrities,
    extracted from videos uploaded to YouTube.

    http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html

    Citiation
    ---------
    @InProceedings{Chung18b,
      author       = "Chung, J.~S. and Nagrani, A. and Zisserman, A.", title        = "VoxCeleb2: Deep Speaker Recognition",
      booktitle    = "INTERSPEECH",
      year         = "2018",
    }
    """

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
    """
    VoxConverse is an audio-visual diarisation dataset consisting of over 50 hours
    of multispeaker clips of human speech, extracted from YouTube videos.

    http://www.robots.ox.ac.uk/~vgg/data/voxconverse/

    Citation
    --------
    @Article{Nagrani19,
      author       = "Joon~Son Chung and Jaesung Huh and Arsha Nagrani and Triantafyllos Afouras and Andrew Zisserman",
      title        = "Spot the conversation: speaker diarisation in the wild",
      journal      = "ArXiv",
      year         = "2020",
    }
    """

    def prepare(self):
        urls = [
            # Development Set
            "http://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip",
            # Test set
            "http://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_challengetest_wav.zip",
            # Labels
            "https://github.com/joonson/voxconverse/archive/master.zip",
        ]

        extract(self.dl(urls))
