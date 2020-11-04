from functools import partial

from pyannote.audio.core.data import DownloadableProtocol, chain, download

from .validate import normalize_wav


class AMIHeadsetMix(DownloadableProtocol):
    def __init__(self, *args, **kwargs):
        protocol_name = "MixHeadset"
        database_glob = "./AMIHeadsetMix/{uri}.wav"
        protocol_entries = {
            "train": {
                "uri": "./MixHeadset.train.lst",
                "annotation": "./MixHeadset.train.rttm",
                "annotated": "./MixHeadset.train.uem",
            },
            "development": {
                "uri": "./MixHeadset.development.lst",
                "annotation": "./MixHeadset.development.rttm",
                "annotated": "./MixHeadset.development.uem",
            },
            "test": {
                "uri": "./MixHeadset.test.lst",
                "annotation": "./MixHeadset.test.rttm",
                "annotated": "./MixHeadset.test.uem",
            },
        }

        super().__init__(
            task="SpeakerDiarization",
            protocol_name=protocol_name,
            protocol_entries=protocol_entries,
            database_glob=database_glob,
            *args,
            **kwargs,
        )

    def prepare(self):
        require_validation = [
            "IS1004d",
            "IS1006c",
            "IS1008d",
            "IS1007a",
            "IS1008c",
            "IS1005a",
            "IS1008b",
            "IS1009b",
            "IS1002c",
            "IS1004b",
            "IS1008a",
            "IS1004a",
            "IS1005b",
            "IS1007c",
            "IS1004c",
            "IS1009c",
            "IS1003c",
            "IS1009a",
            "IS1009d",
            "IS1006a",
            "IS1005c",
            "IS1006d",
            "IS1003d",
            "IS1007b",
        ]

        no_validation = [
            "ES2003c",
            "ES2004a",
            "TS3003d",
            "ES2005b",
            "ES2015a",
            "TS3005b",
            "ES2009c",
            "ES2007d",
            "IS1000c",
            "TS3007a",
            "ES2007c",
            "IN1008",
            "ES2014c",
            "IS1007d",
            "TS3006a",
            "ES2009d",
            "TS3012b",
            "TS3008a",
            "EN2001d",
            "ES2016c",
            "ES2014a",
            "ES2010d",
            "IN1009",
            "TS3005a",
            "EN2002a",
            "TS3008d",
            "IN1005",
            "TS3011c",
            "ES2016d",
            "IS1003b",
            "ES2002d",
            "ES2011b",
            "TS3004d",
            "TS3011d",
            "TS3011b",
            "ES2013b",
            "ES2016b",
            "ES2006c",
            "TS3006b",
            "IB4002",
            "TS3008c",
            "ES2010c",
            "IS1000b",
            "ES2008b",
            "TS3012d",
            "IB4001",
            "ES2011a",
            "IS1002d",
            "ES2013d",
            "EN2002d",
            "TS3005c",
            "TS3006d",
            "EN2006a",
            "ES2010a",
            "ES2012d",
            "ES2014b",
            "ES2013a",
            "TS3004b",
            "ES2005c",
            "TS3010c",
            "ES2003b",
            "EN2004a",
            "ES2016a",
            "ES2004c",
            "EN2001a",
            "IS1001b",
            "TS3006c",
            "ES2006b",
            "ES2008d",
            "IN1007",
            "ES2014d",
            "TS3004c",
            "TS3003b",
            "IS1001d",
            "EN2001b",
            "IS1000a",
            "ES2004d",
            "ES2008c",
            "IB4004",
            "IS1000d",
            "IN1013",
            "EN2001e",
            "EN2006b",
            "ES2007a",
            "ES2003d",
            "ES2015d",
            "TS3007c",
            "EN2002c",
            "IN1001",
            "TS3004a",
            "ES2012a",
            "EN2005a",
            "IN1014",
            "TS3005d",
            "ES2006d",
            "TS3009a",
            "IS1002b",
            "TS3009b",
            "IB4010",
            "ES2007b",
            "ES2006a",
            "ES2010b",
            "IS1006b",
            "ES2009b",
            "TS3010a",
            "EN2003a",
            "EN2009c",
            "IB4011",
            "IN1002",
            "IB4003",
            "ES2012b",
            "ES2011c",
            "TS3008b",
            "ES2008a",
            "IS1001c",
            "TS3012a",
            "ES2002b",
            "ES2005d",
            "IB4005",
            "ES2003a",
            "TS3010d",
            "ES2009a",
            "TS3009d",
            "TS3003a",
            "TS3011a",
            "ES2015b",
            "TS3007b",
            "ES2013c",
            "IS1001a",
            "IN1016",
            "TS3012c",
            "IS1003a",
            "TS3010b",
            "ES2002a",
            "ES2015c",
            "TS3009c",
            "ES2011d",
            "EN2009d",
            "ES2004b",
            "TS3003c",
            "ES2012c",
            "ES2002c",
            "ES2005a",
            "EN2002b",
            "IN1012",
            "TS3007d",
            "EN2009b",
        ]
        dl = partial(download, dest_dir=self.data_dir)

        fmt_string = "http://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{track}/audio/{track}.Mix-Headset.wav"
        downloads = []
        validate = []

        # Create links
        for track in no_validation + require_validation:
            url = fmt_string.format(track=track)
            if url in require_validation:
                validate.append(url)
            else:
                downloads.append(url)

        # Create links for the rttm, lst and uem files
        fmt_string = "https://raw.githubusercontent.com/pyannote/pyannote-audio/develop/tutorials/data_preparation/AMI/MixHeadset.{mode}.{ext}"
        for mode in ["train", "development", "test"]:
            for ext in ["uem", "rttm", "lst"]:
                url = fmt_string.format(mode=mode, ext=ext)
                downloads.append(url)

        # Download these files
        dl(downloads)

        # Download and Validate the others
        chain([dl, normalize_wav], validate)
