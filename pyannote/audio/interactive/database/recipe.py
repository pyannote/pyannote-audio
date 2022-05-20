from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import prodigy
import yaml
from prodigy.components.db import connect

from .answers2file import answers2file


@prodigy.recipe(
    "pyannote.database",
    dataset=("Dataset where the annotations are", "positional", None, str),
    database=("Database name", "positional", None, str),
    path=("Where to save files", "positional", None, str),
    filter=(
        "Filter on specific answers",
        "option",
        None,
        str,
    ),
)
def database(
    dataset: str,
    database: str,
    path: str,
    filter: Optional[str] = "accept",
) -> Dict[str, Any]:

    db = connect("sqlite", {})
    annotations = db.get_dataset(dataset)
    validation_file = answers2file(annotations, filter)
    now = datetime.now()
    date_time = now.strftime("%d-%m-%Y_%H-%M-%S")
    name = "upTo_" + date_time
    path_dtb = set()
    info = {"Databases": {}, "Protocols": {}}

    for file in validation_file:
        audio = Path(file["audio"])
        p = str(audio.parent) + "/{uri}" + audio.suffix
        path_dtb.add(p)

    for i, p in enumerate(path_dtb):
        nd = database
        nf = name
        if i > 0:
            nd = database + str(i)
            nf = name + "_" + str(i)
        info["Databases"][nd] = p
        rttm = nf + ".rttm"
        lst = nf + ".lst"
        uem = nf + ".uem"
        info["Protocols"][nd] = {
            "SpeakerDiarization": {
                nf: {"train": {"annotation": rttm, "uri": lst, "annotated": uem}}
            }
        }

        with open(path + "/" + rttm, "w") as rttmfile, open(
            path + "/" + uem, "w"
        ) as uemfile, open(path + "/" + lst, "w") as urifile:
            for file in validation_file:
                audio = Path(file["audio"])
                fp = str(audio.parent) + "/{uri}" + audio.suffix
                if fp == p:
                    fname = file["uri"]
                    for seg in file["annotated"]:
                        uemfile.write(
                            fname + " NA " + str(seg.start) + " " + str(seg.end) + "\n"
                        )
                    annotation = file["annotation"]
                    for seg in annotation.get_timeline():
                        rttmfile.write(
                            "SPEAKER "
                            + fname
                            + " NA "
                            + str(seg.start)
                            + " "
                            + str(seg.end - seg.start)
                            + " <NA> <NA> "
                            + "".join([label for label in annotation.get_labels(seg)])
                            + " <NA> <NA> \n"
                        )
                    urifile.write(fname + "\n")

    with open(path + "configuration.yml", "w") as conffile:
        yaml.dump(info, conffile)
