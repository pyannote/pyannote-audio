from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import prodigy
import yaml
from prodigy.components.db import connect

from .answers2file import answers2file


def comparePath(array_path):
    minimum_path = list(array_path[0].parts)
    for path in array_path[1:]:
        list_path = list(path.parts)
        for i, p in enumerate(minimum_path):
            if minimum_path[i] != list_path[i]:
                minimum_path = minimum_path[0:i]
                break
    return Path(*minimum_path)


def getUri(p1, p2):
    for dir in p2:
        p1.remove(dir)

    return Path(*p1)


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

    path = path + "/" + dataset + "/"
    Path(path).mkdir(parents=True, exist_ok=True)

    db = connect("sqlite", {})
    annotations = db.get_dataset(dataset)
    validation_file = answers2file(annotations, filter)
    now = datetime.now()
    name = now.strftime("%Y-%m-%d-%H%M%S")
    path_dtb = set()
    info = {"Databases": {}, "Protocols": {}}
    all_suffix = set()

    for file in validation_file:
        audio = Path(file["audio"])
        p = str(audio.parent) + "/{uri}" + audio.suffix
        path_dtb.add(Path(p))
        all_suffix.add(audio.suffix)

    main_path = comparePath(list(path_dtb))

    info["Databases"][database] = [
        str(main_path) + "/{uri}" + suffix for suffix in all_suffix
    ]
    rttm = name + ".rttm"
    lst = name + ".lst"
    uem = name + ".uem"
    info["Protocols"][database] = {
        "SpeakerDiarization": {
            name: {"train": {"annotation": rttm, "uri": lst, "annotated": uem}}
        }
    }

    with open(path + rttm, "w") as rttmfile, open(path + uem, "w") as uemfile, open(
        path + lst, "w"
    ) as urifile:
        for file in validation_file:
            audio = Path(file["audio"])

            uri_path = getUri(list(audio.parent.parts), list(main_path.parts))

            if uri_path != Path("."):
                fname = str(uri_path) + "/" + file["uri"]
            else:
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

    with open(path + "database.yml", "w") as conffile:
        yaml.dump(info, conffile)
