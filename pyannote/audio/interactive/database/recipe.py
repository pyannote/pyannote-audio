import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import prodigy
import yaml
from prodigy.components.db import connect

from .answers2file import answers2file


def comparePath(array_path):
    minimum_path = min(array_path, key=lambda x: len(x.parts))
    for i, dir in enumerate(minimum_path.parts):
        for path in array_path:
            if not path.parts[i] == dir:
                i = i - 1
                break
        else:
            continue
        break
    return Path(*minimum_path.parts[0 : i + 1])


@prodigy.recipe(
    "pyannote.database",
    dataset=("Dataset where the annotations are", "positional", None, str),
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
    path: str,
    filter: Optional[str] = "accept",
) -> Dict[str, Any]:

    db = connect("sqlite", {})
    annotations = db.get_dataset(dataset)

    if annotations is None:
        sys.exit("'" + dataset + "' dataset not found")

    path = path + "/" + dataset + "/"
    Path(path).mkdir(parents=True, exist_ok=True)
    validation_file = answers2file(annotations, filter)
    now = datetime.now()
    name = now.strftime("%Y-%m-%d-%H%M%S")
    path_dtb = set()
    info = {"Databases": {}, "Protocols": {}}
    all_suffix = set()

    for file in validation_file:
        audio = Path(file["audio"])
        path_dtb.add(audio.parent)
        all_suffix.add(audio.suffix)

    main_path = comparePath(list(path_dtb))

    info["Databases"][dataset] = [
        str(main_path) + "/{uri}" + suffix for suffix in all_suffix
    ]
    rttm = name + ".rttm"
    lst = name + ".lst"
    uem = name + ".uem"
    info["Protocols"][dataset] = {
        "SpeakerDiarization": {
            name: {"train": {"annotation": rttm, "uri": lst, "annotated": uem}}
        }
    }

    with open(path + rttm, "w") as rttmfile, open(path + uem, "w") as uemfile, open(
        path + lst, "w"
    ) as lstfile:
        for file in validation_file:
            audio = Path(file["audio"])
            uri = audio.relative_to(main_path)
            annotation = file["annotation"]
            annotation.uri = uri
            annotation.write_rttm(rttmfile)
            annotated = file["annotated"]
            annotated.uri = uri
            annotated.write_uem(uemfile)
            lstfile.write(str(uri) + "\n")

    with open(path + "database.yml", "w") as conffile:
        yaml.dump(info, conffile)
