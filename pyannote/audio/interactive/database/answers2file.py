from pathlib import Path

from pyannote.core import Annotation, Segment, Timeline


def isInFiles(path, files):
    for file in files:
        if path in file["audio"]:
            return True

    return False


def idAnnotation(path, files):
    for i, file in enumerate(files):
        if path in file["audio"]:
            return i
    return False


# Type : accepted, ignore, flagged
def answers2file(answers, type):
    validation_files = []

    for answer in answers:
        if (answer["answer"] == type) or (
            (type == "flagged") and ("flagged" in answer.keys())
        ):
            path = answer["path"]
            uri = Path(path)
            uri = uri.stem
            if not isInFiles(path, validation_files):
                validation_files.append({"uri": uri, "audio": path})
            id = idAnnotation(path, validation_files)
            # Create annotation
            if "annotation" in validation_files[id]:
                annotation = validation_files[id]["annotation"]
            else:
                annotation = Annotation()

            debut = answer["chunk"]["start"]
            fin = answer["chunk"]["end"]

            for a in answer["audio_spans"]:
                annotation[Segment(a["start"], a["end"])] = a["label"]
            validation_files[id]["annotation"] = annotation
            # Create timeline
            if "annotated" in validation_files[id]:
                timeline = validation_files[id]["annotated"]
            else:
                timeline = Timeline()
            timeline.add(Segment(debut, fin))
            validation_files[id]["annotated"] = timeline
    # Fusion
    for file in validation_files:
        file["annotation"] = file["annotation"].support()
        file["annotated"] = file["annotated"].support()
        file["annotated"].uri = file["uri"]

    return validation_files
