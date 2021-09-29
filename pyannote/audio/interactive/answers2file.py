# Que se passe-t-il si pas de Speech ?
# deux segment qui se collent c'est grave ?
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


def answers2file(answers):
    # IN
    # answers = [{'path': 'data/les_fruits04.wav', 'text': 'les_fruits04 [10.0, 20.0]', 'audio': 'data/les_fruits04.wav', 'audio_spans': [{'start': 1.0278124999999996, 'end': 5.330937500000001, 'label': 'Speech', 'id': 'b357ee23-e081-4bf8-995e-fe8878f91d74', 'color': 'gold'}, {'start': 6.225312500000001, 'end': 6.950937500000002, 'label': 'Speech', 'id': '2bdb16a0-3c98-4e9d-8365-0b75b97fd901', 'color': 'gold'}, {'start': 8.5709375, 'end': 9.2459375, 'label': 'Speech', 'id': '04492840-f321-4720-ae0e-48eb706b9ce1', 'color': 'gold'}], 'audio_spans_original': [{'start': 1.0278124999999996, 'end': 5.330937500000001, 'label': 'Speech'}, {'start': 6.225312500000001, 'end': 6.950937500000002, 'label': 'Speech'}, {'start': 8.5709375, 'end': 9.2459375, 'label': 'Speech'}], 'chunk': {'start': 10, 'end': 20}, 'meta': {'file': 'les_fruits04', 'start': '10.0', 'end': '20.0'}, 'recipe': 'pyannote.sad.manual', '_input_hash': 1243057469, '_task_hash': 894852733, '_session_id': None, '_view_id': 'audio_manual', 'answer': 'accept'}, {'path': 'data/les_fruits04.wav', 'text': 'les_fruits04 [20.0, 30.0]', 'audio': 'data/les_fruits04.wav', 'audio_spans': [{'start': 3.0934375000000003, 'end': 3.5996875000000017, 'label': 'Speech', 'id': '7d55e2e6-fdfd-4bdd-b0ce-180fd6834355', 'color': 'gold'}, {'start': 4.9496875, 'end': 5.7934375, 'label': 'Speech', 'id': 'dab518aa-58ae-4bf6-b901-ffee519c443e', 'color': 'gold'}, {'start': 6.7890625, 'end': 7.615937500000001, 'label': 'Speech', 'id': 'f89ba6ac-4e38-4ee6-a481-16171a746215', 'color': 'gold'}, {'start': 8.662187500000002, 'end': 9.4721875, 'label': 'Speech', 'id': '9bdc33b9-43f3-4a8c-9601-41194e1af667', 'color': 'gold'}], 'audio_spans_original': [{'start': 3.0934375000000003, 'end': 3.5996875000000017, 'label': 'Speech'}, {'start': 4.9496875, 'end': 5.7934375, 'label': 'Speech'}, {'start': 6.7890625, 'end': 7.615937500000001, 'label': 'Speech'}, {'start': 8.662187500000002, 'end': 9.4721875, 'label': 'Speech'}], 'chunk': {'start': 20, 'end': 30}, 'meta': {'file': 'les_fruits04', 'start': '20.0', 'end': '30.0'}, 'recipe': 'pyannote.sad.manual', '_input_hash': 213085811, '_task_hash': -339995964, '_session_id': None, '_view_id': 'audio_manual', 'answer': 'accept'}]
    # OUT
    validation_files = []

    for answer in answers:
        if answer["answer"] == "accept":
            path = answer["path"]
            if not isInFiles(path, validation_files):
                validation_files.append({"uri": path, "audio": path})
            id = idAnnotation(path, validation_files)
            # ANNOTATION
            if "annotation" in validation_files[id]:
                annotation = validation_files[id]["annotation"]
            else:
                annotation = Annotation()
            #################################
            debut = answer["chunk"]["start"]
            fin = answer["chunk"]["end"]
            #################################
            for a in answer["audio_spans"]:
                start = debut + a["start"]
                end = debut + a["end"]
                annotation[Segment(start, end)] = a["label"]
            validation_files[id]["annotation"] = annotation
            # TIMELINE
            if "annotated" in validation_files[id]:
                timeline = validation_files[id]["annotated"]
            else:
                timeline = Timeline()
            timeline.add(Segment(debut, fin))
            validation_files[id]["annotated"] = timeline

    # print(validation_files)
    return validation_files
