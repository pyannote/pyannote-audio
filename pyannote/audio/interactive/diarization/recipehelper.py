# The MIT License (MIT)
#
# Copyright (c) 2021-2022 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Jim Petiot
# Hervé Bredin

import random
from copy import deepcopy
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from pyannote.core import Annotation, Segment
from scipy.spatial.distance import cdist

from pyannote.audio import Audio, Inference, Pipeline

from ..common.utils import AudioForProdigy, get_audio_spans, get_chunks


class RecipeHelper:
    def __init__(self, embedding="pyannote/embedding"):
        self.inference = Inference(embedding, window="whole")
        dim = self.inference.model.introspection.dimension
        self.speakers = np.array(
            [], dtype=[("name", "U100"), ("embedding", "f4", dim), ("nb", "i4")]
        )
        self.buffer = {}

    def on_exit(self, controller):
        now = datetime.now()
        date_time = now.strftime("%d-%m-%Y")
        name = "embeddings_" + date_time
        np.save(name, self.speakers)

    def update(self, answers):
        eg = answers[0]
        for span in eg["audio_spans"]:
            if span["label"] in eg:
                speaker = eg[span["label"]]
                span["label"] = eg[span["label"]]
            else:
                if span["label"].startswith("SPEAKER_"):
                    speaker = "G" + span["label"]
                else:
                    speaker = span["label"]

            speaker = speaker.strip()
            segment = Segment(
                span["start"] + eg["chunk"]["start"], span["end"] + eg["chunk"]["start"]
            )
            audio_for_pipeline = Audio(mono=True)
            wav, sample_rate = audio_for_pipeline.crop(eg["path"], segment, mode="pad")

            if speaker not in self.buffer:
                empty_waveform = torch.Tensor([])
                self.buffer[speaker] = [empty_waveform, 0]

            if self.buffer[speaker][1] + segment.duration >= 5:

                combine_waveform = torch.cat((wav, self.buffer[speaker][0]), dim=1)

                embedding = self.getEmb(combine_waveform, sample_rate)

                empty_waveform = torch.Tensor([])
                self.buffer[speaker] = [empty_waveform, 0]

                if not np.isnan(embedding).any():
                    if speaker in self.speakers["name"]:
                        i = np.where(self.speakers["name"] == speaker)
                        i = i[0][0]
                        self.speakers[i]["embedding"] = (
                            (self.speakers[i]["nb"] * self.speakers[i]["embedding"])
                            + embedding
                        ) / (self.speakers[i]["nb"] + 1)
                        self.speakers[i]["nb"] += 1
                    else:
                        size = self.speakers.size + 1
                        self.speakers.resize(size, refcheck=False)
                        self.speakers[self.speakers.size - 1] = (speaker, embedding, 1)
            else:
                combine_waveform = torch.cat((wav, self.buffer[speaker][0]), dim=1)
                duration = self.buffer[speaker][1] + segment.duration
                self.buffer[speaker] = [combine_waveform, duration]

    def validate_answer(self, eg):
        for field in eg["config"]["blocks"]:
            if "field_id" in field:
                speaker = field["field_id"]
                if speaker in eg and eg[speaker] != "":
                    flag = False
                    for span in eg["audio_spans"]:
                        if span["label"] == speaker:
                            flag = True
                            break
                    assert flag, "You have named a speaker not present in the audio"

    def getEmb(self, wav, sample_rate):
        try:
            embedding = self.inference({"waveform": wav, "sample_rate": sample_rate})
        except (RuntimeError, ValueError):
            embedding = [float("nan")]
        return embedding

    def stream(
        self,
        pipeline: Pipeline,
        source: Path,
        labels: List[str],
        chunk: float = 30.0,
        randomize: bool = False,
    ) -> Iterable[Dict]:
        """Stream for `pyannote.audio` recipe

        Applies pretrained pipeline and sends the results for manual correction

        Parameters
        ----------
        pipeline : Pipeline
            Pretrained pipeline.
        source : Path
            Directory containing audio files to process.
        labels : list of string
            List of expected pipeline labels.
        chunk : float, optional
            Duration of chunks, in seconds. Defaults to 30s.

        Yields
        ------
        task : dict
            Prodigy task with the following keys:
            "path" : path to audio file
            "chunk" : chunk start and end times
            "audio" : base64 encoding of audio chunk
            "text" : chunk identifier "{filename} [{start} {end}]"
            "audio_spans" : list of audio spans {"start": ..., "end": ..., "label": ...}
            "audio_spans_original" : deep copy of "audio_spans"
            "meta" : metadata displayed in Prodigy UI {"file": ..., "start": ..., "end": ...}
            "config": {"labels": list of labels}
        """

        context = getattr(pipeline, "context", 2.5)

        audio_for_prodigy = AudioForProdigy()
        audio_for_pipeline = Audio(mono=True)

        chunks = get_chunks(source, chunk_duration=chunk)
        if randomize:
            chunks = list(chunks)
            random.shuffle(chunks)

        for file, excerpt in chunks:

            path = file["path"]
            filename = file["text"]
            text = f"{filename} [{excerpt.start:.1f} - {excerpt.end:.1f}]"

            # load contextualized audio excerpt
            excerpt_with_context = Segment(
                start=excerpt.start - context, end=excerpt.end + context
            )
            waveform_with_context, sample_rate = audio_for_pipeline.crop(
                path, excerpt_with_context, mode="pad"
            )

            # run pipeline on contextualized audio excerpt
            output: Annotation = pipeline(
                {"waveform": waveform_with_context, "sample_rate": sample_rate}
            )

            # crop, shift, and format output for visualization in Prodigy
            audio_spans = get_audio_spans(
                output, excerpt, excerpt_with_context=excerpt_with_context
            )

            # load audio excerpt for visualization in Prodigy
            audio = audio_for_prodigy.crop(path, excerpt)

            labels = sorted(set(labels) | set(output.labels()))
            new_labels = []

            # group by label
            audio_spans = sorted(audio_spans, key=lambda x: x["label"])
            for label, segments in groupby(audio_spans, key=lambda x: x["label"]):

                combine_waveform = torch.Tensor([])
                for segment in list(segments):
                    waveform, sample_rate = audio_for_pipeline.crop(
                        path,
                        Segment(
                            segment["start"] + excerpt.start,
                            segment["end"] + excerpt.start,
                        ),
                    )
                    combine_waveform = torch.cat((combine_waveform, waveform), dim=1)

                embedding = self.getEmb(combine_waveform, sample_rate)

                if not np.isnan(embedding).any():
                    try:
                        distances = cdist(
                            self.speakers["embedding"],
                            embedding.reshape(1, -1),
                            metric="cosine",
                        )
                        index_min = np.argmin(distances)

                    except ValueError:
                        distances = [1000]
                        index_min = 0

                    # TODO update thresold according to the result
                    if distances[index_min] < 0.5:
                        genlabel = self.speakers[index_min]["name"]
                        if genlabel not in new_labels:
                            new_labels.append(genlabel)
                        for span in audio_spans:
                            if span["label"] == label:
                                span.update({"label": genlabel})

            blocks = [{"view_id": "audio_manual"}]
            new_labels = sorted(new_labels)
            all_labels = new_labels + labels
            for label in all_labels:
                blocks.append(
                    {
                        "view_id": "text_input",
                        "field_id": label,
                        "field_label": label,
                        "field_rows": 1,
                        "field_placeholder": label,
                        "field_suggestions": list(self.speakers["name"]),
                    }
                )

            yield {
                "path": path,
                "text": text,
                "audio": audio,
                "audio_spans": audio_spans,
                "audio_spans_original": deepcopy(audio_spans),
                "chunk": {"start": excerpt.start, "end": excerpt.end},
                "config": {"labels": all_labels, "blocks": blocks},
                "meta": {
                    "file": filename,
                    "start": f"{excerpt.start:.1f}",
                    "end": f"{excerpt.end:.1f}",
                },
            }
