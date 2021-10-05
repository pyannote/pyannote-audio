from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import prodigy
from prodigy.components.loaders import Audio as AudioLoader
from utils import SAMPLE_RATE, chunks, normalize, to_audio_spans, to_base64

from pyannote.audio.core.io import Audio
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Annotation, Segment


def remove_base64(examples):
    """Remove base64-encoded string if "path" is preserved in example."""
    for eg in examples:
        if "audio" in eg and eg["audio"].startswith("data:") and "path" in eg:
            eg["audio"] = eg["path"]
        if "video" in eg and eg["video"].startswith("data:") and "path" in eg:
            eg["video"] = eg["path"]
    return examples


def voice_activity_detection_stream(
    pipeline: VoiceActivityDetection, source: Path, chunk: float = 10.0
) -> Iterable[Dict]:
    """
    Stream for pyannote.voice_activity_detection recipe
    Applies (pretrained) speech activity detection and sends the results for
    manual correction chunk by chunk.
    Parameters
    ----------
    pipeline : VoiceActivityDetection
        Pretrained speech activity detection pipeline.
    source : Path
        Directory containing audio files to process.
    chunk : float, optional
        Duration of chunks, in seconds. Defaults to 10s.
    Yields
    ------
    task : dict
        Prodigy task with the following keys:
        "path" : path to audio file
        "text" : name of audio file
        "chunk" : chunk start and end times
        "audio" : base64 encoding of audio chunk
        "audio_spans" : speech spans detected by pretrained SAD model
        "audio_spans_original" : copy of "audio_spans"
        "meta" : additional meta-data displayed in Prodigy UI
    """
    raw_audio = Audio(sample_rate=SAMPLE_RATE, mono=True)

    for audio_source in AudioLoader(source):

        path = audio_source["path"]
        text = audio_source["text"]
        file = {"uri": text, "database": source, "audio": path}

        duration = raw_audio.get_duration(file)
        file["duration"] = duration

        prodigy.log(f"RECIPE: detecting speech regions in '{path}'")

        speech: Annotation = pipeline(file)

        if duration <= chunk:
            waveform, sr = raw_audio.crop(file, Segment(0, duration))
            waveform = waveform.numpy()
            task_audio = to_base64(normalize(waveform), sample_rate=SAMPLE_RATE)
            task_audio_spans = to_audio_spans(speech)

            yield {
                "path": path,
                "text": text,
                "audio": task_audio,
                "audio_spans": task_audio_spans,
                "audio_spans_original": deepcopy(task_audio_spans),
                "chunk": {"start": 0, "end": duration},
                "meta": {"file": text},
            }

        else:
            for focus in chunks(duration, chunk=chunk, shuffle=False):
                task_text = f"{text} [{focus.start:.1f}, {focus.end:.1f}]"
                waveform, sr = raw_audio.crop(file, focus)
                waveform = waveform.numpy().T
                task_audio = to_base64(normalize(waveform), sample_rate=SAMPLE_RATE)
                task_audio_spans = to_audio_spans(
                    speech.crop(focus, mode="intersection"), focus=focus
                )

                yield {
                    "path": path,
                    "text": task_text,
                    "audio": task_audio,
                    "audio_spans": task_audio_spans,
                    "audio_spans_original": deepcopy(task_audio_spans),
                    "chunk": {"start": focus.start, "end": focus.end},
                    "meta": {
                        "file": text,
                        "start": f"{focus.start:.1f}",
                        "end": f"{focus.end:.1f}",
                    },
                }


@prodigy.recipe(
    "pyannote.voice_activity_detection",
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=(
        "Data to annotate (file path or '-' to read from standard input)",
        "positional",
        None,
        str,
    ),
    chunk=(
        "split long audio files into shorter chunks of that many seconds each",
        "option",
        None,
        float,
    ),
)
def voice_activity_detection(
    dataset: str,
    source: Union[str, Iterable[dict]],
    chunk: float = 10.0,
    segmentation_model: Optional[str] = "pyannote/segmentation",
    hyper_parameters: Optional[dict] = {
        "onset": 0.5,
        "offset": 0.5,
        "min_duration_on": 0.0,
        "min_duration_off": 0.0,
    },
) -> Dict[str, Any]:

    pipeline = VoiceActivityDetection(segmentation=segmentation_model, step=0.5)
    pipeline.instantiate(hyper_parameters)
    prodigy.log("RECIPE: Starting recipe voice_activity_detection", locals())

    return {
        "view_id": "audio_manual",
        "dataset": dataset,
        "stream": voice_activity_detection_stream(pipeline, source, chunk=chunk),
        "before_db": remove_base64,
        "config": {
            "labels": ["Speech"],
            "audio_autoplay": True,
            "show_audio_minimap": False,
        },
    }
