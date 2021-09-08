from typing import List, Optional, Union, Iterable, Dict, Any
import prodigy
from prodigy.core import recipe
from prodigy.components.loaders import get_stream
from prodigy.components.preprocess import fetch_media as fetch_media_preprocessor
from prodigy.util import log, msg, get_labels, split_string

from pathlib import Path
from typing import Text, Dict, List, Iterable
from pyannote.core import Segment, Timeline, Annotation

from prodigy.components.loaders import Audio
from prodigy.components.db import connect
from utils import SAMPLE_RATE
from utils import normalize
from utils import to_base64
from utils import to_audio_spans
from utils import chunks
from copy import deepcopy

from pyannote.audio.core.io import Audio as AudioFile
from pyannote.audio.pipelines import VoiceActivityDetection

def remove_base64(examples):
    """Remove base64-encoded string if "path" is preserved in example."""
    for eg in examples:
        if "audio" in eg and eg["audio"].startswith("data:") and "path" in eg:
            eg["audio"] = eg["path"]
        if "video" in eg and eg["video"].startswith("data:") and "path" in eg:
            eg["video"] = eg["path"]
    return examples
    
def sad_manual_stream(pipeline: VoiceActivityDetection, source: Path, chunk: float = 10.0) -> Iterable[Dict]:
    """Stream for pyannote.sad.manual recipe
    Applies (pretrained) speech activity detection and sends the results for
    manual correction chunk by chunk.
    Parameters
    ----------
    pipeline : VoiceActivityDetection
        Pretrained speaker diarization interactive pipeline.
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
        "recipe" : "pyannote.sad.manual"
    """

    raw_audio = AudioFile(sample_rate=SAMPLE_RATE, mono=True)

    for audio_source in Audio(source):

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
                # this is needed by other recipes
                "recipe": "pyannote.sad.manual",
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
                    # this is needed by other recipes
                    "recipe": "pyannote.sad.manual",
                }


@recipe(
    "pyannote.sad.manual",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    chunk=("split long audio files into shorter chunks of that many seconds each","option",None,float),
    loader=("Loader to use", "option", "lo", str),
    #label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    keep_base64=("If 'audio' loader is used: don't remove base64-encoded data from the data on save", "flag", "B", bool),
    autoplay=("Autoplay audio when a new task loads", "flag", "A", bool),
    fetch_media=("Convert URLs and local paths to data URIs", "flag", "FM", bool),
    exclude=("Comma-separated list of dataset IDs whose annotations to exclude", "option", "e", split_string),
    # fmt: on
)
def sad_manual(
    dataset: str,
    source: Union[str, Iterable[dict]],
    loader: Optional[str] = "audio",
    chunk: float = 30.0,
    autoplay: bool = False,
    keep_base64: bool = False,
    fetch_media: bool = False,
    exclude: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Stream in audio files and manually label regions for the given
    labels. Uses the 'audio' loader by default, which encodes the data as base64
    (and removes it from the task afterwards, unless you set --keep-base64).
    Alternatively, you can also use the loaders 'audio-server', 'video',
    'video-server' or 'jsonl' (to stream in pre-formatted data from JSONL).
    """
    log("RECIPE: Starting recipe pyannote.sad.manual", locals())

    label=['Speech']

    pipeline = VoiceActivityDetection(segmentation="pyannote/segmentation")
    HYPER_PARAMETERS = {"onset": 0.5, "offset": 0.5,"min_duration_on": 0.0,"min_duration_off": 0.0}
    pipeline.instantiate(HYPER_PARAMETERS)

    return {
        "view_id": "audio_manual",
        "dataset": dataset,
        "stream": sad_manual_stream(pipeline, source, chunk=chunk),
        "before_db": remove_base64 if not keep_base64 else None,
        "exclude": exclude,
        "config": {
            "labels": label,
            "audio_autoplay": autoplay,
            "force_stream_order": True,
            "show_audio_minimap": False,
        },
    }
