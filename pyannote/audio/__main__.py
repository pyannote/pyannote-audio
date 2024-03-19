# MIT License
#
# Copyright (c) 2022- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from pathlib import Path

import typer
from pyannote.core import Annotation

from pyannote.audio import Audio, Pipeline
from pyannote.audio.pipelines.utils.hook import Hooks, ProgressHook, TimingHook

app = typer.Typer()


@app.command("apply")
def apply(
    pipeline: str,
    audio_in: str,
    rttm_out: str,
):
    pretrained_pipeline = Pipeline.from_pretrained(pipeline)

    io = Audio()

    with open(rttm_out, "w") as rttm:
        audio_duration = io.get_duration(audio_in)

        file = {
            "audio": audio_in,
            "uri": Path(audio_in).stem,
        }

        rttm.write(f"# pipeline.path: {pipeline}\n")
        rttm.write(f"# audio.path: {audio_in}\n")
        rttm.write(f"# audio.duration: {audio_duration:.1f}s\n")

        with Hooks(ProgressHook(), TimingHook()) as hook:
            output: Annotation = pretrained_pipeline(file, hook=hook)

        time_total = file["timing_hook"]["total"]

        rttm.write(f"# processing.duration: {time_total:.1f}s\n")
        rttm.write(f"# processing.rtf: {time_total / audio_duration:.0%}\n")

        output.write_rttm(rttm)
        rttm.flush()


if __name__ == "__main__":
    app()

# python -m pyannote.audio apply pyannote/speaker-diarization-3.0 file.wav file.wav.rttm
