# The MIT License (MIT)
#
# Copyright (c) 2025- pyannoteAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from pyannote.audio.core.io import Audio, AudioFile


def _prepare_audio(file: AudioFile) -> dict:
    """Apply channel selection before handing audio to a mono-only backend.

    Parameters
    ----------
    file : AudioFile
        Audio file with an optional channel selection policy.

    Returns
    -------
    audio : dict
        Selected mono waveform as a NumPy array and its sample rate.

    Raises
    ------
    ValueError
        If the selected waveform contains more than one channel.
    """
    waveform, sample_rate = Audio()(file)
    if waveform.shape[0] != 1:
        raise ValueError(
            'pyannoteAI backends require mono audio. Use channel="downmix", '
            '"random", or a fixed channel index instead of "all" for multi-channel audio.'
        )
    return {"waveform": waveform.numpy(force=True), "sample_rate": sample_rate}
