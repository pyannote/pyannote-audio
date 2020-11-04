import io
import wave

from pyannote.audio.core.data import apply_to_array


@apply_to_array
def normalize_wav(input_file):
    """
    Better name for this?
    Can we just use torchaudio?
    """
    output_file = io.BytesIO()
    with wave.open(str(input_file), "rb") as r_wav, wave.open(
        output_file, "wb"
    ) as w_wav:
        w_wav.setparams(r_wav.getparams())
        w_wav.writeframes(r_wav.readframes(r_wav.getnframes()))

    # writing the new wav into a buffer, to prevent overwriting the original
    # file
    with open(input_file, "wb") as wav_file:
        wav_file.write(output_file.getvalue())
    try:
        from scipy.io.wavfile import read
    except ImportError:
        print(
            "Scipy not installed. "
            "Could not test if the file %s was properly fixed to work "
            "with the scipy wave read function" % input_file
        )
    else:
        # test-opening the file with scipy
        rate, data = read(input_file)
        print("%s has been properly reformated" % input_file)
