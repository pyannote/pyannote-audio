import warnings

from pyannote.audio.core.io import Audio
from pyannote.database import FileFinder, Protocol, get_annotated

get_duration = Audio().get_duration


def check_protocol(protocol: Protocol) -> Protocol:
    """Check that protocol is suitable for training a model

        - does it provide a training set?
        - does it provide a validation set?
        - does it provide a way to access audio content?
        - does it provide a way to delimit annotated content?

    Returns
    -------
    fixed_protocol : Protocol
    has_validation : bool

    Raises
    ------
    ValueError if protocol does not pass the check list and cannot be fixed.

    """

    # does protocol define a training set?
    try:
        file = next(protocol.train())
    except (AttributeError, NotImplementedError):
        msg = f"Protocol {protocol.name} does not define a training set."
        raise ValueError(msg)

    # does protocol provide audio keys?
    if "audio" not in file:
        if "waveform" in file:
            if "sample_rate" not in file:
                msg = f'Protocol {protocol.name} provides audio with "waveform" key but is missing a "sample_rate" key.'
                raise ValueError(msg)
        else:
            file_finder = FileFinder()
            try:
                _ = file_finder(file)
            except (KeyError, FileNotFoundError):
                msg = (
                    f"Protocol {protocol.name} does not provide the path to audio files. "
                    f"See pyannote.database documentation on how to add an 'audio' preprocessor."
                )
                raise ValueError(msg)
            else:
                protocol.preprocessors["audio"] = file_finder
                msg = (
                    f"Protocol {protocol.name} does not provide the path to audio files: "
                    f"adding an 'audio' preprocessor for you. See pyannote.database documentation "
                    f"on how to do that yourself."
                )
                warnings.warn(msg)

    if "annotated" not in file:

        if "duration" not in file:
            protocol.preprocessors["duration"] = get_duration

        protocol.preprocessors["annotated"] = get_annotated

        msg = (
            f"Protocol {protocol.name} does not provide the 'annotated' regions: "
            f"adding an 'annotated' preprocessor for you. See pyannote.database documentation "
            f"on how to do that yourself."
        )
        warnings.warn(msg)

    # does protocol define a validation set?
    try:
        file = next(protocol.development())
    except (AttributeError, NotImplementedError):
        has_validation = False
    else:
        has_validation = True

    return protocol, has_validation
