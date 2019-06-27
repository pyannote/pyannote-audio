from numpy.random import randint
from scipy.signal import resample
from .base import Augmentation
from pyannote.database import FileFinder
from pyannote.database import get_protocol


class Resampler(Augmentation):
    """Resampling data augmentation.

    Parameters
    ----------
    sample_rate : int.
        Sample rate.
    rsr_min : int, optional
        Minimum resampling rate value.
    rsr_max : int, optional
        Maximum resampling rate value.
    db_yml : str, optional
        Path to `pyannote.database` configuration file.
        See `pyannote.database.FileFinder` for more details.
    """

    def __init__(self, sample_rate, rsr_min=None, rsr_max=None, db_yml=None):
        super().__init__()

        if rsr_min is None:
            rsr_min = sample_rate * 0.9

        if rsr_max is None:
            rsr_max = sample_rate * 1.1

        self.resample_rate_min = rsr_min
        self.resample_rate_max = rsr_max
        self.sample_rate = sample_rate

        def stretch_audio(f):
            original = FileFinder(config_yml=db_yml)(f)
            resample_rate = randint(self.resample_rate_min,
                                    self.resample_rate_max)
            self.sample_rate_ratio = resample_rate / self.sample_rate
            resampled_len = int(len(original) * self.sample_rate_ratio)

            augmented = resample(original, resampled_len)
            return augmented

        def stretch_annotations(f):
            stretched_annot = f['annotation'].copy()
            for segment, label in\
                    stretched_annot.itersegments(yield_label=True):
                start, end = segment
                start *= self.sample_rate_ratio
                end *= self.sample_rate_ratio
            return stretched_annot

        preprocessors = {
            'audio': stretch_audio,
            'annotation': stretch_annotations}

        protocol = get_protocol(self.protocol,
                                preprocessors=preprocessors)
        self.files_ = list(getattr(protocol, self.subset)())

    def __call__(self, original, sample_rate):
        """Augment original waveform
        Parameters
        ----------
        original : `np.ndarray`
            (n_samples, n_channels) waveform.
        sample_rate : `int`
            Sample rate.
        Returns
        -------
        augmented : `np.ndarray`
            (n_samples, n_channels) resample-augmented waveform.
        """
        resample_rate = randint(self.resample_rate_min,
                                self.resample_rate_max)

        sample_rate_ratio = resample_rate / sample_rate
        augmented = resample(original, int(len(original) * sample_rate_ratio))

        # Save the augmented

        return augmented
