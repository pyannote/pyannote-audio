import numpy as np
from pyannote.core import SlidingWindowFeature
import random
import torch

class SpecAugmentor(object):
    """2 way spectrogram augmentation :
        1) Frequency masking
        2) Time masking
    """

    def _spec_augment(mel_spectrogram, frequency_masking_para=27,
                      time_masking_para=100, frequency_mask_num=1, time_mask_num=1, scheduler=None,
                      epoch=None, max_epoch=None):
        """Spectrogram augmentation function from https://arxiv.org/abs/1904.08779
        3 steps processing :
        i) Frequency masking : depending on parameter frequency_masking_para and frequency_mask_num
        ii) Time masking : depending on parameter time_masking_para and time_mask_num

        Parameters
        ----------
        mel_spectrogram :       `SlidingWindowFeature` or (n_samples, n_features ) `numpy.ndarray`
            Features.
        frequency_masking_para: frequency mask parameter F
        time_masking_para:      time mask parameter T
        frequency_mask_num:     number of frequency masks m_F
        time_mask_num:          number of time masks m_T

        Returns
        -------
        masked_melspec : `SlidingWindowFeature` or (n_samples, n_features ) `numpy.ndarray`
            augmented spectrogram
        """
        v = mel_spectrogram.shape[0]
        tau = mel_spectrogram.shape[1]

        if scheduler is not None and max_epoch is not None:
            frequency_masking_para = int(frequency_masking_para * min(epoch, max_epoch)/max_epoch)
            time_masking_para = int(time_masking_para * min(epoch, max_epoch)/max_epoch)

        augmented_mel_spectrogram = mel_spectrogram.copy()

        # 1) Frequency masking
        for i in range(frequency_mask_num):
            f = np.random.uniform(low=0.0, high=frequency_masking_para)
            f = int(f)
            f0 = random.randint(0, v - f)
            augmented_mel_spectrogram[f0:f0 + f, :] = 0

        # 2) Time masking
        for i in range(time_mask_num):
            t = np.random.uniform(low=0.0, high=time_masking_para)
            t = int(t)
            t0 = random.randint(0, tau - t)
            augmented_mel_spectrogram[:, t0:t0 + t] = 0

        return augmented_mel_spectrogram

    def __call__(self, features, frequency_masking_para,
                 time_masking_para, nb_frequency_masks, nb_time_masks, scheduler=None,
                 epoch=None, max_epoch=None,
                 sliding_window=None):
        """Apply Google specAugment

        Parameters
        ----------
        features : `SlidingWindowFeature` or (n_samples, n_features ) `numpy.ndarray`
            Features.
        sliding_window : `SlidingWindow`, optional
            Not used.

        Returns
        -------
        normalized : `SlidingWindowFeature` or (n_samples, n_features ) `numpy.ndarray`
            Standardized features
        """

        if isinstance(features, SlidingWindowFeature):
            spec = features.data
        else:
            spec = features

        spec = SpecAugmentor._spec_augment(mel_spectrogram=spec,
                                           frequency_masking_para=frequency_masking_para,
                                           time_masking_para=time_masking_para,
                                           frequency_mask_num=nb_frequency_masks,
                                           time_mask_num=nb_time_masks,
                                           scheduler=scheduler,
                                           epoch=epoch,
                                           max_epoch=max_epoch)

        if isinstance(features, SlidingWindowFeature):
            return SlidingWindowFeature(spec, features.sliding_window)
        else:
            return spec
