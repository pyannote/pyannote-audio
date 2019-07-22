from pyannote.audio.train.callback import Callback
import numpy as np
import random

class SpecAugmentCallback(Callback):
    """
    Callback for spectrogram augmentation. Two-step process :

    1) Apply frequency mask(s)
    2) Apply time mask(s)
    (3) Time warping) : Not implemented yet. Shown as leading to a small improvement
    in the reference.

    Reference
    ---------
    https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html


    Use case
    --------
    Here's what expected in config.yml :

    callbacks:
      - name: pyannote.audio.augmentation.spec_augment_callback.SpecAugmentCallback
        params:
          time_masking_para: 100
          frequency_masking_para: 27
          nb_time_masks: 1
          nb_frequency_masks: 1

    """

    def __init__(self, frequency_masking_para=27, time_masking_para=100,
                 nb_frequency_masks=1, nb_time_masks=1):
        """
        Initialize spectrogram augmentation callback class

        :param self.frequency_masking_para:  Maximal size of the frequency mask, in number of frames
                                        (random between 0 and self.frequency_masking_para)
        :param time_masking_para:       Maximal size of the time mask, in number of frames
                                        (random between 0 and time_masking_para)
        :param nb_frequency_masks:      Number of frequency masks
        :param nb_time_masks:           Number of time masks
        """
        super().__init__()
        self.frequency_masking_para = frequency_masking_para
        self.time_masking_para = time_masking_para
        self.nb_frequency_masks = nb_frequency_masks
        self.nb_time_masks = nb_time_masks

    def on_train_start(self, trainer):
        nb_frames = float(trainer.batch_generator_.duration) / float(trainer.batch_generator_.frame_info.step)

        # We don't want too wide time masks (same as in the google ref.)
        self.time_masking_para = int(min(0.2*nb_frames, self.time_masking_para))

    def on_batch_start(self, trainer, batch):
        for i, spec in enumerate(batch['X']):

            tau = spec.shape[0]
            v = spec.shape[1]

            augmented_spec = spec.copy()

            # 1) Frequency masking
            for i in range(self.nb_frequency_masks):
                f = np.random.uniform(low=0.0, high=self.frequency_masking_para)
                f = int(f)
                f0 = random.randint(0, v - f)
                augmented_spec[:, f0:f0 + f] = 0

            # 2) Time masking
            for i in range(self.nb_time_masks):
                t = np.random.uniform(low=0.0, high=self.time_masking_para)
                t = int(t)
                t0 = random.randint(0, tau - t)
                augmented_spec[t0:t0 + t, :] = 0

            batch['X'][i] = augmented_spec

        return batch
