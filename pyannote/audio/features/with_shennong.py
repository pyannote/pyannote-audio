#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2018 CNRS

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

# AUTHORS
# Mathieu Bernard
# Julien Karadayi
# Marvin Lavechin

"""
Feature extraction with Shennong
--------------------------------
"""

from shennong.audio import Audio
from shennong.features.processor.mfcc import MfccProcessor
import numpy as np
from .base import FeatureExtraction
from pyannote.core.segment import SlidingWindow
from shennong.features.pipeline import get_default_config, extract_features
from shennong.features.processor.mfcc import MfccProcessor
from shennong.features.processor.bottleneck import BottleneckProcessor
from shennong.features.processor.filterbank import FilterbankProcessor
from shennong.features.postprocessor.delta import DeltaPostProcessor
from shennong.features.processor.pitch import (
    PitchProcessor, PitchPostProcessor)
from shennong.features.postprocessor.cmvn import CmvnPostProcessor

class ShennongFeatureExtraction(FeatureExtraction):
    """Shennong feature extraction base class

    Parameters
    ----------
    sample_rate : int, optional
        Defaults to 16000 (i.e. 16kHz)
    augmentation : `pyannote.audio.augmentation.Augmentation`, optional
        Data augmentation.
    duration : float, optional
        Defaults to 0.025.
    step : float, optional
        Defaults to 0.010.
    with_cmvn : bool, optional
        Defaults to False
    with_pitch: bool, optional
        Defaults to False
    """

    def __init__(self, sample_rate=16000, augmentation=None,
                 duration=0.025, step=0.01):

        super().__init__(sample_rate=sample_rate,
                         augmentation=augmentation)
        self.duration = duration
        self.step = step

        self.sliding_window_ = SlidingWindow(start=-.5*self.duration,
                                             duration=self.duration,
                                             step=self.step)

    def get_pitch(self, audio, fmin, fmax):
        """Extract pitch using shennong and output it as a set 
        of features. Can be concatenated with other sets of features
        (mfcc, filterbanks...).

        Parameters
        ----------
        fmin : int, optional
            min frequency for pitch estimation. Defaults to 20.
        fmax : int, optional
            max frequency for pitch estimation. Defaults to 500.

        Output
        ------
        pitch: array
            Pitch output is an array of shape array.shape = (n, 3) 
            where n is the number of mfcc frames.
        """
        # define pitch estimation parameters
        processor = PitchProcessor(frame_shift=self.step,
                                   frame_length=self.duration)
        processor.sample_rate = self.sample_rate
        processor.min_f0 = fmin
        processor.max_f0 = fmax

        # estimate pitch
        pitch = processor.process(audio)

        # post process pitch to output usable features (see shennong)
        postprocessor = PitchPostProcessor()
        postpitch = postprocessor.process(pitch)

        return postpitch

    def concatenate_with_pitch(self, feat, pitch):
        """ When the pitch and the mfcc are not of same length, 
            pad the pitch equally at the begining and at the end
            to match the sizes.
        """
        # get size difference
        n_difference = feat.shape[0] - pitch.shape[0]

        if n_difference > 0:
            # add ceil(n_difference/2) frames at start or pitch array
            # and floor(n_difference/2) frames at end of pitch array
            ceil = int(np.ceil(n_difference/2))
            floor = int(np.floor(n_difference/2))
            pitch = np.insert(pitch, 0, np.zeros((ceil, 3)), axis=0)
            pitch = np.insert(pitch, pitch.shape[0], np.zeros((floor, 3)), axis=0)

        # concatenate pitch and mfcc which are now the same size
        stack = np.concatenate((feat, pitch), axis=1)

        return stack

    def get_sliding_window(self):
            return self.sliding_window_

class ShennongFilterbank(ShennongFeatureExtraction):
    """Shennong Filterbank

    ::
            |  e   |
            | c1   |
            | c2   |  coefficients
            | c3   |
        x = | c4   |  coefficients first derivatives
            | ...  |
            |pitch1|
            |pitch2|  Coefficients of pitch estimation (if pitch is asked)
            |pitch3|


    Parameters
    ----------
    sample_rate : int, optional
        Defaults to 16000 (i.e. 16kHz)
    augmentation : `pyannote.audio.augmentation.Augmentation`, optional
        Data augmentation.
    duration : float, optional
        Defaults to 0.025.
    step : float, optional
        Defaults to 0.010.
    e : bool, optional
        Energy. Defaults to True.
    with_pitch: bool, optional
        Compute Pitch Estimation (w/ same step and Duration as MFCC).
        Defaults to True.
    melNbFilters = int, optional.
        Number of triangular mel-frequency bins. Defaults to 40.
    fftWindow = str, optional
        Windows used for FFT. Defaults to hanning.
    melLowFreq = int, optional.
        Frequency max for filter bins centers. Defaults to sampleFreq / 2 - 100.
    melHighFreq = int, optional.
        Minimal frequency for filter bins centers. Defaults to 20.



    """

    def __init__(self, sample_rate=16000, augmentation=None,
                 duration=0.025, step=0.01,
                 fftWindow='hanning',
                 melLowFreq=20,
                 melHighFreq=0,
                 pitchFmin=20,
                 pitchFmax=500,
                 e=False, D=True, DD=True,
                 melNbFilters=40,
                 with_pitch=True):

        super().__init__(sample_rate=sample_rate, augmentation=augmentation,
                         duration=duration, step=step)

        self.e = e
        self.with_pitch = with_pitch

        # pitch frequencies
        self.pitchFmin = pitchFmin
        self.pitchFmax = pitchFmax

        self.melNbFilters = melNbFilters
        self.fftWindow = fftWindow
        self.melLowFreq = melLowFreq
        self.melHighFreq = melHighFreq


    def get_context_duration(self):
        return 0.

    def get_features(self, y, sample_rate):
        """Feature extraction

        Parameters
        ----------
        y : (n_samples, 1) numpy array
            Waveform
        sample_rate : int
            Sample rate

        Returns
        -------
        data : (n_frames, n_dimensions) numpy array
            Features
        """
        # create audio object for shennong
        audio = Audio(data=y, sample_rate=sample_rate)

        # create filterbank processor
        processor = FilterbankProcessor(sample_rate=sample_rate)

        # use energy ?
        processor.use_energy = self.e

        # set parameters
        processor.frame_length = self.duration
        processor.frame_shift = self.step
        processor.window_type = self.fftWindow
        processor.low_freq = self.melLowFreq
        processor.high_freq = self.melHighFreq
        processor.num_bins = self.melNbFilters
        processor.snip_edges = False 

        # process audio to get filterbanks
        fbank = processor.process(audio)

        # Compute Pitch
        if self.with_pitch:
            # extract pitch
            pitch = self.get_pitch(audio, self.pitchFmin,
                                   self.pitchFmax)

            ## concatenate mfcc w/pitch - sometimes Kaldi adds to pitch
            ## one frame so give 2 frames of tolerance
            #fbank = fbank.concatenate(pitch, 2)
            fbank = self.concatenate_with_pitch(fbank.data, pitch.data)

        return fbank


    def get_dimension(self):
        n_features = 1
        n_features += self.melNbFilters
        n_features += self.with_pitch * 3 # Pitch is two dimensional
        return n_features

class ShennongBottleneck(ShennongFeatureExtraction):
    """Shennong Bottleneck

            |  e   |
            | c1   |
            | c2   |  coefficients
            | c3   |
        x = | c4   |  coefficients first derivatives
            | ...  |
            |pitch1|
            |pitch2|  Coefficients of pitch estimation (if pitch is asked)
            |pitch3|


    Parameters
    ----------
    sample_rate : int, optional
        Defaults to 16000 (i.e. 16kHz)
    augmentation : `pyannote.audio.augmentation.Augmentation`, optional
        Data augmentation.
    duration : float, optional
        Defaults to 0.025.
    step : float, optional
        Defaults to 0.010.
    e : bool, optional
        Energy. Defaults to True.
    with_pitch: bool, optional
        Compute Pitch Estimation (w/ same step and Duration as MFCC).
        Defaults to True.
    weights: str, optional.
        The name of the pretrained weights used to extract the features.
        Must be 'BabelMulti', 'FisherMono' or 'FisherTri'. Defaults to 
        'BabelMulti'.

    """

    def __init__(self, sample_rate=16000, augmentation=None,
                 duration=0.025, step=0.01,
                 weights='BabelMulti',
                 pitchFmin=20,
                 pitchFmax=500,
                 with_pitch=True):

        super().__init__(sample_rate=sample_rate, augmentation=augmentation,
                         duration=duration, step=step)

        self.with_pitch = with_pitch

        # pitch frequencies
        self.pitchFmin = pitchFmin
        self.pitchFmax = pitchFmax

        self.weights = weights

    def get_context_duration(self):
        return 0.

    def get_features(self, y, sample_rate):
        """Feature extraction

        Parameters
        ----------
        y : (n_samples, 1) numpy array
            Waveform
        sample_rate : int
            Sample rate

        Returns
        -------
        data : (n_frames, n_dimensions) numpy array
            Features
        """
        # create audio object for shennong
        audio = Audio(data=y, sample_rate=sample_rate)

        # create processor
        processor = BottleneckProcessor(weights=self.weights)

        # define parameters

        #processor.frame_length = self.duration
        #processor.frame_shift = self.step

        # extract features
        bottleneck = processor.process(audio)

        # Compute Pitch
        if self.with_pitch:
            # extract pitch
            pitch = self.get_pitch(audio, self.pitchFmin,
                                   self.pitchFmax)

            ## concatenate mfcc w/pitch - sometimes Kaldi adds to pitch
            ## one frame so give 2 frames of tolerance
            #bottleneck = bottleneck.concatenate(pitch, 2)
            bottleneck = self.concatenate_with_pitch(bottleneck.data,
                                                     pitch.data)

        return bottleneck


    def get_dimension(self):
        n_features = 0
        n_features += 80 # bottleneck have 80 dimensions
        n_features += self.with_pitch * 3 # Pitch is two dimensional
        return n_features

class ShennongMfcc(ShennongFeatureExtraction):
    """Shennong MFCC

    ::

            | e    |  energy
            | c1   |
            | c2   |  coefficients
            | c3   |
            | ...  |
            | Δe   |  energy first derivative
            | Δc1  |
        x = | Δc2  |  coefficients first derivatives
            | Δc3  |
            | ...  |
            | ΔΔe  |  energy second derivative
            | ΔΔc1 |
            | ΔΔc2 |  coefficients second derivatives
            | ΔΔc3 |
            | ...  |
            |pitch1|
            |pitch2|  Coefficients of pitch estimation
            |pitch3|


    Parameters
    ----------
    sample_rate : int, optional
        Defaults to 16000 (i.e. 16kHz)
    augmentation : `pyannote.audio.augmentation.Augmentation`, optional
        Data augmentation.
    duration : float, optional
        Defaults to 0.025.
    step : float, optional
        Defaults to 0.010.
    e : bool, optional
        Energy. Defaults to True.
    coefs : int, optional
        Number of coefficients. Defaults to 13.
    fmin : int, optional
        min frequency for pitch estimation. Defaults to 20.
    fmax : int, optional
        max frequency for pitch estimation. Defaults to 500.
    D : bool, optional
        Add first order derivatives. Defaults to True.
    DD : bool, optional
        Add second order derivatives. Defaults to True.
    with_pitch: bool, optional
        Compute Pitch Estimation (w/ same step and Duration as MFCC).
        Defaults to True.

    Notes
    -----
    Internal setup
        * fftWindow = Hanning
        * melMaxFreq = sampleFreq / 2 - 100
        * melMinFreq = 20
        * melNbFilters = 40


    """

    def __init__(self, sample_rate=16000, augmentation=None,
                 duration=0.025, step=0.01,
                 mfccWindowType='hanning',
                 mfccLowFreq=20,
                 mfccHighFreq=-100,
                 e=False, coefs=19, D=True, DD=True,
                 pitchFmin=20, pitchFmax=500, n_mels=40,
                 with_pitch=True, with_cmvn=True):

        super().__init__(sample_rate=sample_rate, augmentation=augmentation,
                         duration=duration, step=step)

        self.e = e
        self.coefs = coefs
        self.D = D
        self.DD = DD
        self.with_pitch = with_pitch
        self.with_cmvn = with_cmvn

        # pitch frequencies
        self.pitchFmin = pitchFmin
        self.pitchFmax = pitchFmax

        self.n_mels = n_mels
        self.mfccWindowType = mfccWindowType
        self.mfccLowFreq = mfccLowFreq
        self.mfccHighFreq = mfccHighFreq
    


    def get_context_duration(self):
        return 0.

    def concatenate_with_pitch(self, mfcc, pitch):
        """ When the pitch and the mfcc are not of same length, 
            pad the pitch equally at the begining and at the end
            to match the sizes.
        """
        # get size difference
        n_difference = mfcc.shape[0] - pitch.shape[0]

        if n_difference > 0:
            # add ceil(n_difference/2) frames at start or pitch array
            # and floor(n_difference/2) frames at end of pitch array
            ceil = int(np.ceil(n_difference/2))
            floor = int(np.floor(n_difference/2))

            pitch = np.insert(pitch, 0, np.zeros((ceil, 3)), axis=0)
            pitch = np.insert(pitch, pitch.shape[0], np.zeros((floor, 3)), axis=0)

        # concatenate pitch and mfcc which are now the same size
        stack = np.concatenate((mfcc, pitch), axis=1)

        return stack

    def get_features(self, y, sample_rate):
        """Feature extraction

        Parameters
        ----------
        y : (n_samples, 1) numpy array
            Waveform
        sample_rate : int
            Sample rate

        Returns
        -------
        data : (n_frames, n_dimensions) numpy array
            Features
        """
        # create audio object for shennong
        audio = Audio(data=y, sample_rate=sample_rate)

        # MFCC parameters
        processor = MfccProcessor(sample_rate=sample_rate)
        processor.window_type = 'hanning'
        processor.low_freq = 20
        processor.high_freq = -100 # defines it as (nyquist - 100)
        processor.use_energy = True
        processor.num_ceps = self.coefs
        processor.snip_edges= False # end with correct number of frames

        # MFCC extraction
        #audio = Audio(data=y, sample_rate=sample_rate)
        mfcc = processor.process(audio)
        # compute deltas
        if self.D:
            # define first or second order derivative
            if not self.DD:
                derivative_proc = DeltaPostProcessor(order=1)
            else:
                derivative_proc = DeltaPostProcessor(order=2)

            # process Mfccs
            mfcc = derivative_proc.process(mfcc)

        # Compute Pitch
        if self.with_pitch:
            # extract pitch
            pitch = self.get_pitch(audio, self.pitchFmin,
                                   self.pitchFmax)

            ## concatenate mfcc w/pitch - sometimes Kaldi adds to pitch
            ## one frame so give 2 frames of tolerance
            mfcc = self.concatenate_with_pitch(mfcc.data, pitch.data)
            #mfcc = mfcc.concatenate(pitch, 2)

            ## define pitch estimation parameters
            #processor = PitchProcessor(frame_shift=self.step,
            #                           frame_length=self.duration)
            #processor.sample_rate = sample_rate
            #processor.min_f0 = self.fmin
            #processor.max_f0 = self.fmax

            ## estimate pitch
            #pitch = processor.process(audio)


        # Compute CMVN
        if self.with_cmvn:
            # define cmvn
            postproc = CmvnPostProcessor(self.get_dimension(), stats=None)

            # accumulate stats
            stats = postproc.accumulate(mfcc)

            # process cmvn
            mfcc = postproc.process(mfcc)

        return mfcc

    def get_dimension(self):
        n_features = 0
        n_features += self.coefs
        n_features += self.coefs * self.D
        n_features += self.coefs * self.DD
        n_features += self.with_pitch * 3 # Pitch is two dimensional
        return n_features
