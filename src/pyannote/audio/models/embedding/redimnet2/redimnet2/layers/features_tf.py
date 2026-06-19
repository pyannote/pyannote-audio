# MIT License
# 
# Copyright (c) 2024 ID R&D, Inc.
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

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import windows

def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1 + hz / 700.)

def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10 ** (mel / 2595.0) - 1)

def get_filterbanks(low_freq: int = 20,
                    high_freq: int = 7600,
                    nfilt: int = 80,
                    nfft: int = 512,
                    samplerate: int = 16000):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param low_freq: lowest band edge of mel filters, default 0 Hz
    :param high_freq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """

    # compute points evenly spaced in mels
    lowmel = hz2mel(low_freq)
    highmel = hz2mel(high_freq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    
    lower_edge_mel = melpoints[:-2].reshape(1, -1)
    center_mel = melpoints[1:-1].reshape(1, -1)
    upper_edge_mel = melpoints[2:].reshape(1, -1)

    spectrogram_bins_mel = hz2mel(np.linspace(0, samplerate // 2, nfft))[1:].reshape(-1, 1)

    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
        center_mel - lower_edge_mel)
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
        upper_edge_mel - center_mel)

    mel_weights_matrix = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))
    return np.vstack([np.zeros((1, nfilt)), mel_weights_matrix])[:, :].astype('float32')

class SpectralFeaturesTF(nn.Module):
    def __init__(self,
                 frame_length: int = 400,
                 frame_step: int = 160,
                 fft_length: int = 512,
                 sample_rate: int = 16000,
                 window: str = 'hann',
                 normalize_spectrogram: bool = False,
                 normalize_signal: bool = False,
                 eps: float = 1e-8,
                 mode: str = 'melbanks',
                 low_freq: int = 20,
                 high_freq: int = 7600,
                 num_bins: int = 80,
                 log_mels: bool = True,
                 fft_mode: str = 'abs',
                 sqrt_real_imag: bool = False,
                 return_img: bool = False,
                 **kwargs):
        """
        Requirements
        ------------
        input shape must meet the conditions: mod((input.shape[0] - length), shift) == 0
        fft_length >= frame_length

        Parameters
        ------------
        :param frame_length: Length of each segment in # of samples
        :param frame_step: Shift between segments in # of samples
        :param fft_length: number of dft points, if None => fft_length == frame_length
        :param fft_mode: "abs" - amplitude spectrum; "real" - only real part, "imag" - only imag part,
        "complex" - concatenate real and imag part.
        :param kwargs: unuse

        Input
        -----
        input mut have shape: [n_batch, signal_length, 1]

        Returns
        -------
        A keras model that has output shape of
        (None, nfft / 2, n_time) (if type == "abs" || "real" || "imag") or
        (None, nfft / 2, n_frame, 2) (if type = "abs" & `img_dim_ordering() == 'tf').
        (None, nfft / 2, n_frame, 2) (if type = "complex" & `img_dim_ordering() == 'tf').

        number of time point of output spectrogram: n_time = (input.shape[0] - length) / shift + 1
        """
        super().__init__()

        assert mode in ['fft', 'melbanks', 'mfcc', 'complex']
        assert isinstance(frame_length, int) and isinstance(frame_step, int) and isinstance(fft_length, int)

        self.length = frame_length
        self.shift = frame_step
        self.sqrt_real_imag = sqrt_real_imag
        self.normalize_spectrogram = normalize_spectrogram
        self.normalize_signal = normalize_signal
        self.window = window
        self.eps = eps
        if fft_length is None:
            self.nfft = frame_length
        else:
            self.nfft = fft_length

        self.samplerate = sample_rate
        self.features = mode
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_bins = num_bins
        
        self.return_img = return_img

        if mode in ['melbanks', 'mfcc']:
            fft_mode = 'abs'
        self.fft_mode = fft_mode
        self.log_mels = log_mels
        self.build()

    def build(self):
        assert self.nfft >= self.length

        if self.window:
            if self.window == 'hamming':
                self.window = windows.hamming(self.length)
            elif self.window in ['hann', 'hanning']:
                self.window = np.array([0.5 - 0.5 * (np.cos((2 * np.pi * l) / (self.length - 1) )) \
                                        for l in range(self.length)])
            elif self.window == 'sqrt_hann':
                self.window = np.array([0.5 - 0.5 * (np.cos((2 * np.pi * l) / (self.length - 1) )) \
                                        for l in range(self.length)]) ** 0.5
            elif self.window == 'kaiser':
                self.window = windows.kaiser(self.length)
            else:
                self.window = np.ones(self.length)
        self.window = self.window.astype("float32")

        # real kernel
        real_kernel = np.asarray([np.cos(2 * np.pi * np.arange(0, self.nfft) * n / self.nfft)
                                        for n in range(self.nfft)]).astype("float32").T
        self.real_kernel = real_kernel[:self.length, :self.nfft // 2]
        if self.window is not None:
            self.real_kernel *= self.window[:, None]
        self.real_kernel = self.real_kernel[:,None,:]

        # imag kernel
        image_kernel = np.asarray([np.sin(2 * np.pi * np.arange(0, self.nfft) * n / self.nfft)
                                        for n in range(self.nfft)]).astype("float32").T
        self.image_kernel = image_kernel[:self.length, :self.nfft // 2]
        if self.window is not None:
            self.image_kernel *= self.window[:, None]
        self.image_kernel = self.image_kernel[:,None,:]

        self.register_buffer('real_kernel_pt', 
                             torch.from_numpy(self.real_kernel).permute(2,1,0).float())
        self.register_buffer('image_kernel_pt', 
                             torch.from_numpy(self.image_kernel).permute(2,1,0).float())

        if self.features in ['melbanks']:
            linear_to_mel_weight_matrix = get_filterbanks(
                                            nfilt=self.num_bins,
                                            nfft=self.nfft // 2,
                                            samplerate=self.samplerate,
                                            low_freq=self.low_freq,
                                            high_freq=self.high_freq)
            linear_to_mel_weight_matrix = linear_to_mel_weight_matrix[:,:,None]
            self.register_buffer('melbanks_pt', 
                                 torch.from_numpy(linear_to_mel_weight_matrix).permute(1,0,2).float())

    def forward(self, inputs):
        # inputs.size() : (bs,1,T)
        dtype = inputs.dtype
        inputs = inputs.float()
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
            
        if self.normalize_signal:
            inputs = (inputs - inputs.mean(dim=2, keepdims=True)) /\
                     (inputs.std(dim=2, keepdims=True, unbiased=False) + self.eps)
            
        real_part = F.conv1d(inputs, self.real_kernel_pt, stride=self.shift, padding=self.shift//2)
        imag_part = F.conv1d(inputs, self.image_kernel_pt, stride=self.shift, padding=self.shift//2)

        if self.features == 'complex':
            return [real_part, imag_part]
        
        fft = torch.square(real_part) + torch.square(imag_part)
        if self.sqrt_real_imag:
            fft = torch.sqrt(fft)
            
        feat = fft.clip(self.eps, 1/self.eps)
        
        if self.fft_mode == 'log':
            feat = torch.log(feat)

        if self.features in ['melbanks']:
            mel_spectrograms = F.conv1d(feat, self.melbanks_pt, stride=1, padding=0)
            mel_spectrograms = mel_spectrograms.clip(self.eps, 1/self.eps)
            if self.log_mels:
                feat = torch.log(mel_spectrograms)
            else:
                feat = mel_spectrograms

        if self.normalize_spectrogram:
            feat = (feat - feat.mean(dim=(1, 2), keepdims=True)) /\
                    (feat.std(dim=(1, 2), keepdims=True, unbiased=False) + self.eps)
        if self.return_img:
            feat = feat[:,None,:,:]
        return feat.to(dtype)

class LogSpec(nn.Module):
    def __init__(self,eps:float=1e-10):
        super().__init__()
        self.eps = eps
        
    def forward(self,x):
        return x.clip(self.eps,1e+8).log()
    
class NormalizeAudio(nn.Module):
    def __init__(self,eps:float=1e-10):
        super().__init__()
        self.eps = eps
        
    def forward(self,x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return ((x - x.mean(dim=2, keepdims=True)) /\
                     (x.std(dim=2, keepdims=True, unbiased=False) + self.eps)).squeeze(1)

class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), 'reflect')
        return F.conv1d(x, self.flipped_filter).squeeze(1)

class FbankAug(nn.Module):
    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10), freq_start_bin=0):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        self.freq_start_bin = freq_start_bin
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(self.freq_start_bin, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x

class TFMelBanks(nn.Module):
    def __init__(self, 
        sample_rate=16000, 
        n_fft=512, 
        win_length=400, 
        hop_length=160,
        f_min = 20, 
        f_max = 7600, 
        n_mels = 80, 
        do_spec_aug=False,
        norm_signal=False,
        do_preemph=True,
        freq_start_bin = 0,
        freq_mask_width = (0, 8), 
        time_mask_width = (0, 10),
        eps = 1e-8
    ):
        super(TFMelBanks, self).__init__()
        self.torchfbank = torch.nn.Sequential(
            NormalizeAudio(eps) if norm_signal else nn.Identity(),
            PreEmphasis() if do_preemph else nn.Identity(),
            
            SpectralFeaturesTF(
                frame_length = win_length,
                frame_step = hop_length,
                fft_length = n_fft,
                sample_rate = sample_rate,
                window = 'hamming',
                normalize_spectrogram = False,
                normalize_signal = False,
                eps = eps,
                mode = 'melbanks',
                low_freq = f_min,
                high_freq = f_max,
                num_bins = n_mels,
                log_mels = False,
                fft_mode = 'abs',
                sqrt_real_imag = False,
                return_img = False,
            )
        )
        self.eps = eps
        if do_spec_aug:
            self.specaug = FbankAug(
                freq_start_bin=freq_start_bin,
                freq_mask_width=freq_mask_width,
                time_mask_width=time_mask_width) # Spec augmentation
        else:
            self.specaug = nn.Identity()

    def forward(self, x):
        xdtype = x.dtype
        x = x.float()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfbank(x)+self.eps
                x = x.log()   
                x = x - torch.mean(x, dim=-1, keepdim=True)
                if self.training:
                    x = self.specaug(x)
        return x.to(xdtype)

class TFSpectrogram(nn.Module):
    def __init__(self, 
        sample_rate=16000, 
        n_fft=512, 
        win_length=400, 
        hop_length=160,
        f_min = 20, 
        f_max = 7600, 
        n_mels = 80, 
        window = 'hamming',
        normalize_spectrogram = False,
        normalize_signal = False,
        mode = 'fft',
        fft_mode = 'abs',
        pool_freqs = (2,1),
                 
        do_spec_aug=False,
        norm_signal=False,
        do_preemph=True,
                 
        freq_start_bin = 0,
        num_apply_spec_aug = 1,
        freq_mask_width = (0, 8), 
        time_mask_width = (0, 10),
        eps = 1e-8
    ):
        super(TFSpectrogram, self).__init__()
        self.num_apply_spec_aug = num_apply_spec_aug
        self.spectrogram = torch.nn.Sequential(
            NormalizeAudio() if norm_signal else nn.Identity(),            
            PreEmphasis() if do_preemph else nn.Identity(),            
            
            SpectralFeaturesTF(
                frame_length = win_length,
                frame_step = hop_length,
                fft_length = n_fft,
                sample_rate = sample_rate,
                window = window,
                eps = eps,
                mode = mode,
                low_freq = f_min,
                high_freq = f_max,
                num_bins = n_mels,

                normalize_spectrogram = False,
                normalize_signal = False,
                
                fft_mode = 'abs',
                log_mels = False,
                sqrt_real_imag = False,
                return_img = False,
            )
        )
        
        if pool_freqs is not None:
            self.pool_freq = nn.AvgPool2d(pool_freqs, stride=pool_freqs)
        else:
            self.pool_freq = nn.Identity()
            
        self.eps = eps
        if do_spec_aug:
            self.specaug = FbankAug(
                freq_start_bin=freq_start_bin,
                freq_mask_width=freq_mask_width,
                time_mask_width=time_mask_width) # Spec augmentation
        else:
            self.specaug = nn.Identity()

    def forward(self, x):
        xdtype = x.dtype
        x = x.float()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.spectrogram(x)+self.eps
                x = x.log()   
                x = x - torch.mean(x, dim=-1, keepdim=True)
                if self.training:
                    for _ in range(self.num_apply_spec_aug):
                        x = self.specaug(x)
                x = self.pool_freq(x.unsqueeze(1))
        return x.to(xdtype)