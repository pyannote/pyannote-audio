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

import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F

class NormalizeAudio(nn.Module):
    def __init__(self,eps:float=1e-6):
        super().__init__()
        self.eps = eps
        
    def forward(self,x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return (x - x.mean(dim=2, keepdims=True)) /\
                     (x.std(dim=2, keepdims=True, unbiased=False) + self.eps)

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

class MelBanks(nn.Module):
    def __init__(self, 
        sample_rate=16000, 
        n_fft=512, 
        win_length=400, 
        hop_length=160,
        f_min = 20, 
        f_max = 7600, 
        n_mels=80, 
        do_spec_aug=False,
        norm_signal=False,
        do_preemph=True,
        spec_norm='mn',
        freq_start_bin = 0,
        num_apply_spec_aug = 1,
        freq_mask_width = (0, 8), 
        time_mask_width = (0, 10),
    ):
        super(MelBanks, self).__init__()
        self.num_apply_spec_aug = num_apply_spec_aug
        self.torchfbank = torch.nn.Sequential(
            NormalizeAudio() if norm_signal else nn.Identity(),            
            PreEmphasis() if do_preemph else nn.Identity(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length, \
                                                 f_min = f_min, f_max = f_max, n_mels=n_mels, window_fn=torch.hamming_window),
            )
        self.spec_norm = spec_norm
        if spec_norm == 'mn':
            self.spec_norm = lambda x : x - torch.mean(x, dim=-1, keepdim=True)
        elif spec_norm == 'mvn':
            self.spec_norm = lambda x : (x - torch.mean(x, dim=-1, keepdim=True)) / (torch.std(x, dim=-1, keepdim=True)+1e-8)
        elif spec_norm == 'bn':
            self.spec_norm = nn.BatchNorm1d(n_mels)
        else:
            pass
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
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            # print(f"spec : {x.size()}")
            # x = x - torch.mean(x, dim=-1, keepdim=True)
            x = self.spec_norm(x)
            if self.training:
                for _ in range(self.num_apply_spec_aug):
                    x = self.specaug(x)
        return x.to(xdtype)