# Part of this code was taken from https://github.com/cvqluu/TDNN

# Please give proper credit to the authors if you are using TDNN based or X-Vector based
# models by citing their papers:

# Peddinti, Vijayaditya, Daniel Povey and Sanjeev Khudanpur.
# "A time delay neural network architecture for efficient modeling of long temporal contexts."
# INTERSPEECH (2015).
# https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

# Snyder, David, Daniel Garcia-Romero, Gregory Sell, Daniel Povey and Sanjeev Khudanpur.
# "X-Vectors: Robust DNN Embeddings for Speaker Recognition."
# 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2018): 5329-5333.
# https://www.danielpovey.com/files/2018_icassp_xvectors.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def stats_pool(x: torch.Tensor):
    """Calculate mean and standard deviation of the input frames and concatenate them

    Parameters
    ----------
    x : (batch_size, n_frames, out_channels)
        Batch of frames

    Returns
    -------
    mean_std : (batch_size, 2 * out_channels)
    """
    mean, std = torch.mean(x, dim=1), torch.std(x, dim=1)
    return torch.cat((mean, std), dim=1)


class TDNNLayer(nn.Module):
    """
    TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

    Affine transformation not applied globally to all frames but smaller windows with local context

    batch_norm: True to include batch normalisation after the non linearity

    Context size and dilation determine the frames selected
    (although context size is not really defined in the traditional sense)
    For example:
        context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
        context size 3 and dilation 2 is equivalent to [-2, 0, 2]
        context size 1 and dilation 1 is equivalent to [0]
    """
    
    def __init__(self,
                 input_dim: int = 23,
                 output_dim: int = 512,
                 context_size: int = 5,
                 stride: int = 1,
                 dilation: int = 1,
                 batch_norm: bool = True,
                 dropout_p: float = 0.0):
        super(TDNNLayer, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
      
        self.kernel = nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x: torch.Tensor):
        """Calculate TDNN layer activations

        Parameters
        ----------
        x : (batch_size, n_frames, out_channels)
            Batch of frames

        Returns
        -------
        new_frames : (batch_size, new_n_frames, new_out_channels)
        """
        _, _, d = x.shape
        assert (d == self.input_dim), f'Input dimension was wrong. Expected ({self.input_dim}), got ({d})'
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(x, (self.context_size, self.input_dim),
                     stride=(1, self.input_dim),
                     dilation=(self.dilation, 1))

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1, 2)
        x = self.kernel(x)
        x = self.nonlinearity(x)
        
        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)

        return x


class XVectorNet(nn.Module):
    """
    X-Vector neural network architecture as defined by https://www.danielpovey.com/files/2018_icassp_xvectors.pdf

    Parameters
    ----------
    input_dim : int, default 24
        dimension of the input frames
    embedding_dim : int, default 512
        dimension of latent embeddings
    """

    @property
    def dimension(self):
        return self.embedding_dim

    def __init__(self, input_dim: int = 24, embedding_dim: int = 512):
        super(XVectorNet, self).__init__()
        frame1 = TDNNLayer(input_dim=input_dim, output_dim=512, context_size=5, dilation=1)
        frame2 = TDNNLayer(input_dim=512, output_dim=512, context_size=3, dilation=2)
        frame3 = TDNNLayer(input_dim=512, output_dim=512, context_size=3, dilation=3)
        frame4 = TDNNLayer(input_dim=512, output_dim=512, context_size=1, dilation=1)
        frame5 = TDNNLayer(input_dim=512, output_dim=1500, context_size=1, dilation=1)
        self.tdnn = nn.Sequential(frame1, frame2, frame3, frame4, frame5)
        self.segment6 = nn.Linear(3000, embedding_dim)
        self.segment7 = nn.Linear(embedding_dim, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, return_intermediate: Optional[str] = None):
        """Calculate X-Vector network activations.
           Return the requested intermediate layer without computing unnecessary activations.

        Parameters
        ----------
        x : (batch_size, n_frames, out_channels)
            Batch of frames
        return_intermediate : 'stats_pool' | 'segment6' | 'segment7' | None
            If specified, return the activation of this specific layer.
            segment6 and segment7 activations are returned before the application of non linearity.

        Returns
        -------
        activations :
            (batch_size, 3000)               if return_intermediate == 'stats_pool'
            (batch_size, embedding_dim)      if return_intermediate == 'segment6' | 'segment7' | None
        """

        x = stats_pool(self.tdnn(x))

        if return_intermediate == 'stats_pool':
            return x

        x = self.segment6(x)

        if return_intermediate == 'segment6':
            return x

        x = self.segment7(F.relu(x))

        if return_intermediate == 'segment7':
            return x

        return F.relu(x)
