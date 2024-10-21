# MIT License

# Copyright (c) 2024 BUT Speech@FIT

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copied from https://github.com/BUTSpeechFIT/DiariZen/blob/e747106e753bb17799602b24d396f60b13da81b4/diarizen/models/module/conformer.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, maxlen=1000, embed_v=False):
        super(RelativePositionalEncoding, self).__init__()

        self.d_model = d_model
        self.maxlen = maxlen
        self.pe_k = nn.Embedding(2 * maxlen, d_model)
        if embed_v:
            self.pe_v = nn.Embedding(2 * maxlen, d_model)
        self.embed_v = embed_v

    def forward(self, pos_seq):
        pos_seq.clamp_(-self.maxlen, self.maxlen - 1)
        pos_seq = pos_seq + self.maxlen
        if self.embed_v:
            return self.pe_k(pos_seq), self.pe_v(pos_seq)
        else:
            return self.pe_k(pos_seq), None


class MultiHeadSelfAttention(nn.Module):
    """Multi head self-attention layer"""

    def __init__(self, n_units: int, h: int, dropout: float) -> None:
        super().__init__()
        self.linearQ = nn.Linear(n_units, n_units)
        self.linearK = nn.Linear(n_units, n_units)
        self.linearV = nn.Linear(n_units, n_units)
        self.linearO = nn.Linear(n_units, n_units)

        self.d_k = n_units // h
        self.h = h
        self.dropout = nn.Dropout(p=dropout)
        self.att = None  # attention for plot

    def __call__(self, x: torch.Tensor, batch_size: int, pos_k=None) -> torch.Tensor:
        # x: (BT, F)
        q = self.linearQ(x).reshape(batch_size, -1, self.h, self.d_k)
        k = self.linearK(x).reshape(batch_size, -1, self.h, self.d_k)
        v = self.linearV(x).reshape(batch_size, -1, self.h, self.d_k)

        q = q.transpose(1, 2)  # (batch, head, time, d_k)
        k = k.transpose(1, 2)  # (batch, head, time, d_k)
        v = v.transpose(1, 2)  # (batch, head, time, d_k)
        att_score = torch.matmul(q, k.transpose(-2, -1))

        if pos_k is not None:
            reshape_q = q.reshape(batch_size * self.h, -1, self.d_k).transpose(0, 1)
            att_score_pos = torch.matmul(reshape_q, pos_k.transpose(-2, -1))
            att_score_pos = att_score_pos.transpose(0, 1).reshape(
                batch_size, self.h, pos_k.size(0), pos_k.size(1)
            )
            scores = (att_score + att_score_pos) / np.sqrt(self.d_k)
        else:
            scores = att_score / np.sqrt(self.d_k)

        # scores: (B, h, T, T)
        self.att = F.softmax(scores, dim=3)
        p_att = self.dropout(self.att)
        x = torch.matmul(p_att, v)
        x = x.permute(0, 2, 1, 3).reshape(-1, self.h * self.d_k)
        return self.linearO(x)


class Swish(nn.Module):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * inputs.sigmoid()


class ConformerMHA(nn.Module):
    """
    Conformer MultiHeadedAttention(RelMHA) module with residule connection and dropout.
    """

    def __init__(
        self,
        in_size: int = 256,
        num_head: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln_norm = nn.LayerNorm(in_size)
        self.mha = MultiHeadSelfAttention(n_units=in_size, h=num_head, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_k=None) -> torch.Tensor:
        """
        x: B, T, N
        """
        bs, time, idim = x.shape
        x = x.reshape(-1, idim)
        res = x
        x = self.ln_norm(x)
        x = self.mha(x, bs, pos_k)
        x = self.dropout(x)
        x = res + x
        x = x.reshape(bs, time, -1)
        return x


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward layer
                    with scaled residule connection and dropout.
    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, in_size, ffn_hidden, dropout=0.1, swish=Swish()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.ln_norm = nn.LayerNorm(in_size)
        self.w_1 = nn.Linear(in_size, ffn_hidden)
        self.swish = swish
        self.dropout1 = nn.Dropout(dropout)
        self.w_2 = nn.Linear(ffn_hidden, in_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """Forward function."""
        res = x
        x = self.ln_norm(x)
        x = self.swish(self.w_1(x))
        x = self.dropout1(x)
        x = self.dropout2(self.w_2(x))

        return res + 0.5 * x


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model
                    with residule connection and dropout.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.

    """

    def __init__(
        self, channels, kernel_size=31, dropout_rate=0.1, swish=Swish(), bias=True
    ):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0
        self.ln_norm = nn.LayerNorm(channels)
        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.bn_norm = nn.BatchNorm1d(channels)
        self.swish = swish
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        # exchange the temporal dimension and the feature dimension
        res = x
        x = self.ln_norm(x)
        x = x.transpose(1, 2)  # B, N, T

        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = self.glu(x)  # (batch, channel, dim)

        x = self.depthwise_conv(x)
        x = self.swish(self.bn_norm(x))
        x = self.dropout(self.pointwise_conv2(x))

        return res + x.transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        in_size: int = 256,
        ffn_hidden: int = 1024,
        num_head: int = 2,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ffn1 = PositionwiseFeedForward(
            in_size=in_size, ffn_hidden=ffn_hidden, dropout=dropout
        )
        self.mha = ConformerMHA(in_size=in_size, num_head=num_head, dropout=dropout)
        self.conv = ConvolutionModule(channels=in_size, kernel_size=kernel_size)
        self.ffn2 = PositionwiseFeedForward(
            in_size=in_size, ffn_hidden=ffn_hidden, dropout=dropout
        )
        self.ln_norm = nn.LayerNorm(in_size)

    def forward(self, x: torch.Tensor, pos_k=None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
        """
        x = self.ffn1(x)
        x = self.mha(x, pos_k)
        x = self.conv(x)
        x = self.ffn2(x)

        return self.ln_norm(x)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        attention_in: int = 256,
        ffn_hidden: int = 1024,
        num_head: int = 4,
        num_layer: int = 4,
        kernel_size: int = 31,
        dropout: float = 0.1,
        use_posi: bool = False,
        output_activate_function="ReLU",
    ) -> None:
        super().__init__()

        if not use_posi:
            self.pos_emb = None
        else:
            self.pos_emb = RelativePositionalEncoding(attention_in // num_head)

        self.conformer_layer = nn.ModuleList(
            [
                ConformerBlock(
                    in_size=attention_in,
                    ffn_hidden=ffn_hidden,
                    num_head=num_head,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layer)
            ]
        )

        # Activation function layer
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            elif output_activate_function == "LeakyReLU":
                self.activate_function = nn.LeakyReLU()
            elif output_activate_function == "PReLU":
                self.activate_function = nn.PReLU()
            elif output_activate_function == "Sigmoid":
                self.activate_function = nn.Sigmoid()
            else:
                raise NotImplementedError(
                    f"Not implemented activation function {self.activate_function}"
                )
        self.output_activate_function = output_activate_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
        """
        if self.pos_emb is not None:
            x_len = x.shape[1]
            pos_seq = torch.arange(0, x_len).long().to(x.device)
            pos_seq = pos_seq[:, None] - pos_seq[None, :]
            pos_k, _ = self.pos_emb(pos_seq)
        else:
            pos_k = None

        for layer in self.conformer_layer:
            x = layer(x, pos_k)
        if self.output_activate_function:
            x = self.activate_function(x)
        return x
