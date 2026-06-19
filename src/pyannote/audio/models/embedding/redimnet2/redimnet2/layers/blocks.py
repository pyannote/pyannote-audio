import torch.nn as nn
from .layernorm import LayerNorm
from .resblocks import ResBasicBlock
from .convnext import ConvNeXtLikeBlock
from .attention import TransformerEncoderLayer

#------------------------------------------
#              Main blocks
#------------------------------------------

class ConvBlock2d(nn.Module):
    def __init__(self, c, f, block_type="convnext_like", Gdiv=1, kernel_sizes=None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [(3, 3)]
        if block_type == "convnext_like":
            self.conv_block = ConvNeXtLikeBlock(c, dim=2, kernel_sizes=kernel_sizes,
                                                Gdiv=Gdiv, padding='same', activation='gelu')
        elif block_type == "convnext_like_relu":
            self.conv_block = ConvNeXtLikeBlock(c, dim=2, kernel_sizes=kernel_sizes,
                                                Gdiv=Gdiv, padding='same', activation='relu')
        elif block_type == "basic_resnet":
            self.conv_block = ResBasicBlock(c, c, f, stride=1, se_channels=min(64,max(c,32)), Gdiv=Gdiv, use_fwSE=False)
        elif block_type == "basic_resnet_fwse":
            self.conv_block = ResBasicBlock(c, c, f, stride=1, se_channels=min(64,max(c,32)), Gdiv=Gdiv, use_fwSE=True)
        else:
            raise NotImplemented()

    def forward(self, x):
        return self.conv_block(x)

#------------------------------------------
#                1D block
#------------------------------------------

class PosEncConv(nn.Module):
    def __init__(self, C, ks, groups=None):
        super().__init__()
        assert ks % 2 == 1
        self.conv = nn.Conv1d(C,C,ks,
                              padding=ks//2,
                              groups=C if groups is None else groups)
        self.norm = LayerNorm(C, eps=1e-6, data_format="channels_first")
        
    def forward(self,x):        
        return x + self.norm(self.conv(x))

class TimeContextBlock1d(nn.Module):
    def __init__(self, 
        C, 
        hC,
        pos_ker_sz = 59,
        block_type = 'att',
        red_dim_conv = None,
        exp_dim_conv = None
    ):
        super().__init__()
        assert pos_ker_sz 
        
        self.red_dim_conv = nn.Sequential(
            nn.Conv1d(C,hC,1),
            LayerNorm(hC, eps=1e-6, data_format="channels_first")
        )
        if block_type == 'fc':
            self.tcm = nn.Sequential(
                nn.Conv1d(hC,hC*2,1),
                LayerNorm(hC*2, eps=1e-6, 
                          data_format="channels_first"),
                nn.GELU(),
                nn.Conv1d(hC*2,hC,1)
            )
        elif block_type == 'conv':
            # Just large kernel size conv like in convformer
            self.tcm = nn.Sequential(*[ConvNeXtLikeBlock(
                hC, dim=1, kernel_sizes=[7, 15, 31], Gdiv=1, padding='same'
            ) for i in range(4)])
        elif block_type == 'att':
            # Basic Transformer self-attention encoder block
            self.tcm = nn.Sequential(
                PosEncConv(hC, ks=pos_ker_sz, groups=hC),
                TransformerEncoderLayer(
                    n_state=hC, 
                    n_mlp=hC*2, 
                    n_head=4
                )
            )
        elif block_type == 'conv+att':
            # Basic Transformer self-attention encoder block
            self.tcm = nn.Sequential(
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[7], Gdiv=1, padding='same'),
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[19], Gdiv=1, padding='same'),
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[31], Gdiv=1, padding='same'),
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[59], Gdiv=1, padding='same'),
                TransformerEncoderLayer(
                    n_state=hC, 
                    n_mlp=hC, 
                    n_head=4
                )
            )
        else:
            raise NotImplemented()
            
        self.exp_dim_conv = nn.Conv1d(hC,C,1)
        
    def forward(self,x):
        skip = x
        x = self.red_dim_conv(x)
        x = self.tcm(x)
        x = self.exp_dim_conv(x)
        return skip + x