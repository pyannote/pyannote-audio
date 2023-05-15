# MIT License
#
# Copyright (c) 2020 CNRS
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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.models import wav2vec2_model, Wav2Vec2Model
from torchaudio.pipelines import Wav2Vec2Bundle

#All torchaudio Self-Sup. models can be found at https://pytorch.org/audio/main/pipelines.html
#ex : WAVLM_BASE, HUBERT_BASE, WAV2VEC2_BASE

class SelfSupModel(nn.Module):

    def __init__(self, model_name,layer_nb):
        super().__init__()
        self.model_name = model_name
        print("\nThe selected Self-Supervised Model is "+ model_name+".\n")
        SelfSupModel.__name__ = model_name #Overwrite the class name to that of the selected model       
        bundle = getattr(torchaudio.pipelines, model_name)
        self.feat_size = bundle._params['encoder_embed_dim'] #Get the encoder feature size
        torch.hub.set_dir("./models")
        self.ssl_model = bundle.get_model() #Load the model
        
        if layer_nb == None :
            print("\nLayer number not specified. Default to the first one (layer 0).\n")
            self.layer_nb = 0
        else :        
            self.layer_nb = layer_nb
            print("\nSelected frozen layer is "+ str(layer_nb) +". \n")
               
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        
        waveforms = torch.squeeze(waveforms,1) #waveforms : (batch, channel, sample) -> (batch,sample)
        with torch.no_grad():
            features, _ = self.ssl_model.extract_features(waveforms)  #Compute the features and extract last hidden layer weights
        outputs = features[self.layer_nb]
        
        return (outputs)
