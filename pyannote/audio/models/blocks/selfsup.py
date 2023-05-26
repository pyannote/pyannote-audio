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
from transformers import AutoModel, Wav2Vec2FeatureExtractor, AutoConfig

class SelfSupModel(nn.Module):

    def __init__(self, model,layer, cache):
        super().__init__()
        self.model = model
        print("\nThe selected Self-Supervised Model from HuggingFace is "+ model+".\n")
        SelfSupModel.__name__ = model.rsplit('/', 1)[1] #Overwrite the class name to that of the selected model
        if cache is not None :
            print("Model and configuration file location is : "+str(cache))
            config = AutoConfig.from_pretrained(model, cache_dir = cache)
            config.cache_dir= cache
        else :
            config = AutoConfig.from_pretrained(model)
        
        config.output_hidden_states = True

        
        self.ssl_model = AutoModel.from_pretrained(model, config = config, cache_dir = cache) #Load the model
        self.ssl_model.eval()
        
        self.feat_size = config.hidden_size #Get the encoder feature size
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model, return_tensors="pt")

        if layer == None :
            print("\nLayer number not specified. Default to the first one (layer 0).\n")
            self.layer = 0
        else :        
            self.layer = layer
            print("\nSelected frozen layer is "+ str(layer) +". \n")

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        waveforms = torch.squeeze(waveforms,1) #waveforms : (batch, channel, sample) -> (batch,sample)
        if self.processor.do_normalize == True :
            waveforms = F.layer_norm(waveforms, waveforms.shape)
        
        with torch.no_grad():
            features = self.ssl_model(waveforms)  #Compute the features and extract last hidden layer weights
        
        outputs = features.hidden_states[self.layer + 1]

        return (outputs)
