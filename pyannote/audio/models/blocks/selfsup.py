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
from torchaudio.models.wav2vec2.utils import import_fairseq_model

class SelfSupModel(nn.Module):

    def __init__(self, model,layer, cache,fairseq_ckpt):
        super().__init__()
        if fairseq_ckpt != None :
            import fairseq
            from fairseq import checkpoint_utils
            #Load the fairseq checkpoint
            models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([fairseq_ckpt])
            model = models[0]
            model.eval()
            model_name = model.__class__.__name__
            print("\nThe pre trained model "+model_name+" from fairseq is loaded.")
            SelfSupModel.__name__ = model_name
            
            #Convert the fairseq model to torchaudio to facilitate feature extraction from any layer.
            if model.__class__.__name__ is not "Wav2Vec2Model" :
                model.__class__.__name__ = "Wav2Vec2Model"
            self.feat_size = model.cfg.encoder_embed_dim
            model = import_fairseq_model(model).eval()
            self.ssl_model = model
            self.pretraining = True 

        else :
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
            config.num_hidden_layers = layer + 1

            self.ssl_model = AutoModel.from_pretrained(model, config = config, cache_dir = cache) #Load the model
            self.ssl_model.eval()

            self.feat_size = config.hidden_size #Get the encoder feature size
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model, return_tensors="pt")
            self.pretraining = False #If a pretrained model from fairseq is loaded instead, set to False
            
        if layer == None :
            print("\nLayer number not specified. Default to the first one (layer 0).\n")
            self.layer = 0
        else :        
            self.layer = layer
            print("\nSelected frozen layer is "+ str(layer) +". \n")
        
        #self.feat_layer_mean = [7,8,9]
    
    def mean_mat(self,features,feat_list):
        num_feat = len(feat_list)
        stack_feat = []
        for layer in feat_list:
            stack_feat.append(features[layer])
        stack_feat = torch.stack(stack_feat)
        feat_sum = torch.sum(stack_feat, dim=0)
        mean_feat = feat_sum / num_feat
        return(mean_feat)
    
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        waveforms = torch.squeeze(waveforms,1) #waveforms : (batch, channel, sample) -> (batch,sample)
        if self.pretraining == False :
            if self.processor.do_normalize == True :
                waveforms = F.layer_norm(waveforms, waveforms.shape)

            with torch.no_grad():
                features = self.ssl_model(waveforms)  #Compute the features and extract hidden layers

            outputs = features.hidden_states[self.layer + 1]
        else : 
            with torch.no_grad():
                feat,_ = self.ssl_model.extract_features(waveforms,None,self.layer+1)
                
            outputs = feat[self.layer]
                
        return (outputs)
