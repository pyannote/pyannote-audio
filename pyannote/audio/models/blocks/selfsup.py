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
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models.wav2vec2 import wav2vec2_model
from collections import OrderedDict

class SelfSupModel(nn.Module):

    def __init__(self,checkpoint,name,layer,cfg):
        super().__init__()
        print("A checkpoint from a Self-Supervised Model is used for training")
        if torch.cuda.is_available():  
            ckpt = torch.load(checkpoint)
        else:
            ckpt = torch.load(checkpoint,map_location=torch.device('cpu'))
            #Check if the checkpoint is from an already finetuned Diarization model (containing SSL), or from a SSL pretrained model only
        if 'pyannote.audio' in ckpt: #1: Check if there is a Segmentation model attached onto or not
            name,config,ordered_dict = self.dict_finetune(ckpt)
                
        else: #Otherwise, load the dictionary of the SSL checkpoint
            print("The checkpoint is a pretrained SSL model to use for Segmentation.\nBuilding the SSL model.")
            name,config,ordered_dict = self.dict_pretrained(ckpt)
            
        self.model_name = name
        SelfSupModel.__name__ = self.model_name #Assign name of the class
        
        model = wav2vec2_model(**config) #Assign config to the model
        model.load_state_dict(ordered_dict) #Assign state dict to the model
        self.config = config #Assign the configuration
        self.ssl_model = model.eval()
        self.feat_size = config['encoder_embed_dim']
            
        if layer == None :
            print("\nLayer number not specified. Default to layer 1.\n")
            self.layer = 0
        else :        
            self.layer = layer
            print("\nSelected layer is "+ str(layer) +". \n")
            
    def dict_finetune(self, ckpt):
        #Need to reconstruct the dictionary
        #Get dict
        print("The checkpoint is used for finetuning. \nThe attached SSL model will be used for feature extraction !")
        dict_modules = list(ckpt['state_dict'].keys()) #Get the list of ssl modules
        ssl_modules = [key for key in dict_modules if 'selfsupervised' in key] #Extract only the SSL parts
        weights = [ckpt['state_dict'][key] for key in ssl_modules] #Get the weights corresponding to the modules
        modules_torchaudio = ['.'.join(key.split('.')[2:]) for key in ssl_modules] #Get a new list which contains only torchaudio keywords
        ordered_dict = OrderedDict((key,weight) for key,weight in zip(modules_torchaudio,weights)) #Recreate the state_dict
        config = ckpt['hyper_parameters']['selfsupervised']['cfg'] #Get config
        name = ckpt['hyper_parameters']['selfsupervised']['name'] #Get model name
            
        return(name,config,ordered_dict)
    
    def dict_pretrained(self, ckpt):
        ordered_dict = ckpt['state_dict'] #Get dict
        config = ckpt['config'] #Get config
        name = ckpt['model_name'] #Get model name
        
        return(ckpt['model_name'],ckpt['config'],ckpt['state_dict'])
    
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
        
        with torch.no_grad():
            feat,_ = self.ssl_model.extract_features(waveforms,None,self.layer+1)
                
        outputs = feat[self.layer]
        return (outputs)
