import torch
import torchaudio
import torchvision
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pytorch_lightning as pl

import numpy as np
from tqdm import tqdm
from collections import Counter

import warnings
import glob
import pickle
import copy
import os
import sys

from data import LAVAData, Kinetics700Data
from metrics import nce_loss, centroid_loss
from utils import get_src_conditional_mask, position_embed, position_embed_3d
# from aai.alexandria.layers.functional.positional_embedding import position_embed, position_embed_3d

torchaudio.set_audio_backend("sox_io")
warnings.filterwarnings("ignore")

class LanguageFeatureModel(torch.nn.Module):
    def __init__(self,
                model_dimension=512,
                dropout=0.1):

        super(LanguageFeatureModel, self).__init__()

        self.model_dimension = model_dimension
        self.drop = dropout

        self.projection_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.model_dimension, self.model_dimension),
            torch.nn.Dropout(p=self.drop),
            torch.nn.BatchNorm1d(self.model_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.model_dimension, self.model_dimension),
        )

    def forward(self, t):
        """ encode text with projection_mlp and return """
        return self.projection_mlp(t) 

class AudioFeatureModel(torch.nn.Module):
    def __init__(self, 
                dropout=0.1, 
                model_dimension=512):

        super(AudioFeatureModel, self).__init__()

        self.mel_freq = 128
        self.model_dimension = model_dimension
        self.time_stpes = 300

        self.conv1 = torch.nn.Conv1d(
                    in_channels=self.mel_freq, 
                    out_channels=self.model_dimension, 
                    kernel_size=2, 
                    stride=2,
        )

        self.conv2 = torch.nn.Conv1d(
                    in_channels=self.model_dimension, 
                    out_channels=self.model_dimension, 
                    kernel_size=2,
                    stride=2,
        )

        self.pool1 = torch.nn.MaxPool1d(
                kernel_size=2,
                stride=2,
        )

        self.drop = torch.nn.Dropout(p=0.1)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.bn = torch.nn.BatchNorm1d(num_features=self.model_dimension)
        self.ln = torch.nn.LayerNorm(normalized_shape=(self.model_dimension, self.time_stpes))

        self.audio_conv = nn.Sequential(
                self.conv1,
                self.bn,
                self.relu,
                self.conv2,
                self.pool1,
                self.bn,
                self.relu,
        )

    def forward(self, a):
        #Input [N * C * T]

        audio_encoded = self.audio_conv(a)
        audio_encoded = audio_encoded.permute(0,2,1)
        audio_encoded = position_embed(audio_encoded)

        #Output [N * T * D]
        return audio_encoded

#Contains implemenation from https://github.com/CannyLab/aai/blob/e51bc4f0926530c39f289a948e0a1daebed3475a/aai/research/gptcaptions/models/encoders/predictive_byol.py#L21
class VideoFeatureModel(torch.nn.Module):
    def __init__(self, 
                dropout=0.1, 
                model_dimension=512,
                seq_len=16,
                target_len=256,
                position_emb_len=48):

        super(VideoFeatureModel, self).__init__()

        self.model_dimension = model_dimension
        self.seq_len = seq_len
        self.target_len = target_len
        self.position_emb_len = 48

        self.drop = dropout

        self.resnet_model = torchvision.models.resnet18(pretrained=True)

        self.feature_model = torch.nn.Sequential(
            *(list(self.resnet_model.children())[:-2]))

        # 560 -> 512 -> 512
        self.feature_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.model_dimension + self.position_emb_len, self.model_dimension),
            torch.nn.Dropout(p=self.drop),
            torch.nn.BatchNorm1d(self.target_len),
            torch.nn.ReLU(),
            torch.nn.Linear(self.model_dimension, self.model_dimension),
        )

    def forward(self, v):

        # N = bsz, S = 16, H = 128, W = 128,  C = 3, 
        # H' = 4, W' = 4, T = 256, D = 512

        # Input [N x S x H x W x C]

        # [(N * S) x C x H x W]
        video_frames = v.reshape(-1, *v.shape[2:]).permute(0, 3, 1, 2)

        # [(N * S) x H' x W' x D]
        frames_encoded = self.feature_model(video_frames.contiguous()).permute(0, 2, 3, 1)

        # [N x S x H' x W' x D]
        frames_encoded = frames_encoded.reshape(*v.shape[:2], *frames_encoded.shape[1:])

        # [N x S x H' x W' x (D + P)]
        frames_pos_encoded = position_embed_3d(frames_encoded)

        # [N x T x (D+P)]
        frames_mean = frames_pos_encoded.reshape(v.shape[0], -1, frames_pos_encoded.shape[-1])

        # [N x T x D]
        frames_features = self.feature_mlp(frames_mean)

        return frames_features

class LAVA(torch.nn.Module):

    def __init__(self, 
                dropout=0.1,
                model_dimension=512, 
                feat_dimension=512,
                seqlen=256,
                batch_size=64, 
                learning_rate=1e-3,
                num_heads=8, 
                num_layers=2,):

        super(LAVA, self).__init__()

        self._model_dimension = model_dimension
        self._feature_dimension = feat_dimension
        self._seqlen = seqlen
        self._batch_size = batch_size
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._dropout=dropout


        self._audio_feature_model = AudioFeatureModel(
                                dropout=0.1,
                                model_dimension=self._feature_dimension)

        self._video_feature_model = VideoFeatureModel(
                                dropout=0.1,
                                model_dimension=self._feature_dimension)

        self._text_feature_model = LanguageFeatureModel(
                                dropout=0.1,
                                model_dimension=self._feature_dimension)

        self._audio_token = torch.randn(self._batch_size, 1, self._feature_dimension)

        self._video_token = torch.randn(self._batch_size, 1, self._feature_dimension)

        self._text_token = torch.randn(self._batch_size, 1, self._feature_dimension)
        
        self._audio_input_projection = torch.nn.Sequential(
            torch.nn.Linear(self._feature_dimension, self._model_dimension),
            torch.nn.BatchNorm1d(self._model_dimension),
            torch.nn.ReLU(),
        )

        self._video_input_projection = torch.nn.Sequential(
            torch.nn.Linear(self._feature_dimension, self._model_dimension),
            torch.nn.BatchNorm1d(self._model_dimension),
            torch.nn.ReLU(),
        )

        self._text_input_projection = torch.nn.Sequential(
            torch.nn.Linear(self._feature_dimension, self._model_dimension),
            torch.nn.BatchNorm1d(self._model_dimension),
            torch.nn.ReLU(),
        )

        self._audio_encoder_layer = torch.nn.modules.TransformerEncoderLayer(d_model=self._model_dimension,
                                                                 nhead=self._num_heads,
                                                                 dim_feedforward=self._model_dimension,
                                                                 dropout=self._dropout,
                                                                 activation='relu')

        self._video_encoder_layer = torch.nn.modules.TransformerEncoderLayer(d_model=self._model_dimension,
                                                            nhead=self._num_heads,
                                                            dim_feedforward=self._model_dimension,
                                                            dropout=self._dropout,
                                                            activation='relu')
        
        self._audio_encoder = torch.nn.modules.TransformerEncoder(encoder_layer=self._audio_encoder_layer,
                                                                    num_layers=self._num_layers)
        
        self._video_encoder = torch.nn.modules.TransformerEncoder(encoder_layer=self._video_encoder_layer,
                                                                    num_layers=self._num_layers)        


        self._audio_representation_mlp = torch.nn.Sequential(
            torch.nn.Linear(self._model_dimension, self._model_dimension),
            torch.nn.BatchNorm1d(self._model_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self._model_dimension, self._model_dimension),
        )
        
        self._video_representation_mlp = torch.nn.Sequential(
            torch.nn.Linear(self._model_dimension, self._model_dimension),
            torch.nn.BatchNorm1d(self._model_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self._model_dimension, self._model_dimension),
        )

        self._text_representation_mlp = torch.nn.Sequential(
            torch.nn.Linear(self._model_dimension, self._model_dimension),
            torch.nn.BatchNorm1d(self._model_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self._model_dimension, self._model_dimension),
        )

        self._learning_rate = learning_rate

    def get_temporal_modality_views(self, audio, video):
        a1, a2 = torch.split(audio, split_size_or_sections=audio.shape[1]//2, dim=1) 
        v1, v2 = torch.split(video, split_size_or_sections=video.shape[1]//2, dim=1)

        a1, a2 = torch.cat((self._audio_token.to(a1.device), a1), dim=1), torch.cat((self._audio_token.to(a2.device), a2), dim=1)
        v1, v2 = torch.cat((self._video_token.to(v1.device), v1), dim=1), torch.cat((self._video_token.to(v2.device), v2), dim=1)

        view1 = torch.cat((a1, v2), dim=1)
        view2 = torch.cat((v1, a2), dim=1)
        return view1, view2


    def get_paramaters(self,):
        params = []

        params += list(self._audio_feature_model.parameters())
        params += list(self._video_feature_model.parameters())
        params += list(self._text_feature_model.parameters())

        params += list(self._audio_input_projection.parameters())
        params += list(self._video_input_projection.parameters())
        params += list(self._text_input_projection.parameters())

        params += list(self._audio_encoder.parameters())
        params += list(self._video_encoder.parameters())

        params += list(self._audio_representation_mlp.parameters())
        params += list(self._video_representation_mlp.parameters())
        params += list(self._text_representation_mlp.parameters())

        return params

    
    def _feature_project(self, x, mode):
        if mode=='audio':
            x = self._audio_feature_model(x)
            return self._audio_input_projection(x.reshape(-1, self._feature_dimension)).reshape(
                x.shape[0], x.shape[1], self._model_dimension)
        
        x = self._video_feature_model(x)
        return self._video_input_projection(x.reshape(-1, self._feature_dimension)).reshape(
                x.shape[0], x.shape[1], self._model_dimension)

    def _encode_sequence(self, seq, seqlen, mode):
        if mode=='audio':
            encoder = self._audio_encoder
            mlp = self._audio_representation_mlp
        else:
            encoder = self._video_encoder
            mlp = self._video_representation_mlp
        encoded = encoder(
            src=seq,
            mask=get_src_conditional_mask(seq.shape[0]).to(seq.device),
        ).transpose(0, 1)
  
        #transpose [T * N * D] -> [N * T * D] after mlp, mean pool to [N * D]
        encoded = mlp(encoded.reshape(
            -1, self._model_dimension)).reshape(*encoded.shape).transpose(0,1).mean(dim=1)

        return encoded
    
    def _encode_text(self, text):
        text = self._text_feature_model(text.squeeze())
        text = self._text_input_projection(text)
        text = self._text_representation_mlp(text)
        return text
    
    def forward(self, a, v, t):
        a = self._feature_project(a, mode='audio')
        a = self._encode_sequence(a,self._seqlen, mode='audio')
        a = torch.nn.functional.normalize(a, p=2, dim=-1)

        v = self._feature_project(v, mode='video')
        v = self._encode_sequence(v,self._seqlen, mode='video')
        v = torch.nn.functional.normalize(v, p=2, dim=-1)

        t = self._encode_text(t)
        t = torch.nn.functional.normalize(t, p=2, dim=-1)

        return a, v, t
    
    def loss(self, a, v, t):
        av_loss = 0.5*(nce_loss(a, v) + nce_loss(v, a))
        at_loss = 0.5*(nce_loss(a, t) + nce_loss(t, a))
        vt_loss = 0.5*(nce_loss(v, t) + nce_loss(t, v))
        avt_loss = centroid_loss(a, v, t)

        total_loss = av_loss + at_loss + vt_loss + avt_loss
                                                                             
        metrics = {
            'av_loss': av_loss,
            'at_loss': at_loss,
            'vt_loss': vt_loss,
            'avt_loss': avt_loss,
            'total_loss': total_loss,
        }
        return metrics

class LinearClassifierAVT(torch.nn.Module):
    def __init__(self,
                data=Kinetics700Data,
                num_classes=700,
                feature_dimension=512,
                model_dimension=128,
                num_modalities=3,
                batch_size=32,
                learning_rate=1e-3):

        super(LinearClassifierAVT, self).__init__()

        self.data = data
        self.num_classes = num_classes
        self.feature_dimension = feature_dimension
        self.model_dimension = model_dimension
        self.num_modalities = num_modalities
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model_path = "/home/sgurram/Desktop/video_lava/lava/364ti9hv/checkpoints/epoch=30.ckpt"
        self.model = LAVA()
        self.model.load_state_dict(torch.load(self.model_path), strict=False)
        self.model.eval()

        self.a_feature_model = self.model._audio_feature_model
        self.a_projection = self.model._audio_input_projection
        self.a_encoder = self.model._audio_encoder

        self.v_feature_model = self.model._video_feature_model
        self.v_projection = self.model._video_input_projection
        self.v_encoder = self.model._video_encoder

        self.t_feature_model = self.model._text_feature_model
        self.t_projection = self.model._text_input_projection

        self.fc1 = torch.nn.Linear(self.num_modalities * self.model_dimension, self.num_classes)
        # self.fc1 = torch.nn.Linear(self.model_dimension, self.num_classes)

    def encode_audio(self, x):
        x = self.a_feature_model(x)
        x = self.a_projection(x.reshape(-1, self.feature_dimension)).reshape(
                x.shape[0], x.shape[1], self.model_dimension)

        x = self.a_encoder(src=x).mean(dim=1)
        return x

    def encode_video(self, x):
        x = self.v_feature_model(x)
        x = self.v_projection(x.reshape(-1, self.feature_dimension)).reshape(
                x.shape[0], x.shape[1], self.model_dimension)

        x = self.v_encoder(src=x).mean(dim=1)
        return x
    
    def encode_text(self, x):
        x = self.t_feature_model(x.squeeze())
        x = self.t_projection(x)
        return x
    
    def forward(self,  a, v, t):
        with torch.no_grad():
            a = self.encode_audio(a)
            v = self.encode_video(v)
            t = self.encode_text(t)
            # print(any([p.requires_grad for p in [a, v, t]]))

        # representation = torch.stack((a,v,t)).squeeze().mean(dim=0)
        representation = torch.cat((a,v,t), dim=-1)
        pred = self.fc1(representation)
        return pred
