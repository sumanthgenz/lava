import torch
import torchaudio
import torchvision
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.checkpoint import checkpoint_sequential, checkpoint
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

from torchaudio.models.wavernn import *

from aai.experimental.sgurram.lava.src.data import LAVAData, Kinetics700Data
from aai.experimental.sgurram.lava.src.metrics import nce_loss, centroid_loss, instance_loss
from aai.experimental.sgurram.lava.src.utils import get_src_conditional_mask, position_embed, position_embed_3d
from aai.experimental.sgurram.lava.src.references import lava_weights_path
# from aai.alexandria.layers.functional.positional_embedding import position_embed, position_embed_3d

torchaudio.set_audio_backend("sox_io")
warnings.filterwarnings("ignore")

class LanguageFeatureModel(torch.nn.Module):
    def __init__(self,
                dropout=0.1,
                model_dimension=512,):

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
        # return self.projection_mlp(t) 
        return checkpoint_sequential(self.projection_mlp, 4, t).to(dtype=torch.float)

class AudioFeatureModel(torch.nn.Module):
    def __init__(self, 
                dropout=0.1, 
                model_dimension=512,
                mel_freq=128,
                time_steps=2048,
                melresnet_dim=508,
                seq_len=256):

        super(AudioFeatureModel, self).__init__()


        self.model_dimension = model_dimension
        self.mel_freq = mel_freq
        self.time_steps = time_steps
        self.melresnet_dim = melresnet_dim
        self.seq_len = seq_len
        self.drop = dropout

        # Mel ResNet

        self.melresnet = MelResNet()

        # 508 -> 256 -> 64
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.melresnet_dim,  self.seq_len),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear( self.seq_len,  self.seq_len // 4),
        )

        # 128 -> 256 -> 512
        self.freq_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.mel_freq, self.model_dimension // 2),
            torch.nn.BatchNorm1d(self.seq_len),
            torch.nn.Linear(self.model_dimension // 2, self.model_dimension),
        )

    def forward(self, a):
       
        # Mel ResNet
        # Input [N x C x T], N = bsz, C = 128, T = 2048

        # [(N*4) x C x T/4]
        a = a.reshape((4*a.shape[0], a.shape[1], -1))

        # [(N*4) x C x T_1], T_1 = 508
        a = self.melresnet(a)

        # [(N*4) x C x T_2], T_2 = 64
        # a = self.time_mlp(a)
        a = checkpoint_sequential(self.time_mlp, 2, a)

        # [N x T_3 x C], T_3 = 128
        a = a.reshape(-1, a.shape[1], self.seq_len).permute(0, 2, 1)

        # [N x T_3 x F], F = 512
        # audio_encoded = self.freq_mlp(a)
        audio_encoded = checkpoint_sequential(self.freq_mlp, 2, a)

        return audio_encoded

class VideoFeatureModel(torch.nn.Module):
    def __init__(self, 
                dropout=0.1, 
                model_dimension=512,
                seq_len=16,
                target_len=256,
                feature_map_spatial_shape=(4,4),
                position_emb_len=48,
                resnet='50'):

        super(VideoFeatureModel, self).__init__()

        self.model_dimension = model_dimension
        self.temporal_seq_len = seq_len
        self.target_len = target_len
        self.position_emb_len = position_emb_len

        self.drop = dropout

        if resnet == '50':
            self.resnet_model = torchvision.models.resnet50(pretrained=False)
            self.resnet_dimension = 2048
        else:
            self.resnet_model = torchvision.models.resnet18(pretrained=False)
            self.resnet_dimension = 512


        self.feature_model = torch.nn.Sequential(
            *(list(self.resnet_model.children())[:-2]))

        # target_len 
        self.spatial_temporal_seq_len = feature_map_spatial_shape[0] * feature_map_spatial_shape[1] * self.temporal_seq_len # flattening spatial dim

        # 560 -> 512 -> 512
        self.feature_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.resnet_dimension + self.position_emb_len, self.resnet_dimension // 2),
            torch.nn.Dropout(p=self.drop),
            torch.nn.BatchNorm1d(self.spatial_temporal_seq_len),
            torch.nn.ReLU(),
            torch.nn.Linear(self.resnet_dimension // 2, self.model_dimension),
        )

    def forward(self, v):

        # N = bsz, S = 16, H = 128, W = 128,  C = 3, 
        # H' = 4, W' = 4, T = 256, D = 512

        #input (N, T, H, W, C) --> (N, T_1, D) which will be used as an input to the transformer

        # (N * S , C, H,  W]
        video_frames = v.reshape(v.shape[0] * v.shape[1], v.shape[2], v.shape[3], v.shape[4]).permute(0, 3, 1, 2)

        # (N * S, H_1, W_1, F]
        frames_encoded = self.feature_model(video_frames.contiguous()).permute(0, 2, 3, 1)

        # (N, S,  H_1, W_1, F]
        frames_encoded = frames_encoded.reshape(v.shape[0], 
                                                v.shape[1], 
                                                frames_encoded.shape[1], 
                                                frames_encoded.shape[2], 
                                                frames_encoded.shape[3])

        # (N, S, H_1, W_1, (F + P)]
        frames_pos_encoded = position_embed_3d(frames_encoded) # concat = True

        # (N, S* H_1 * W_1 = T, (F+P)]
        frames_mean = frames_pos_encoded.reshape(v.shape[0], 
                                                 frames_pos_encoded.shape[1] * frames_encoded.shape[2] * frames_encoded.shape[3],
                                                 frames_pos_encoded.shape[4])

        # (N, T, F+P)
        # frames_features = self.feature_mlp(frames_mean)
        frames_features = checkpoint_sequential(self.feature_mlp, 4, frames_mean)

        # (N, T, F)
        return frames_features


class LAVA(torch.nn.Module):

    def __init__(self, 
                dropout=0.1,
                model_dimension=1024, 
                feat_dimension=512,
                seqlen=256,
                batch_size=12, 
                learning_rate=3e-4,
                num_heads=8, 
                num_layers=8,):

        super(LAVA, self).__init__()

        self._model_dimension = model_dimension
        self._feature_dimension = feat_dimension
        self._seqlen = seqlen
        self._batch_size = batch_size
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._dropout = dropout
        self._learning_rate = learning_rate
        self._cos_sim = torch.nn.CosineSimilarity()

        self._audio_feature_model = AudioFeatureModel(
                                dropout=self._dropout,
                                model_dimension=self._feature_dimension)

        self._video_feature_model = VideoFeatureModel(
                                dropout=self._dropout,
                                model_dimension=self._feature_dimension)

        self._text_feature_model = LanguageFeatureModel(
                                dropout=self._dropout,
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

    def encode_audio(self, x):
        x = self._audio_feature_model(x)
        x = self._audio_input_projection(x.reshape(-1, self._feature_dimension)).reshape(
                x.shape[0], x.shape[1], self._model_dimension)

        x = self._audio_encoder(src=x).transpose(0, 1)
        x = self._audio_representation_mlp(x.reshape(
            -1, self._model_dimension)).reshape(*x.shape).transpose(0,1).mean(dim=1)
        return x

    def encode_video(self, x):
        x = self._video_feature_model(x)
        x = self._video_input_projection(x.reshape(-1, self._feature_dimension)).reshape(
                x.shape[0], x.shape[1], self._model_dimension)

        x = self._video_encoder(src=x).transpose(0, 1)
        x = self._video_representation_mlp(x.reshape(
            -1, self._model_dimension)).reshape(*x.shape).transpose(0,1).mean(dim=1)
        return x
    
    def encode_text(self, x):
        x = self._text_feature_model(x.squeeze())
        x = self._text_input_projection(x)
        x = self._text_representation_mlp(x)
        return x

    def forward(self, a, v, t):
        a = self.encode_audio(a)
        v = self.encode_video(v)
        t = self.encode_text(t)

        a = torch.nn.functional.normalize(a, p=2, dim=-1)
        v = torch.nn.functional.normalize(v, p=2, dim=-1)
        t = torch.nn.functional.normalize(t, p=2, dim=-1)

        return a, v, t
    
    def loss(self, a, v, t):
        av_loss = 0.5*(nce_loss(a, v) + nce_loss(v, a))
        at_loss = 0.5*(nce_loss(a, t) + nce_loss(t, a))
        vt_loss = 0.5*(nce_loss(v, t) + nce_loss(t, v))

        avt_loss = centroid_loss(a, v, t)

        av_cos_sim = self._cos_sim(a, v).mean()
        at_cos_sim = self._cos_sim(a, t).mean()
        vt_cos_sim = self._cos_sim(v, t).mean()

        a_cos_sim = torch.triu(torch.mm(a, a.t()), diagonal=-1).mean()
        v_cos_sim = torch.triu(torch.mm(v, v.t()), diagonal=-1).mean()
        t_cos_sim = torch.triu(torch.mm(v, t.t()), diagonal=-1).mean()

        total_loss = av_loss + at_loss + vt_loss + avt_loss

        metrics = {
            'loss_av': av_loss.item(),
            'loss_at': at_loss.item(),
            'loss_vt': vt_loss.item(),
            'loss_avt': avt_loss.item(),
            'loss': total_loss,
            'cos_sim_a': a_cos_sim.item(),
            'cos_sim_v': v_cos_sim.item(),
            'cos_sim_t': t_cos_sim.item(),
            'cos_sim_av': av_cos_sim.item(),
            'cos_sim_at': at_cos_sim.item(),
            'cos_sim_vt': vt_cos_sim.item(),
        }
        return metrics

class LinearClassifierAVT(torch.nn.Module):
    def __init__(self,
                data=Kinetics700Data,
                num_classes=700,
                feature_dimension=512,
                model_dimension=1024,
                num_modalities=3,
                batch_size=32,
                learning_rate=1e-3,
                model_path=None):

        super(LinearClassifierAVT, self).__init__()

        self.data = data
        self.num_classes = num_classes
        self.feature_dimension = feature_dimension
        self.model_dimension = model_dimension
        self.num_modalities = num_modalities
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_path = model_path if model_path else lava_weights_path

        self.model = LAVA()
        # self.model.load_state_dict(torch.load(self.model_path), strict=False)
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
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x

    def encode_video(self, x):
        x = self.v_feature_model(x)
        x = self.v_projection(x.reshape(-1, self.feature_dimension)).reshape(
                x.shape[0], x.shape[1], self.model_dimension)

        x = self.v_encoder(src=x).mean(dim=1)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x
    
    def encode_text(self, x):
        x = self.t_feature_model(x.squeeze())
        x = self.t_projection(x)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x
    
    def forward(self,  a, v, t):
        with torch.no_grad():
            a = self.encode_audio(a)
            v = self.encode_video(v)
            t = self.encode_text(t)
            representation = torch.cat((a,v,t), dim=-1)

        pred = self.fc1(representation)
        return pred


class SupervisedVideoClassifier(torch.nn.Module):
    def __init__(self,
                data=Kinetics700Data,
                num_classes=700,
                feature_dimension=512,
                model_dimension=512,
                num_modalities=3,
                batch_size=32,
                learning_rate=1e-3):

        super(SupervisedVideoClassifier, self).__init__()

        self.data = data
        self.num_classes = num_classes
        self.feature_dimension = feature_dimension
        self.model_dimension = model_dimension
        self.num_modalities = num_modalities
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model = LAVA()

        self.v_feature_model = self.model._video_feature_model
        self.v_projection = self.model._video_input_projection
        self.v_encoder = self.model._video_encoder

        self.fc1 = torch.nn.Linear(self.model_dimension, self.num_classes)

    def encode_video(self, x):
        x = self.v_feature_model(x)
        x = self.v_projection(x.reshape(-1, self.feature_dimension)).reshape(
                x.shape[0], x.shape[1], self.model_dimension)

        x = self.v_encoder(src=x).mean(dim=1)
        return x   
    
    def forward(self,  a, v, t):
        v_encoded = self.encode_video(v)
        pred = self.fc1(v_encoded)
        return pred
