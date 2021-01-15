import torch
import torchaudio
import torchvision
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet

import numpy as np
import pandas as pd 
import warnings
import glob
from tqdm import tqdm
import pickle
from collections import Counter
import copy
import os

from data import *
from metrics import *
from utils import *


class AudioFeatureModel(torch.nn.Module):
    def __init__(self, 
                dropout=0.1, 
                model_dimension=512):

        super(AudioFeatureModel, self).__init__()

        self.mel_freq = 128
        self.model_dimension = 512
        self.time_stpes = 300

        #audio convnet 
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

        self.pool1 = nn.MaxPool1d(
                kernel_size=2,
                stride=2,
        )

        self.drop = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.bn = nn.BatchNorm1d(num_features=self.model_dimension)
        self.ln = nn.LayerNorm(normalized_shape=(self.model_dimension, self.time_stpes))

        self.audio_conv = nn.Sequential(
                self.conv1,
                self.conv2,
                self.bn,
                self.relu,
                self.pool1,
                self.drop,
        )

    def forward(self, input_audio):
        #Input [N * C * T]

        x = self.audio_conv(input_audio)
        x = torch.einsum('ndt->ntd', [x])

        #Output [N * T * D]
        return x

#Contains implemenation from https://github.com/CannyLab/aai/blob/e51bc4f0926530c39f289a948e0a1daebed3475a/aai/research/gptcaptions/models/encoders/predictive_byol.py#L21
class VideoFeatureModel(torch.nn.Module):
    def __init__(self, 
                dropout=0.1, 
                model_dimension=512):

        super(VideoFeatureModel, self).__init__()

        self.model_dimension = model_dimension

        self.dropout = dropout

        self.resnet_model = torchvision.models.resnet18(pretrained=True)

        self.feature_model = torch.nn.Sequential(
            self.resnet_model.conv1,
            self.resnet_model.bn1,
            self.resnet_model.relu,
            self.resnet_model.maxpool,
            self.resnet_model.layer1,
            self.resnet_model.layer2,
            self.resnet_model.layer3,
            self.resnet_model.layer4,
        )

    def forward(self, v):
        #Input [N * T * H * W * C]

        # x = x.type(torch.FloatTensor)

        video_frames = v.reshape(v.shape[0]*v.shape[2], v.shape[1], v.shape[3], v.shape[3])
        frames_encoded = self.feature_model(video_frames.contiguous())
        frames_encoded = frames_encoded.reshape(v.shape[0], -1,
                                                *frames_encoded.shape[1:]).mean(dim=(3, 4))

        #Output [N * T * D]
        return frames_encoded

#Contains implementation from https://github.com/CannyLab/aai/blob/ddc76404bdfe15fb8218c31d9dc6859f3d5420db/aai/research/gptcaptions/models/encoders/predictive_byol.py
class BYOLEncoder(torch.nn.Module):

    def __init__(self, 
                dropout=0.1,
                model_dimension=128, 
                feat_dimension=512,
                seqlen=277,
                batch_size=5, 
                learning_rate=1e-3,
                num_heads=4, 
                num_layers=4, 
                ema_beta=0.95,
                cos_loss_weight=0.5,
                kldiv_loss_weight=0.45,
                random_loss_weight=0.05):

        super(BYOLEncoder, self).__init__()

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

        self._audio_token = torch.randn(self._batch_size, 1, self._feature_dimension)

        self._video_token = torch.randn(self._batch_size, 1, self._feature_dimension)


        self._frame_input_projection = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self._feature_dimension),
            torch.nn.Linear(self._feature_dimension, self._model_dimension),
            torch.nn.ReLU(),
        )

        self._encoder_layer = torch.nn.modules.TransformerEncoderLayer(d_model=self._model_dimension,
                                                                 nhead=self._num_heads,
                                                                 dim_feedforward=self._model_dimension,
                                                                 dropout=self._dropout,
                                                                 activation='relu')
        self._encoder = torch.nn.modules.TransformerEncoder(encoder_layer=self._encoder_layer,
                                                                    num_layers=self._num_layers)


        self._representation_mlp = torch.nn.Sequential(
            torch.nn.Linear(self._model_dimension, self._model_dimension),
            torch.nn.BatchNorm1d(self._model_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self._model_dimension, self._model_dimension),
        )
        self._byol_predictor = torch.nn.Sequential(
            torch.nn.Linear(self._model_dimension, self._model_dimension),
            torch.nn.BatchNorm1d(self._model_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self._model_dimension, self._model_dimension),
        )

        self._translation_mlp = torch.nn.Sequential(
            torch.nn.Linear(2*self._model_dimension, self._model_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self._model_dimension, self._model_dimension),
        )

        self._target_encoder = None
        self._target_networks_initialized = False
        self._ema_beta = ema_beta
        self._cosine_loss_weight = cosine_loss_weight
        self._kldiv_loss_weight = kldiv_loss_weight
        self._random_loss_weight = random_loss_weight
        self._learning_rate = learning_rate

    def get_temporal_modality_views(self, audio, video):
        a1, a2 = torch.split(audio, split_size_or_sections=audio.shape[1]//2, dim=1) 
        v1, v2 = torch.split(video, split_size_or_sections=video.shape[1]//2, dim=1)

        a1, a2 = torch.cat((self._audio_token, a1), dim=1), torch.cat((self._audio_token, a2), dim=1)
        v1, v2 = torch.cat((self._video_token, v1), dim=1), torch.cat((self._video_token, v2), dim=1)

        view1 = torch.cat((a1, v2), dim=1)
        view2 = torch.cat((v1, a2), dim=1)
        return view1, view2


    def get_paramaters(self,):
        params = []
        params += list(self._audio_feature_model.parameters())
        params += list(self._video_feature_model.parameters())
        params += list(self._frame_input_projection.parameters())
        params += list(self._encoder.parameters())
        if self._target_encoder is not None:
            params += list(self._target_encoder.parameters())
        params += list(self._representation_mlp.parameters())
        params += list(self._byol_predictor.parameters())
        params += list(self._translation_mlp.parameters())

        return params

    def reset_moving_average(self):
        del self._target_encoder
        self._target_encoder = None


    def _ema_copy_model(self, online_model, target_model):
        for current_params, target_params in zip(online_model.parameters(), target_model.parameters()):
            old_weight, new_weight = target_params.data, current_params.data
            target_params.data = old_weight * self._ema_beta + (1 - self._ema_beta) * new_weight


    def update_moving_average(self):
        if self._target_encoder is not None:
            self._ema_copy_model(self._encoder, self._target_encoder)


    def byol_encode(self, x, online=True):
        if online:
            x = self._encoder(x)
        else:
            if not self._target_networks_initialized:
                self._target_encoder = copy.deepcopy(self._encoder)
                self._target_networks_initialized = True
            x = self._target_encoder(x)
        x = self._representation_mlp(x)
        return x
    
    def _feature_project(self, x):
        return self._frame_input_projection(x.reshape(-1, self._feature_dimension)).reshape(
            x.shape[0], x.shape[1], self._model_dimension)

    def _encode_sequence(self, seq, seqlen, online=True):
        if online:
            encoded = self._encoder(
                src=seq,
                mask=get_src_conditional_mask(seq.shape[0]).to(seq.device),
            ).transpose(0, 1)
        else:
            if not self._target_networks_initialized:
                self._target_encoder = copy.deepcopy(self._encoder)
                self._target_networks_initialized = True

            #transpose [N * T * D] -> [T * N * D] as input to  batchnorm for representation_mlp
            encoded = self._target_encoder(
                src=seq,
                mask=get_src_conditional_mask(seq.shape[0]).to(seq.device),
            ).transpose(0, 1)

        #transpose [T * N * D] -> [N * T * D] after mlp
        encoded = self._representation_mlp(encoded.reshape(
            -1, self._model_dimension)).reshape(*encoded.shape).transpose(0,1)

        return encoded
    
    def forward(self, a, v):
        # a, v = x

        a = self._audio_feature_model(a)
        v = self._video_feature_model(v)

        #x, y are temporally-ordered cross-modal views of source video
        x, y = self.get_temporal_modality_views(a,v)

        x, y = self._feature_project(x), self._feature_project(y)
        x_online = self._encode_sequence(x,
                                        self._seqlen,
                                        online=True,)

        y_online = self._encode_sequence(y,
                                        self._seqlen,
                                        online=True,)

        # x_online, y_online = self._encoder(x), self._encoder(y)

        with torch.no_grad():
            x_target = self._encode_sequence(x,
                                    self._seqlen,
                                    online=False,)

            y_target = self._encode_sequence(y,
                                            self._seqlen,
                                            online=False,)


        x_online = self._byol_predictor(x_online.reshape(
            -1, self._model_dimension)).reshape(*x_online.shape)

        y_online = self._byol_predictor(y_online.reshape(
            -1, self._model_dimension)).reshape(*y_online.shape)

        x_online = torch.nn.functional.normalize(x_online, p=2, dim=-1)
        y_online = torch.nn.functional.normalize(y_online, p=2, dim=-1)

        x_target = torch.nn.functional.normalize(x_target, p=2, dim=-1) 
        y_target = torch.nn.functional.normalize(y_target, p=2, dim=-1)

        #byol-encoded views
        return x_online, y_online, x_target, y_target
    
    def loss(self, x_online, y_online, x_target, y_target):
        cos_loss = cosine_loss(x_online, y_target) + cosine_loss(x_target, y_online)
        kl_loss = kldiv_loss(x_online, y_target) + kl_div_loss(x_target, y_online)
        rand_loss = random_loss(x_online, y_target) + random_loss(x_target, y_online)

        total_loss = torch.zeros([]).cuda()
        total_loss += self._cosine_loss_weight * cos_loss 
        total_loss += self._kldiv_loss_weight * kl_loss 
        total_loss += self._random_loss_weight * rand_loss 

        metrics = {
            'cosine_loss': cos_loss,
            'kldiv_loss': kl_loss,
            'random_loss': rand_loss,
            'total_loss': total_loss
        }

        return metrics