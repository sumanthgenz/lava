import torch
import torch.nn as nn
import torchaudio
import torchvision
import torchtext
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.checkpoint import checkpoint_sequential, checkpoint
from torchvision import transforms
import pytorch_lightning as pl

import numpy as np
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

import warnings
import glob
import pickle
import copy
import os
import sys

from aai.experimental.sgurram.lava.src.metrics import nce_loss, centroid_loss, instance_loss, compute_accuracy
from aai.experimental.sgurram.lava.src.references import lava_weights_path, sp_model_path, sp_vocab_size
from aai.experimental.sgurram.lava.src.utils import get_src_conditional_mask, position_embed, position_embed_3d, pad_batch, visualize_batch
from aai.experimental.sgurram.lava.src.data import*

class AudioEncoder(nn.Module):
    def __init__(self, 
                batch_size=8,
                model_dimension=1024,
                mel_freq=80, #80
                time_steps=512,
                melresnet_dim=508,
                patch_size=8, #8
                max_seqlen=64, #64
                num_heads=8, 
                num_layers=8,
                dropout=0.0,
                pool="token",
            ):

        super(AudioEncoder, self).__init__()

        self.batch_size = batch_size
        self.model_dimension = model_dimension
        self.mel_freq = mel_freq
        self.time_steps = time_steps
        self.melresnet_dim = melresnet_dim
        self.max_seqlen = time_steps // patch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.dropout = dropout
        self.pool =  pool
        
        self.scale = model_dimension ** -0.5

        self.feature_dimension = self.mel_freq * self.patch_size

        self.pos_embedding = nn.Parameter(self.scale * torch.rand(self.max_seqlen+1, self.model_dimension))

        self.feat_token = nn.Parameter(self.scale * torch.randn(1, self.model_dimension))

        self.proj = nn.Linear(self.feature_dimension, self.model_dimension)

        self.encoder_layer = nn.modules.TransformerEncoderLayer(d_model=self.model_dimension,
                                                                 nhead=self.num_heads,
                                                                 dim_feedforward=self.model_dimension,
                                                                 dropout=self.dropout,
                                                                 activation='gelu')
                                                                
        self.encoder = nn.modules.TransformerEncoder(encoder_layer=self.encoder_layer,
                                                                    num_layers=self.num_layers)

        self.fc = nn.Sequential(
            nn.LayerNorm(self.model_dimension),
            nn.Linear(self.model_dimension, self.model_dimension)
        )
            
    def forward(self, x):
       
        # Input [N x M x T]
        n, m, t = x.shape
        s, p = t//self.patch_size, self.patch_size
        f, d = self.feature_dimension, self.model_dimension

        feat_token = self.feat_token.repeat(x.shape[0], 1).unsqueeze(1)

        # , S = T/P
        x = x.permute(0, 2, 1).unfold(1, p, p) # [N x T x M] -> [N x S x P x M]
        x = x.reshape(*x.shape[:-2], -1) # [N x S x P x M] -> [N x S x F]

        x = self.proj(x) # [N x S x F] -> [N x S x D]
        
        # [(N x (S+1) x D]
        x = torch.cat((x, feat_token,), dim=1)

        x = x + self.pos_embedding.to(x.dtype)

        x = x.permute(1, 0, 2) # [N x S x D] -> [S x N x D]
        x = self.encoder(x)
        x = x.permute(1, 0, 2) # [S x N x D] -> [N x S x D]

        x = x[:, -1, :].squeeze() if self.pool == "token" else x.mean(dim=1).squeeze() # [N x D]

        x = self.fc(x)

        return x


class TextEncoder(nn.Module):
    def __init__(self,
                batch_size=8,
                model_dimension=1024,
                num_heads=8, 
                num_layers=8,
                vocab_size=48000,
                max_seqlen=128,
                dropout=0.0,
                pool="token",
            ):

        super(TextEncoder, self).__init__()

        self.batch_size = batch_size
        self.model_dimension = model_dimension
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_seqlen = max_seqlen
        self.token_embedding = nn.Embedding(self.vocab_size, self.model_dimension)
        self.dropout = dropout
        self.pool = pool

        self.scale = model_dimension ** -0.5

        self.feat_token = nn.Parameter(self.scale * torch.randn(1, self.model_dimension))

        self.pos_embedding = nn.Parameter(self.scale * torch.rand(self.max_seqlen+1, self.model_dimension))

        self.proj = nn.Linear(self.model_dimension, self.model_dimension)

        self.encoder_layer = nn.modules.TransformerEncoderLayer(d_model=self.model_dimension,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.model_dimension,
                                                            dropout=self.dropout,
                                                            activation='gelu')
        
        self.encoder = nn.modules.TransformerEncoder(encoder_layer=self.encoder_layer,
                                                                    num_layers=self.num_layers)

        
        self.fc = nn.Sequential(
            nn.LayerNorm(self.model_dimension),
            nn.Linear(self.model_dimension, self.model_dimension)
        )

    def forward(self, x):
        feat_token = self.feat_token.repeat(x.shape[0], 1).unsqueeze(1)
        x = self.token_embedding(x)
        x = self.proj(x)

        x = torch.cat((x, feat_token,), dim=1)
        x = x + self.pos_embedding.to(x.dtype)

        x = x.permute(1, 0, 2) # [N x S x D] -> [S x N x D]
        x = self.encoder(x)
        x = x.permute(1, 0, 2) # [S x N x D] -> [N x S x D]

        x = x[:, -1, :].squeeze() if self.pool == "token" else x.mean(dim=1).squeeze() # [N x D]

        x = self.fc(x)
        
        return x
       
class VisionEncoder(nn.Module):
    def __init__(self, 
                model_dimension=1024,
                batch_size=8,
                frame_size=224,
                space_patch_size=28,
                time_patch_size=4,
                num_frames=16,
                num_patches=8,
                max_seqlen=256,
                num_channels=3,
                num_heads=8, 
                num_layers=8,
                dropout=0.0,
                pool="token",
    ):

        super(VisionEncoder, self).__init__()

        self.model_dimension = model_dimension
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.space_patch_size = space_patch_size
        self.time_patch_size = time_patch_size
        self.feature_dimension = num_channels * time_patch_size * (space_patch_size**2)
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.max_seqlen =  (num_frames//time_patch_size) * ((frame_size//space_patch_size)**2)
        self.num_channels = 3
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.pool = pool


        self.proj = nn.Linear(self.feature_dimension, self.model_dimension)

        self.input_shape = (self.batch_size, self.num_frames, self.num_patches**2, self.model_dimension)

        self.scale = model_dimension ** -0.5

        self.feat_token = nn.Parameter(self.scale * torch.randn(1, self.model_dimension))

        self.pos_embedding = nn.Parameter(self.scale * torch.rand(self.max_seqlen+1, self.model_dimension))

        self.encoder_layer = nn.modules.TransformerEncoderLayer(d_model=self.model_dimension,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.model_dimension,
                                                            dropout=self.dropout,
                                                            activation='gelu')
        
        self.encoder = nn.modules.TransformerEncoder(encoder_layer=self.encoder_layer,
                                                                    num_layers=self.num_layers)

                                        
        self.fc = nn.Sequential(
            nn.LayerNorm(self.model_dimension),
            nn.Linear(self.model_dimension, self.model_dimension)
        )
    def forward(self, x):
        # Input: (N, T, H, W, C]
        n, t, h, w, c = x.shape 

        f, d = self.feature_dimension, self.model_dimension
        sp, tp = self.space_patch_size, self.time_patch_size

        x = x.unfold(1, tp, tp).unfold(2, sp, sp).unfold(3, sp, sp)
        x = x.permute(0, 1, 2, 3, 5, 6, 7, 4)
        x = x.reshape(*x.shape[:-4], -1)

        x = self.proj(x)
        x = x.reshape(n, -1, d)  # [N x T/tp x H/sp x W/sp x D] -> [N x S x D]

        feat_token = self.feat_token.repeat(x.shape[0], 1).unsqueeze(1)
        x = torch.cat((x, feat_token,), dim=1)
        x = x + self.pos_embedding.to(x.dtype)

        x = x.permute(1, 0, 2) # [N x S x D] -> [S x N x D]
        x = self.encoder(x)
        x = x.permute(1, 0, 2) # [S x N x D] -> [N x S x D]

        x = x[:, -1, :].squeeze() if self.pool == "token" else x.mean(dim=1).squeeze() # [N x D]
        x = self.fc(x)
        return x


class LAVA(nn.Module):
    def __init__(self, 
                model_dimension=1024, 
                feature_dimension=512,
                a_seqlen=64,
                v_seqlen=256,
                t_seqlen=128,
                batch_size=20, 
                learning_rate=3e-4,
                num_heads=8, 
                num_layers=8,
                dropout=0.0,
                pretrained_text=False):

        super(LAVA, self).__init__()

        self.model_dimension = model_dimension
        self.feature_dimension = feature_dimension
        self.batch_size = batch_size
        self.a_seqlen = a_seqlen
        self.v_seqlen = v_seqlen
        self.t_seqlen = t_seqlen
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.pretrained_text = pretrained_text

        self.cos_sim = nn.CosineSimilarity()

        self.temperature = nn.Parameter(torch.tensor([14.285]), requires_grad=True)

        self.a_encoder = AudioEncoder(
                                model_dimension=self.model_dimension,
                                batch_size=self.batch_size,
                                num_heads=self.num_heads,
                                num_layers=self.num_layers)


        self.v_encoder = VisionEncoder(
                                model_dimension=self.model_dimension,
                                batch_size=self.batch_size,
                                num_heads=self.num_heads,
                                num_layers=self.num_layers)

        self.t_encoder = TextEncoder(
                                    model_dimension=self.model_dimension,
                                    batch_size=self.batch_size,
                                    num_heads=self.num_heads,
                                    num_layers=self.num_layers,
                                    max_seqlen=self.t_seqlen,)

        self.i_encoder = VisionEncoder(
                            model_dimension=self.model_dimension,
                            batch_size=self.batch_size,
                            num_heads=self.num_heads,
                            num_layers=self.num_layers,
                            num_frames=4,
                            frame_size=128,
                            space_patch_size=32,
                            time_patch_size=1,
                            num_patches=4,)
                           

        self.a_proj = nn.Linear(self.model_dimension, self.model_dimension)
        self.v_proj = nn.Linear(self.model_dimension, self.model_dimension)
        self.t_proj = nn.Linear(self.model_dimension, self.model_dimension)
        self.i_proj = nn.Linear(self.model_dimension, self.model_dimension)


    def encode_audio(self, x):
        x = self.a_encoder(x)
        return x

    def encode_video(self, x):
        x = self.v_encoder(x)
        return x

    def encode_text(self, x):
        x = self.t_encoder(x)
        return x

    def encode_images(self, x):
        x = self.i_encoder(x)
        return x

    def forward(self, a, v, t, i):
        a = self.encode_audio(a)
        v = self.encode_video(v)
        t = self.encode_text(t)
        i = self.encode_images(i)

        a = nn.functional.normalize(a, p=2, dim=-1)
        v = nn.functional.normalize(v, p=2, dim=-1)
        t = nn.functional.normalize(t, p=2, dim=-1)
        i = nn.functional.normalize(i, p=2, dim=-1)

        return a, v, t, i

    def norm(self, x):
        return nn.functional.normalize(x, p=2, dim=-1)

    def loss(self, a, v, t, i, t_mask):
        # zero out t vectors from default string
        t = (t.T * t_mask).T

        a = self.norm(self.a_proj(a))
        v = self.norm(self.v_proj(v))
        t = self.norm(self.t_proj(t))
        i = self.norm(self.i_proj(i))

        # approximate centroid vector
        centroid = (a + v + t + i)/4
        centroid = self.norm(centroid)

        avt_loss = nce_loss(a, centroid, temp=self.temperature)
        avt_loss += nce_loss(v, centroid, temp=self.temperature)
        avt_loss += nce_loss(t, centroid, temp=self.temperature)
        avt_loss += nce_loss(i, centroid, temp=self.temperature)

        av_loss = nce_loss(a, v, temp=self.temperature)
        # at_loss =  nce_loss(a, t, temp=self.temperature)
        ai_loss = nce_loss(a, i, temp=self.temperature)

        vt_loss = nce_loss(v, t, temp=self.temperature)
        vi_loss = nce_loss(v, i, temp=self.temperature)

        # it_loss = nce_loss(i, t, temp=self.temperature)

        # loss = avt_loss + av_loss + at_loss + ai_loss + vt_loss + vi_loss
        loss = avt_loss + av_loss  + ai_loss + vt_loss + vi_loss

        av = a @ v.T
        at = a @ t.T
        vt = v @ t.T
        ai = a @ i.T
        vi = v @ i.T

        # av = torch.diag(av).mean()
        # at = torch.diag(at).mean()
        # vt = torch.diag(vt).mean()
        # ai = torch.diag(vt).mean()
        # vi = torch.diag(vt).mean()

        # av_cos_sim_neg = torch.triu(av, diagonal=1).mean()
        # at_cos_sim_neg = torch.triu(at, diagonal=1).mean()
        # vt_cos_sim_neg = torch.triu(vt, diagonal=1).mean()
        # ai_cos_sim_neg = torch.triu(ai, diagonal=1).mean()
        # vi_cos_sim_neg = torch.triu(vi, diagonal=1).mean()

        labels = torch.arange(self.batch_size).to(device=v.device)
        av_top1 = 0.5 * (compute_accuracy(av, labels , top_k=1) +  compute_accuracy(av.T, labels , top_k=1))
        # at_top1 = 0.5 * (compute_accuracy(at, labels , top_k=1) +  compute_accuracy(at.T, labels , top_k=1))
        vt_top1 = 0.5 * (compute_accuracy(vt, labels , top_k=1) +  compute_accuracy(vt.T, labels , top_k=1))
        ai_top1 = 0.5 * (compute_accuracy(ai, labels , top_k=1) +  compute_accuracy(ai.T, labels , top_k=1))
        vi_top1 = 0.5 * (compute_accuracy(vi, labels , top_k=1) +  compute_accuracy(vi.T, labels , top_k=1))

        metrics = {
            'loss': loss.item(),
            'av_loss': av_loss.item(),
            # 'at_loss': at_loss.item(),
            'vt_loss': vt_loss.item(),
            'ai_loss': ai_loss.item(),
            'vi_loss': vi_loss.item(),
            'av_top1': av_top1.item(),
            # 'at_top1': at_top1.item(),
            'vt_top1': vt_top1.item(),
            'ai_top1': ai_top1.item(),
            'vi_top1': vi_top1.item(),
            'temp': self.temperature,
            'vt_matrix': vt.T.detach().to(dtype=torch.float32),
        }
        
        return loss, metrics
       
class LinearClassifierAVT(torch.nn.Module):
    def __init__(self,
                model,
                data,
                num_classes=700,
                feature_dimension=1024,
                model_dimension=1024,
                num_modalities=3,
                batch_size=20,
                learning_rate=1e-3,
                model_path=None,
                pretrained_text=True):

        super(LinearClassifierAVT, self).__init__()

        self.data = data
        self.num_classes = 700
        self.feature_dimension = feature_dimension
        self.model_dimension = model_dimension
        self.num_modalities = num_modalities
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pretrained_text = pretrained_text

        self.model = model(batch_size=batch_size,
                        model_dimension=model_dimension,
                        num_heads=4,
                        num_layers=4,
                        pretrained_text=pretrained_text)

        self.model.load_state_dict(torch.load(model_path, map_location='cuda:1')['state_dict'], strict=True)

        self.fc = torch.nn.Linear(self.num_modalities * self.model_dimension, self.num_classes)

    
    def forward(self, a, v, t):
        with torch.no_grad():
            v = self.model.encoder.v_encoder(v)
            # t = self.model.encoder.t_encoder(t)


        similarity = torch.zeros(32, 32).to(device=v.device)
        pred = self.fc(v)
        return pred, similarity
