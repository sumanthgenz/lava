import torch
import torch.nn as nn
import torchaudio
import torchvision
import torchtext
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.checkpoint import checkpoint_sequential, checkpoint
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

from aai.experimental.sgurram.lava.src.metrics import nce_loss, centroid_loss, instance_loss
from aai.experimental.sgurram.lava.src.references import lava_weights_path, sp_model_path, sp_vocab_size
from aai.experimental.sgurram.lava.src.utils import get_src_conditional_mask, position_embed, position_embed_3d, pad_batch
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
                frame_size=128,
                space_patch_size=16,
                time_patch_size=4,
                num_frames=16,
                num_patches=4,
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
        self.max_seqlen =  num_frames * (num_patches**2)
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


        self.a_proj = nn.Linear(self.model_dimension, self.model_dimension)
        

        self.t_proj = nn.Linear(self.model_dimension, self.model_dimension)
        
        self.v_proj = nn.Sequential(
            nn.Linear(self.model_dimension, self.model_dimension),
            nn.BatchNorm1d(self.model_dimension),
            nn.GELU(),
            nn.Linear(self.model_dimension, self.model_dimension),
        )

    def encode_audio(self, x):
        x = self.a_encoder(x)
        x = self.a_proj(x)
        return x

    def encode_video(self, x):
        x = self.v_encoder(x)
        x = self.v_proj(x)
        return x

    def encode_text(self, x):
        x = self.t_encoder(x)
        x = self.t_proj(x)
        return x

    def forward(self, a, v1, v2, t):
        a = self.encode_audio(a)
        v1 = self.encode_video(v1)
        v2 = self.encode_video(v2)
        t = self.encode_text(t)

        a = nn.functional.normalize(a, p=2, dim=-1)
        v1 = nn.functional.normalize(v1, p=2, dim=-1)
        v2 = nn.functional.normalize(v2, p=2, dim=-1)
        t = nn.functional.normalize(t, p=2, dim=-1)

        return a, v1, v2, t

    def loss(self, a, v1, v2, t, t_mask):
        # zero out t vectors from default string
        t = (t.T * t_mask).T

        # approximate centroid vector
        centroid = v2
        # centroid = (a + v + t)/3
        # centroid = nn.functional.normalize(centroid, p=2, dim=-1)

        av_loss = nce_loss(a, centroid, temp=self.temperature)
        vv_loss = nce_loss(v1, centroid, temp=self.temperature)
        vt_loss = nce_loss(t, centroid, temp=self.temperature)

        loss = av_loss + vv_loss + vt_loss
        # loss += nce_loss(a, v, temp=self.temperature)
        # loss += nce_loss(a, t, temp=self.temperature)
        # loss += nce_loss(v, t, temp=self.temperature)

        av = a @ v2.T
        at = a @ t.T
        vv = v1 @ v2.T
        vt = v2 @ t.T

        av_cos_sim = torch.diag(av).mean()
        at_cos_sim = torch.diag(at).mean()
        vv_cos_sim = torch.diag(vv).mean()
        vt_cos_sim = torch.diag(vt).mean()

        av_cos_sim_neg = torch.triu(av, diagonal=1).mean()
        at_cos_sim_neg = torch.triu(at, diagonal=1).mean()
        vv_cos_sim_neg =  torch.triu(vv, diagonal=1).mean()
        vt_cos_sim_neg = torch.triu(vt, diagonal=1).mean()

        a_cos_sim = torch.triu(a @ a.T, diagonal=1).mean()
        v_cos_sim = torch.triu(v1 @ v1.T, diagonal=1).mean()
        t_cos_sim = torch.triu(t @ t.T, diagonal=1).mean()

        metrics = {
            'loss': loss.item(),
            'av_loss': av_loss.item(),
            'vv_loss': vv_loss.item(),
            'vt_loss': vt_loss.item(),
            'cos_sim_a_neg': a_cos_sim.item(),
            'cos_sim_v_neg': v_cos_sim.item(),
            'cos_sim_t_neg': t_cos_sim.item(),
            'cos_sim_av': av_cos_sim.item(),
            'cos_sim_at': at_cos_sim.item(),
            'cos_sim_vv': vv_cos_sim.item(),
            'cos_sim_vt': vt_cos_sim.item(),
            'cos_sim_av_neg': av_cos_sim_neg.item(),
            'cos_sim_at_neg': at_cos_sim_neg.item(),
            'cos_sim_vv_neg': vv_cos_sim_neg.item(),
            'cos_sim_vt_neg': vt_cos_sim_neg.item(),
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
        self.num_classes = 101
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

        # self.model = model
        self.model.load_state_dict(torch.load(model_path, map_location='cuda:1')['state_dict'], strict=True)

        # self.model = LAVA(batch_size=batch_size,
        #                 model_dimension=feature_dimension)

        # self.model.load_state_dict(torch.load('overfit_lava.pt'), strict=True)
 
        # print(torch.load(model_path)['state_dict'].keys())
        # self.model.encoder.eval()
        # print(self.model.encoder.pairwise_temp)
        # print(self.model.encoder.triplet_temp)

        # self.a_encoder = self.model.encoder.a_encoder
        # self.v_encoder = self.model.encoder.v_encoder
        # self.t_encoder = self.model.encoder.t_encoder

        # self.model.eval()
        # self.a_encoder = self.model.a_encoder
        # self.v_encoder = self.model.v_encoder
        # self.t_encoder = self.model.t_encoder
        # self.fc = torch.nn.Sequential(
        #     nn.Linear(self.num_modalities * self.model_dimension, self.model_dimension),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(self.model_dimension),
        #     nn.Linear(self.model_dimension, self.model_dimension),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(self.model_dimension),
        #     nn.Linear(self.model_dimension, self.model_dimension),
        #     nn.ReLU(),
        #     nn.Linear(self.model_dimension, self.num_classes),
        # )
        # self.projection = torch.nn.Linear(512, 1024)
        self.fc = torch.nn.Linear(self.num_modalities * self.model_dimension, self.num_classes)
        # self.fc1 = torch.nn.Linear(self.model_dimension, self.num_classes)

    
    def forward(self, a, v, t):
        # with torch.no_grad():
        #     a = self.model.encoder.encode_audio(a)
        v = self.model.encoder.v_encoder(v)
            # t = self.model.encoder.t_encoder(t)

        #     a = nn.functional.normalize(a, p=2, dim=-1)
            # v = nn.functional.normalize(v, p=2, dim=-1)
            # t = nn.functional.normalize(t, p=2, dim=-1)

        # if self.pretrained_text:
            # representation = torch.cat((a,v,t), dim=-1)
            # representation = torch.cat((a,v), dim=-1)
            # similarity = (t @ v.T).detach()
            # t = t.squeeze()
            # with torch.no_grad():
            #     representation = self.projection(t)
            #     similarity = None

        # representation = t
        # similarity = (t @ v.T).detach()
        similarity = torch.zeros(32, 32).to(device=v.device)
        pred = self.fc(v)
        return pred, similarity

if __name__=="__main__":
    # v = torch.rand(8, 16, 128, 128, 3)
    data = LAVAData("val", False)
    lava = LAVA()
    
    a, v, t, url, mask = data[12]
    # v = torch.from_numpy(v)
    v = v.unsqueeze(0)
    # print(v.shape)
    b, t, h, w, c = v.shape

    # # v = torch.arange(1*16*128*128*3).reshape(1, 16, 128, 128, 3)
    p = 16
    # patches = v.reshape(b, t//p, p, h//p, p, w//p, p, 3).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, -1, t//p, h//p, w//p, 3)
    patches = v.unfold(1, p, p).unfold(2, p, p).unfold(3, p, p)
    patches = patches.permute(0, 1, 2, 3, 5, 6, 7, 4)
    # patches = patches.reshape(*patches.shape[:-4], -1)
    print(patches.shape)
    # # print(v)
    tubelet = patches[0, 0, 2, 6, 2].squeeze()
    # plt.imshow(v[0, 0])
    plt.imshow(tubelet.to(dtype=torch.uint8))
    plt.savefig("a_tubelet_sanity")
    print(tubelet)
