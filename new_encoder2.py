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

class AudioEncoder(nn.Module):
    def __init__(self, 
                batch_size=8,
                model_dimension=1024,
                mel_freq=80, #80
                time_steps=2048,
                melresnet_dim=508,
                patch_size=32, #8
                max_seqlen=64, #64
                num_heads=8, 
                num_layers=8,
                dropout=0.0,):

        super(AudioEncoder, self).__init__()

        # MelT architecture (based on ViT)
        # Patches are time-slices of mel spectrogram with full frequency dimension per slice

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
        
        self.scale = model_dimension ** -0.5

        # Position Embedding
        self.pos_embedding = nn.Parameter(self.scale * torch.rand(self.max_seqlen+1, self.model_dimension))

        # CLS Token instead of mean pooling
        self.feat_token = nn.Parameter(self.scale * torch.randn(1, self.model_dimension))

        # 128 -> 64 -> 16
        self.freq_mlp = nn.Sequential(
            nn.Linear(self.mel_freq, self.mel_freq // 2),
            nn.Dropout(self.dropout),
            nn.Linear(self.mel_freq // 2, self.model_dimension // self.patch_size),
        )

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
       
        # Mel ResNet
        # Input [N x F x T], N = bsz, F = 128, T = 2048 

        n, f, t = x.shape
        s, p = t//self.patch_size, self.patch_size

        feat_token = self.feat_token.repeat(x.shape[0], 1).unsqueeze(1)

        # [N x T/P x P x F], S = T/P
        x = x.permute(0, 2, 1).unfold(1, p, p)

        # [(N x S x D], D = 1024
        x = self.freq_mlp(x.reshape(-1, f)).reshape(n, s, -1)
        
        # [(N x (S+1) x D], D = 1024
        x = torch.cat((x, feat_token,), dim=1)

        # x = position_embed(x)
        x = x + self.pos_embedding.to(x.dtype)

        x = self.encoder(
            src=x.transpose(0,1),
            mask=get_src_conditional_mask(self.max_seqlen+1).to(x.device)
        ).transpose(0,1)

        # [N, D]
        x = x[:, -1, :].squeeze() + x[:, :-1, :].mean(dim=1).squeeze()

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
                dropout=0.0,):

        super(TextEncoder, self).__init__()

        self.batch_size = batch_size
        self.model_dimension = model_dimension
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_seqlen = max_seqlen
        self.token_embedding = nn.Embedding(self.vocab_size, self.model_dimension)
        self.dropout = dropout

        self.scale = model_dimension ** -0.5

        # CLS Token
        self.feat_token = nn.Parameter(self.scale * torch.randn(1, self.model_dimension))

        # Position Embedding
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
        feat_token = self.feat_token.repeat(x.shape[0], 1).unsqueeze(1)
        x = self.token_embedding(x)
        x = torch.cat((x, feat_token,), dim=1)

        # x = position_embed(x)
        x = x + self.pos_embedding.to(x.dtype)

        x = self.encoder(
            src=x.transpose(0,1),
            mask=get_src_conditional_mask(self.max_seqlen+1).to(x.device)
        ).transpose(0,1)

        x = x[:, -1, :].squeeze() + x[:, :-1, :].mean(dim=1).squeeze()
        x = self.fc(x)
        return x

class TimeSformerBlock(nn.Module):
    def __init__(
        self,
        vision_shape,
        model_dimension=1024,
        num_heads = 8,
        dropout = 0.0):

        super().__init__()

        self.vision_shape = vision_shape
        self.model_dimension = model_dimension # 1024
        self.num_heads = num_heads # 8
        self.head_dim = model_dimension // num_heads #128
        self.dropout = dropout


        self.time_attn = nn.MultiheadAttention(self.model_dimension, self.num_heads)
        self.space_attn = nn.MultiheadAttention(self.model_dimension, self.num_heads)
        self.mlp = nn.Sequential(
                            nn.Linear(self.model_dimension, self.model_dimension * 4),
                            nn.GELU(),
                            nn.Dropout(self.dropout),
                            nn.Linear(self.model_dimension * 4, self.model_dimension),)   

        self.t_ln = nn.LayerNorm(self.model_dimension)
        self.s_ln = nn.LayerNorm(self.model_dimension)
        self.m_ln = nn.LayerNorm(self.model_dimension)

    # def attention(self, x, attn_mask=None):
    #     return self.attn(x, x, x, need_weights=False, attn_mask=None)[0]

    def forward(self, 
                src, 
                src_mask=None, 
                src_key_padding_mask=None):

        n, f, s, d = self.vision_shape
        n = src.shape[1]
        x = src

        # input x has shape [FS x N x D]

        # time attn
        x = x.reshape(f, -1, d)  # input x to [F x SN x D]
        x = x + self.time_attn(x, x, x, need_weights=False, attn_mask=src_mask)[0]
        x = self.s_ln(x)

        # space attn
        x = x.reshape(f, s, n, d).permute(1, 0, 2, 3).reshape(s, -1, d)  # input x to [S x FN x D]
        x = x + self.space_attn(x, x, x, need_weights=False, attn_mask=src_mask)[0]
        x = self.t_ln(x)

        # mlp
        x = x.reshape(s, f, n, d).permute(1, 0, 2, 3).reshape(-1, n, d) # input x to [FS x N x D]
        x = x + self.mlp(x)
        x = self.m_ln(x)

        return x
       
class VisionEncoder(nn.Module):
    def __init__(self, 
                model_dimension=1024,
                batch_size=8,
                frame_size=128,
                patch_size=32,
                num_frames=16,
                num_patches=4,
                max_seqlen=256,
                num_channels=3,
                num_heads=8, 
                num_layers=8,
                dropout=0.0,
                resnet=18):

        super(VisionEncoder, self).__init__()

        # Based on TimeSformer Divided Space-Time Attention from Facebook AI
        # Based on ViT from OpenAI's CLIP

        # assert frame_size == patch_size * num_patches, f"frame_size {frame_size} does not match patch_size {patch_size} and num_patches {num_patches}"

        self.model_dimension = model_dimension
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.max_seqlen =  num_frames * (num_patches**2)
        self.num_channels = 3
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        if resnet==50:
            if self.num_patches == 4:
                resnet_layer_idx = -2
                self.feature_dimension = 2048
            elif self.num_patches == 8:
                resnet_layer_idx = -3
                self.feature_dimension = 1024
            elif self.num_patches == 16:
                resnet_layer_idx = -4
                self.feature_dimension = 512
        else: #resnet 18
            if self.num_patches == 4:
                resnet_layer_idx = -2
                self.feature_dimension = 512
            elif self.num_patches == 8:
                resnet_layer_idx = -3
                self.feature_dimension = 256
            elif self.num_patches == 16:
                resnet_layer_idx = -4
                self.feature_dimension = 128

        if resnet==50:
            base_model = torchvision.models.resnet50(pretrained=True)
        else:
            base_model = torchvision.models.resnet18(pretrained=True)

        self.resnet_model =  torch.nn.Sequential(
            *(list(base_model.children())[:resnet_layer_idx]))

        # self.feature_dimension = self.num_channels * self.patch_size**2

        self.input_shape = (self.batch_size, self.num_frames, self.num_patches**2, self.model_dimension)

        self.scale = model_dimension ** -0.5

        # CLS Token
        self.feat_token = nn.Parameter(self.scale * torch.randn(1, self.feature_dimension))

        # Position Embedding
        self.pos_embedding = nn.Parameter(self.scale * torch.rand(self.max_seqlen, self.model_dimension))


        self.in_fc = nn.Linear(self.feature_dimension, self.model_dimension)

        self.encoder_layer = TimeSformerBlock(vision_shape=self.input_shape, 
                                            model_dimension=self.model_dimension, 
                                            num_heads=self.num_heads) 
                                            
        self.timesformer = nn.TransformerEncoder(self.encoder_layer, self.num_layers)

        # self.timesformer = nn.Sequential(*[
        #             TimeSformerBlock(self.input_shape, self.model_dimension, self.num_heads) 
        #             for _ in range(self.num_layers)]
        # )

        self.ln = nn.LayerNorm(self.model_dimension)

    def forward(self, x):

        # Input: (N, F, H, W, C]

        n, f, h, w, c = x.shape 
        d = self.feature_dimension

        x = x.permute(0, 1, 4, 2, 3).reshape(-1, c, h, w)  # [NF x C x H x W]
        with torch.no_grad():
            x = self.resnet_model(x.contiguous()).permute(0, 2, 3, 1)  # [NF x P x P x D], S = P**2
        x = x.reshape(n, f, -1, d) # [N x F x S x 2D]
        x = x.reshape(n, -1, x.shape[-1])  # [N x FS x 2D]

        # x = torch.rand(20, 256, 512).to(x.device)
        x = self.in_fc(x) # [N x FS x D]

        # x = torch.rand(self.batch_size, self.num_frames**2 * self.batch_size, self.model_dimension)
        # x = torch.rand(20, 256, 512).to(x.device)

        # feat_token = self.feat_token.repeat(x.shape[0], 1).unsqueeze(1) # [N x 1 x D]
        # x = torch.cat((x, feat_token), dim=1) # [N x (FS+1) x D]

        # x = position_embed(x)
        x = x + self.pos_embedding.to(x.dtype)

        x = x.permute(1, 0, 2) # [(FS) x N x D]
        x = self.timesformer(x)
        x = x.permute(1, 0, 2) # [N x (FS) x D]

        x = x.mean(dim=1).squeeze() # [ N x D]
        x = self.ln(x)

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

        self.pairwise_temp = nn.Parameter(torch.ones([]), requires_grad=True)
        self.triplet_temp = nn.Parameter(torch.ones([]), requires_grad=True)

        # self.pairwise_temp = nn.Parameter(torch.tensor([14.285]), requires_grad=True)
        # self.triplet_temp = nn.Parameter(torch.tensor([14.285]), requires_grad=True)

        self.a_encoder = AudioEncoder(
                                model_dimension=self.model_dimension,
                                batch_size=self.batch_size,
                                num_heads=self.num_heads,
                                num_layers=self.num_layers,
                                max_seqlen=self.a_seqlen,)

        self.v_encoder = VisionEncoder(
                                model_dimension=self.model_dimension,
                                batch_size=self.batch_size,
                                num_heads=self.num_heads,
                                num_layers=self.num_layers,
                                max_seqlen=self.v_seqlen,)

        if not self.pretrained_text:
            self.t_encoder = TextEncoder(
                                    model_dimension=self.model_dimension,
                                    batch_size=self.batch_size,
                                    num_heads=self.num_heads,
                                    num_layers=self.num_layers,
                                    max_seqlen=self.t_seqlen,)

        else:
            self.t_encoder = nn.Linear(512, self.model_dimension)

        # self.a_proj = nn.Sequential(
        #     nn.Linear(self.model_dimension, self.model_dimension),
        # )

        # self.v_proj = nn.Sequential(
        #     nn.Linear(self.model_dimension, self.model_dimension),
        # )

        # self.t_proj = nn.Sequential(
        #     nn.Linear(self.model_dimension, self.model_dimension),
        # )

        # self.a_proj = nn.Sequential(
        #     nn.Linear(self.model_dimension, self.model_dimension),
        #     nn.BatchNorm1d(self.model_dimension),
        #     nn.GELU(),
        #     nn.Linear(self.model_dimension, self.model_dimension),
        # )
        
        # self.v_proj = nn.Sequential(
        #     nn.Linear(self.model_dimension, self.model_dimension),
        #     nn.BatchNorm1d(self.model_dimension),
        #     nn.GELU(),
        #     nn.Linear(self.model_dimension, self.model_dimension),
        # )

        # self.t_proj = nn.Sequential(
        #     nn.Linear(self.model_dimension, self.model_dimension),
        #     nn.BatchNorm1d(self.model_dimension),
        #     nn.GELU(),
        #     nn.Linear(self.model_dimension, self.model_dimension),
        # )

    def encode_audio(self, x):
        x = self.a_encoder(x)
        # x = self.a_proj(x)
        return x

    def encode_video(self, x):
        x = self.v_encoder(x)
        # x = self.v_proj(x)
        return x

    def encode_text(self, x):
        x = self.t_encoder(x)
        # x = self.t_proj(x)
        return x

    def forward(self, a, v, t):
        a = self.encode_audio(a)
        v = self.encode_video(v)

        a = nn.functional.normalize(a, p=2, dim=-1)
        v = nn.functional.normalize(v, p=2, dim=-1)

        if not self.pretrained_text:
            t = self.encode_text(t)
            t = nn.functional.normalize(t, p=2, dim=-1)
        else:
            t = t.squeeze()
            with torch.no_grad():
                t = self.t_encoder(t)

        return a, v, t

    def loss_av(self, a, v):

        av_loss = nce_loss(a, v, temp=self.pairwise_temp)
        av = a @ v.T
        av_cos_sim = torch.diag(av).mean()
        av_cos_sim_neg = torch.triu(av, diagonal=1).mean()
        a_cos_sim = torch.triu(a @ a.T, diagonal=1).mean()
        v_cos_sim = torch.triu(v @ v.T, diagonal=1).mean()
        total_loss = av_loss 

        metrics = {
            'loss_av': av_loss.item(),
            'loss': total_loss.item(),
            'cos_sim_a_neg': a_cos_sim.item(),
            'cos_sim_v_neg': v_cos_sim.item(),
            'cos_sim_av': av_cos_sim.item(),
            'cos_sim_av_neg': av_cos_sim_neg.item(),
            'pairwise_temp': self.pairwise_temp,
            'triplet_temp': self.triplet_temp,
            'av_matrix': av.T.detach(),
        }
        return total_loss, metrics

    def loss_vt(self, v, t, t_mask):

        t = (t.T * t_mask).T

        vt_loss = nce_loss(v, t, temp=self.pairwise_temp)
        vt = v @ t.T
        vt_cos_sim = torch.diag(vt).mean()
        vt_cos_sim_neg = torch.triu(vt, diagonal=1).mean()
        v_cos_sim = torch.triu(v @ v.T, diagonal=1).mean()
        t_cos_sim = torch.triu(t @ t.T, diagonal=1).mean()
        total_loss = vt_loss 

        metrics = {
            'loss_av': av_loss.item(),
            'loss': total_loss.item(),
            'cos_sim_a_neg': a_cos_sim.item(),
            'cos_sim_v_neg': v_cos_sim.item(),
            'cos_sim_av': av_cos_sim.item(),
            'cos_sim_av_neg': av_cos_sim_neg.item(),
            'pairwise_temp': self.pairwise_temp,
            'triplet_temp': self.triplet_temp,
            'vt_matrix': vt.T.detach().to(dtype=torch.float32),
        }
        return total_loss, metrics
    
    def loss(self, a, v, t, t_mask):
        if not self.pretrained_text:
            # zero out t vectors from default string
            t = (t.T * t_mask).T

            pairwise_temp = self.pairwise_temp
            triplet_temp = self.triplet_temp
            # pairwise_temp = torch.ones([])
            # triplet_temp = torch.ones([])

            av_loss = nce_loss(a, v, temp=pairwise_temp)
            at_loss = nce_loss(a, t, temp=pairwise_temp)
            vt_loss = nce_loss(v, t, temp=pairwise_temp)

            centroid = (a + v + t)/3
            centroid = nn.functional.normalize(centroid, p=2, dim=-1)

            avt_loss = nce_loss(a, centroid, temp=triplet_temp)
            avt_loss += nce_loss(v, centroid, temp=triplet_temp)
            avt_loss += nce_loss(t, centroid, temp=triplet_temp)

            av = a @ v.T
            at = a @ t.T
            vt = v @ t.T

            av_cos_sim = torch.diag(av).mean()
            at_cos_sim = torch.diag(at).mean()
            vt_cos_sim = torch.diag(vt).mean()

            av_cos_sim_neg = torch.triu(av, diagonal=1).mean()
            at_cos_sim_neg = torch.triu(at, diagonal=1).mean()
            vt_cos_sim_neg = torch.triu(vt, diagonal=1).mean()

            a_cos_sim = torch.triu(a @ a.T, diagonal=1).mean()
            v_cos_sim = torch.triu(v @ v.T, diagonal=1).mean()
            t_cos_sim = torch.triu(t @ t.T, diagonal=1).mean()

            total_loss = av_loss + at_loss + vt_loss + avt_loss

            metrics = {
                'loss_av': av_loss.item(),
                'loss_at': at_loss.item(),
                'loss_vt': vt_loss.item(),
                'loss_avt': avt_loss.item(),
                'loss': total_loss.item(),
                'cos_sim_a_neg': a_cos_sim.item(),
                'cos_sim_v_neg': v_cos_sim.item(),
                'cos_sim_t_neg': t_cos_sim.item(),
                'cos_sim_av': av_cos_sim.item(),
                'cos_sim_at': at_cos_sim.item(),
                'cos_sim_vt': vt_cos_sim.item(),
                'cos_sim_av_neg': av_cos_sim_neg.item(),
                'cos_sim_at_neg': at_cos_sim_neg.item(),
                'cos_sim_vt_neg': vt_cos_sim_neg.item(),
                'pairwise_temp': self.pairwise_temp,
                'triplet_temp': self.triplet_temp,
                'vt_matrix': vt.T.detach().to(dtype=torch.float32),
            }
        else:
            t = (t.T * t_mask).T

            pairwise_temp = self.pairwise_temp
            triplet_temp = self.triplet_temp
            # pairwise_temp = torch.ones([]).to(device=self.pairwise_temp.device)
            # triplet_temp = torch.ones([]).to(device=self.pairwise_temp.device)
            # pairwise_temp = torch.tensor([0.07]).to(device=self.pairwise_temp.device)
            # triplet_temp = torch.tensor([0.7]).to(device=self.pairwise_temp.device)

            # centroid = nn.functional.normalize(t, p=2, dim=-1)
            centroid = t

            av_loss = nce_loss(a, v, temp=pairwise_temp)
            at_loss = nce_loss(a, centroid, temp=triplet_temp)
            vt_loss = nce_loss(v, centroid, temp=triplet_temp)

            # avt_loss = centroid_loss(a, v, t)

            av = a @ v.T
            at = a @ t.T
            vt = v @ t.T

            av_cos_sim = torch.diag(av).mean()
            at_cos_sim = torch.diag(at).mean()
            vt_cos_sim = torch.diag(vt).mean()

            av_cos_sim_neg = torch.triu(av, diagonal=1).mean()
            at_cos_sim_neg = torch.triu(at, diagonal=1).mean()
            vt_cos_sim_neg = torch.triu(vt, diagonal=1).mean()

            a_cos_sim = torch.triu(a @ a.T, diagonal=1).mean()
            v_cos_sim = torch.triu(v @ v.T, diagonal=1).mean()
            t_cos_sim = torch.triu(t @ t.T, diagonal=1).mean()

            total_loss = av_loss + at_loss + vt_loss
            metrics = {
                'loss_av': av_loss.item(),
                'loss_at': at_loss.item(),
                'loss_vt': vt_loss.item(),
                'loss': total_loss.item(),
                'cos_sim_a_neg': a_cos_sim.item(),
                'cos_sim_v_neg': v_cos_sim.item(),
                'cos_sim_t_neg': t_cos_sim.item(),
                'cos_sim_av': av_cos_sim.item(),
                'cos_sim_at': at_cos_sim.item(),
                'cos_sim_vt': vt_cos_sim.item(),
                'cos_sim_av_neg': av_cos_sim_neg.item(),
                'cos_sim_at_neg': at_cos_sim_neg.item(),
                'cos_sim_vt_neg': vt_cos_sim_neg.item(),
                'pairwise_temp': self.pairwise_temp,
                'triplet_temp': self.triplet_temp,
                'vt_matrix': vt.T.detach().to(dtype=torch.float32),
            }
        return total_loss, metrics

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
        self.num_classes = num_classes
        self.feature_dimension = feature_dimension
        self.model_dimension = model_dimension
        self.num_modalities = num_modalities
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pretrained_text = pretrained_text

        # self.model_path = model_path if model_path else lava_weights_path
        # self.model_path = "/home/sgurram/Projects/aai/aai/experimental/sgurram/lava/src/wandb/run-20210322_193330-131o6s57/files/lava/131o6s57/checkpoints/epoch=0-step=1430.ckpt"
        # self.model_path = "/home/sgurram/Projects/aai/aai/experimental/sgurram/lava/src/wandb/run-20210323_222842-14ahw6dm/files/lava/14ahw6dm/checkpoints/epoch=34-step=25059.ckpt"

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
        self.model.encoder.eval()
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
        with torch.no_grad():
        #     a = self.model.encoder.encode_audio(a)
            v = self.model.encoder.v_encoder(v)
            t = self.model.encoder.t_encoder(t)

        #     a = nn.functional.normalize(a, p=2, dim=-1)
            # v = nn.functional.normalize(v, p=2, dim=-1)
            # t = nn.functional.normalize(t, p=2, dim=-1)

        if self.pretrained_text:
            # representation = torch.cat((a,v,t), dim=-1)
            # representation = torch.cat((a,v), dim=-1)
            # similarity = (t @ v.T).detach()
            t = t.squeeze()
            # with torch.no_grad():
            #     representation = self.projection(t)
            #     similarity = None

        # representation = t
        similarity = (t @ v.T).detach()
        pred = self.fc(v)
        return pred, similarity


