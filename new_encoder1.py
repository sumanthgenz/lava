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

import warnings
import glob
import pickle
import copy
import os
import sys

from references import sp_model_path, sp_vocab_size
from utils import attention

from aai.experimental.sgurram.lava.src.metrics import nce_loss, centroid_loss, instance_loss
from aai.experimental.sgurram.lava.src.references import lava_weights_path
# from aai.alexandria.layers.functional.positional_embedding import position_embed, position_embed_3d
from aai.experimental.sgurram.lava.src.utils import get_src_conditional_mask, position_embed, position_embed_3d
from aai.experimental.sgurram.lava.src.utils import visualize_batch


class AudioEncoder(nn.Module):
    def __init__(self, 
                batch_size=8,
                model_dimension=512,
                mel_freq=128,
                time_steps=2048,
                melresnet_dim=508,
                patch_size=32,
                max_seqlen=64,
                num_heads=8, 
                num_layers=8,
                dropout=0.1,):

        super(AudioEncoder, self).__init__()

        # MelT architecture (based on ViT)
        # Patches are time-slices of mel spectrogram with full frequency dimension per slice

        self.batch_size = batch_size
        self.model_dimension = model_dimension
        self.mel_freq = mel_freq
        self.time_steps = time_steps
        self.melresnet_dim = melresnet_dim
        self.max_seqlen = max_seqlen
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.dropout = dropout

        # CLS Token instead of mean pooling
        self.feat_token = nn.Parameter(torch.randn(1, self.model_dimension))

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

        x = position_embed(x)

        x = self.encoder(
            src=x.transpose(0,1),
            mask=get_src_conditional_mask(self.max_seqlen+1).to(x.device)
        ).transpose(0,1)

        # [D]
        out = x[:, -1].squeeze()

        out = self.fc(out)

        return out


class TextEncoder(nn.Module):
    def __init__(self,
                batch_size=8,
                model_dimension=1024,
                num_heads=8, 
                num_layers=8,
                vocab_size=48000,
                max_seqlen=128,
                dropout=0.1,):

        super(TextEncoder, self).__init__()

        self.batch_size = batch_size
        self.model_dimension = model_dimension
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_seqlen = max_seqlen
        self.token_embedding = nn.Embedding(self.vocab_size, self.model_dimension)
        self.feat_token = nn.Parameter(torch.randn(1, self.model_dimension))
        self.dropout = dropout

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
        x = position_embed(x)
        x = self.encoder(
            src=x.transpose(0,1),
            mask=get_src_conditional_mask(self.max_seqlen+1).to(x.device)
        ).transpose(0,1)

        out = x[:, -1].squeeze()
        out = self.fc(out)
        return out

class SpatioTemporalAttention(nn.Module):
    def __init__(
        self,
        input_shape,
        feature_dimension=192,
        model_dimension=1024,
        num_heads = 8,
        dropout = 0.1):

        super().__init__()

        self.input_shape = input_shape
        self.feature_dimension = feature_dimension # 8*8*3 = 192
        self.model_dimension = model_dimension # 1024
        self.num_heads = num_heads # 8
        self.head_dim = model_dimension // num_heads #128
        self.dropout = dropout

        self.in_fc = nn.Sequential(
            nn.Linear(self.model_dimension, self.model_dimension),
            nn.LayerNorm(self.model_dimension)
        )

        self.q = nn.Linear(self.feature_dimension, self.model_dimension, bias = False)
        self.k = nn.Linear(self.feature_dimension, self.model_dimension, bias = False)
        self.v = nn.Linear(self.feature_dimension, self.model_dimension, bias = False)

        self.out_fc = nn.Sequential(
            nn.LayerNorm(self.model_dimension),
            nn.Linear(self.model_dimension, self.feature_dimension),
        )

    def forward(self, x, att_type):
        n, f, s, d = self.input_shape
        n = x.shape[0]
        h, hd = self.num_heads, self.head_dim
        nh = n*h

        x = self.in_fc(x)
        q, k, v = self.q(x), self.k(x), self.v(x)
        q, k, v = map(lambda y: y.reshape(n, -1, h, hd).permute(0, 2, 1, 3).reshape(nh, -1, hd), (q, k, v))

        (feat_q, q), (feat_k, k), (feat_v, v) = map(lambda y: (y[:, :1], y[:, 1:]), (q, k, v))

        token_att = attention(feat_q, k, v, hd)

        if att_type == "time":
            # time att, [NH x SF x D] -> [NHS x F x D]
            q, k, v = map(lambda y: y.reshape(-1, f, s, hd).permute(0, 2, 1, 3).reshape(-1, f, hd), (q, k, v))
            feat_k, feat_v = map(lambda y: y.repeat(s, 1, 1), (feat_k, feat_v))

        else:
            # space att, [NH x SF x D] -> [NHF x S x D]
            q, k, v = map(lambda y: y.reshape(-1, f, s, hd).reshape(-1, s, hd), (q, k, v))
            feat_k, feat_v = map(lambda y: y.repeat(f, 1, 1), (feat_k, feat_v))

        k, v = torch.cat((feat_k, k), dim=1), torch.cat((feat_v, v), dim=1)

        seq_att = attention(q, k, v, hd)

        if att_type == "time":
            # time att, [NHS x F x D] ->  [NH x SF x D]
            seq_att = seq_att.reshape(-1, s, f, hd).permute(0, 2, 1, 3).reshape(nh, -1, hd)

        else:
            # space att, [NHF x S x D] -> [N x SF x D]
            seq_att = seq_att.reshape(-1, f, s, hd).reshape(nh, -1, hd)
        
        out = torch.cat((feat_q, seq_att), dim=1)

        out = out.reshape(n, h, -1, hd).permute(0, 2, 1, 3).reshape(n, -1, h*hd)

        out = self.out_fc(out)
        
        return out
       
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
                dropout=0.1,):

        super(VisionEncoder, self).__init__()

        # Based on TimeSformer Divided Space-Time Attention from Facebook AI

        assert frame_size == patch_size * num_patches, f"frame_size {frame_size} does not match patch_size {patch_size} and num_patches {num_patches}"

        self.model_dimension = model_dimension
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.max_seqlen = max_seqlen
        self.num_channels = 3
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.feature_dimension = self.num_channels * self.patch_size**2
        self.input_shape = (self.batch_size, self.num_frames, self.num_patches**2, self.feature_dimension)

        self.feat_token = nn.Parameter(torch.randn(1, self.feature_dimension))

        self.in_fc = nn.Linear(self.feature_dimension, self.model_dimension)

        self.timesformer = nn.ModuleList([])
        for _ in range(num_layers):
            self.timesformer.append(nn.ModuleList([
                SpatioTemporalAttention(input_shape=self.input_shape,
                                        feature_dimension=self.model_dimension,
                                        model_dimension=self.model_dimension,
                                        num_heads=self.num_heads,
                                        dropout=self.dropout),
                SpatioTemporalAttention(input_shape=self.input_shape,
                                        feature_dimension=self.model_dimension,
                                        model_dimension=self.model_dimension,
                                        num_heads=self.num_heads,
                                        dropout=self.dropout),
                 nn.Sequential(
                            nn.Linear(self.model_dimension, self.model_dimension),
                            nn.GELU(),
                            nn.Dropout(self.dropout),
                            nn.Linear(self.model_dimension, self.model_dimension),)                   
            ]))

        self.out_fc = nn.Sequential(
            nn.LayerNorm(self.model_dimension),
            nn.Linear(self.model_dimension, self.model_dimension)
        )

    def forward(self, x):

        # Input: (N, F, H, W, C]

        n, f, h, w, c = x.shape
        p, ps = self.num_patches, self.patch_size
        s, d = (h//ps) * (w//ps), c * ps**2

        shape = (n, f, s, d)

        
        # [N, F, P, P, PS, PS, C]
        x = x.unfold(2, ps, ps).unfold(3, ps, ps)

        # [N x (SF) x D]
        x = x.reshape(-1, s*f, d)
      
        feat_token = self.feat_token.repeat(x.shape[0], 1).unsqueeze(1)
    
        # [N x (SF+1) x D]
        x = torch.cat((x, feat_token,), dim=1)

        # Feature Dim to Model Dim
        x = self.in_fc(x)

        x = position_embed(x)

        # Divided Time + Space Attention from TimeSformer
        for (time_att, space_att, fc) in self.timesformer:
            x = x + time_att(x, att_type="time")
            x = x + space_att(x, att_type="space")
            x = x + fc(x)

        out = x[:, 0]

        out = self.out_fc(out)

        return out


class LAVA(nn.Module):

    def __init__(self, 
                model_dimension=1024, 
                feat_dimension=512,
                a_seqlen=64,
                v_seqlen=256,
                t_seqlen=128,
                batch_size=20, 
                learning_rate=3e-4,
                num_heads=8, 
                num_layers=8,
                dropout=0.1,):

        super(LAVA, self).__init__()

        self.model_dimension = model_dimension
        self.feature_dimension = feat_dimension
        self.batch_size = batch_size
        self.a_seqlen = a_seqlen
        self.v_seqlen = v_seqlen
        self.t_seqlen = t_seqlen
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.cos_sim = nn.CosineSimilarity()

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

        self.t_encoder = TextEncoder(
                                model_dimension=self.model_dimension,
                                batch_size=self.batch_size,
                                num_heads=self.num_heads,
                                num_layers=self.num_layers,
                                max_seqlen=self.t_seqlen,)

        self.a_mlp = nn.Sequential(
            nn.Linear(self.model_dimension, self.model_dimension),
            nn.BatchNorm1d(self.model_dimension),
            nn.GELU(),
            nn.Linear(self.model_dimension, self.model_dimension),
        )
        
        self.v_mlp = nn.Sequential(
            nn.Linear(self.model_dimension, self.model_dimension),
            nn.BatchNorm1d(self.model_dimension),
            nn.GELU(),
            nn.Linear(self.model_dimension, self.model_dimension),
        )

        self.t_mlp = nn.Sequential(
            nn.Linear(self.model_dimension, self.model_dimension),
            nn.BatchNorm1d(self.model_dimension),
            nn.GELU(),
            nn.Linear(self.model_dimension, self.model_dimension),
        )

    def encode_audio(self, x):
        x = self.a_encoder(x)
        x = self.a_mlp(x)
        return x

    def encode_video(self, x):
        x = self.v_encoder(x)
        x = self.v_mlp(x)
        return x

    def encode_text(self, x):
        x = self.t_encoder(x)
        x = self.t_mlp(x)
        return x

    def forward(self, a, v, t):
        a = self.encode_audio(a)
        v = self.encode_video(v)
        t = self.encode_text(t)

        a = nn.functional.normalize(a, p=2, dim=-1)
        v = nn.functional.normalize(v, p=2, dim=-1)
        t = nn.functional.normalize(t, p=2, dim=-1)

        return a, v, t
    
    def loss(self, a, v, t):
        av_loss = 0.5*(nce_loss(a, v) + nce_loss(v, a))
        at_loss = 0.5*(nce_loss(a, t) + nce_loss(t, a))
        vt_loss = 0.5*(nce_loss(v, t) + nce_loss(t, v))

        avt_loss = centroid_loss(a, v, t)

        av_cos_sim = self.cos_sim(a, v).mean()
        at_cos_sim = self.cos_sim(a, t).mean()
        vt_cos_sim = self.cos_sim(v, t).mean()

        av_cos_sim_neg = torch.triu(torch.mm(a, v.t()), diagonal=1).mean()
        at_cos_sim_neg = torch.triu(torch.mm(a, t.t()), diagonal=1).mean()
        vt_cos_sim_neg = torch.triu(torch.mm(v, t.t()), diagonal=1).mean()

        a_cos_sim = torch.triu(torch.mm(a, a.t()), diagonal=1).mean()
        v_cos_sim = torch.triu(torch.mm(v, v.t()), diagonal=1).mean()
        t_cos_sim = torch.triu(torch.mm(v, t.t()), diagonal=1).mean()

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
                model_path=None):

        super(LinearClassifierAVT, self).__init__()

        self.model = model
        self.data = data
        self.num_classes = num_classes
        self.feature_dimension = feature_dimension
        self.model_dimension = model_dimension
        self.num_modalities = num_modalities
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model = model(batch_size=batch_size,
                        model_dimension=feature_dimension,)

        self.model.load_state_dict(torch.load(model_path)['state_dict'], strict=True)

        self.model.encoder.eval()

        self.a_encoder = self.model.encoder.encode_audio
        self.v_encoder = self.model.encoder.encode_video
        self.t_encoder = self.model.encoder.encode_text

        self.fc1 = torch.nn.Linear(self.num_modalities * self.model_dimension, self.num_classes)
        # self.fc1 = torch.nn.Linear(self.model_dimension, self.num_classes)

    
    def forward(self, a, v, t):
        with torch.no_grad():
            a = self.a_encoder(a)
            v = self.v_encoder(v)
            t = self.t_encoder(t)

            a = nn.functional.normalize(a, p=2, dim=-1)
            v = nn.functional.normalize(v, p=2, dim=-1)
            t = nn.functional.normalize(t, p=2, dim=-1)

            # representation = torch.cat((a,v,t), dim=-1)
            representation = torch.cat((a,v), dim=-1)
            similarity = torch.mm(v, t.t()).detach()

        pred = self.fc1(representation)
        return pred, similarity
