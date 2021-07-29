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

from aai.experimental.sgurram.lava.src.metrics import nce_loss, centroid_loss, instance_loss, compute_accuracy
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

        # x = x[:, -1, :].squeeze() if self.pool == "token" else x.mean(dim=1).squeeze() # [N x D]

        # x = self.fc(x)

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

        # x = x[:, -1, :].squeeze() if self.pool == "token" else x.mean(dim=1).squeeze() # [N x D]

        # x = self.fc(x)
        
        return x
       
class VisionEncoder(nn.Module):
    def __init__(self, 
                model_dimension=1024,
                batch_size=8,
                frame_size=224,
                space_patch_size=28,
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

        # x = x[:, -1, :].squeeze() if self.pool == "token" else x.mean(dim=1).squeeze() # [N x D]
        # x = self.fc(x)

        return x


class Volqano(nn.Module):
    def __init__(self, 
                num_embeddings=512,
                model_dimension=1024, 
                feature_dimension=512,
                a_seqlen=64,
                v_seqlen=256,
                t_seqlen=128,
                seqlen=32,
                batch_size=20, 
                learning_rate=3e-4,
                num_heads=8, 
                num_layers=8,
                dropout=0.0,
                pretrained_text=False):

        super(Volqano, self).__init__()

        self.num_embeddings = num_embeddings
        self.model_dimension = model_dimension
        self.feature_dimension = feature_dimension
        self.batch_size = batch_size
        self.a_seqlen = a_seqlen
        self.v_seqlen = v_seqlen
        self.t_seqlen = t_seqlen
        self.seqlen = seqlen
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.pretrained_text = pretrained_text

        self.cos_sim = nn.CosineSimilarity()

        # self.codebook = nn.Embedding(self.num_embeddings, self.model_dimension)

        self.codebook = nn.Parameter(torch.randn(self.num_embeddings, self.model_dimension))

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
        
        self.v_proj = nn.Linear(self.model_dimension, self.model_dimension)

    def encode_audio(self, x):
        x = self.a_encoder(x)
        x = x[:, :self.seqlen]
        x = self.a_proj(x)
        return x

    def encode_video(self, x):
        x = self.v_encoder(x)
        x = x[:, :self.seqlen]
        x = self.v_proj(x)
        return x

    def encode_text(self, x):
        x = self.t_encoder(x)[:, self.seqlen:]
        x = x[:, :self.seqlen]
        x = self.t_proj(x)
        return x

    def forward(self, a, v, t):
        a = self.encode_audio(a)
        v = self.encode_video(v)
        t = self.encode_text(t)

        a = nn.functional.normalize(a, p=2, dim=-1)
        v = nn.functional.normalize(v, p=2, dim=-1)
        t = nn.functional.normalize(t, p=2, dim=-1)

        return a, v, t

    def loss(self, a, v, t, t_mask):
        # zero out t vectors from default string

        codes = torch.nn.functional.normalize(self.codebook, p=2, dim=-1)
        softmax = torch.nn.Softmax(dim=-1)
        #cross entropy
        ce_loss = torch.nn.CrossEntropyLoss()
        temp = 1e+3

        a_logits = a @ codes.T
        v_logits = v @ codes.T
        t_logits = t @ codes.T

        a_onehot = softmax(a_logits*temp).view(-1, self.num_embeddings)
        v_onehot = softmax(v_logits*temp).view(-1, self.num_embeddings)
        t_onehot = softmax(t_logits*temp).view(-1, self.num_embeddings)

        #contastive
        # a_latent = (a_onehot.view(-1, self.num_embeddings) * codes.T).sum(-1)
        # v_latent = (v_onehot.view(-1, self.num_embeddings) * codes.T).sum(-1)
        # t_latent = (t_onehot.view(-1, self.num_embeddings) * codes.T).sum(-1)

        # av_loss = 0.5 * (nce_loss(a, t_latent, temp=self.temperature) + nce_loss(v, t_latent, temp=self.temperature))
        # at_loss = 0.5 * (nce_loss(a, v_latent, temp=self.temperature) + nce_loss(t, v_latent, temp=self.temperature))
        # vt_loss = 0.5 * (nce_loss(v, a_latent, temp=self.temperature) + nce_loss(t, a_latent, temp=self.temperature))

        a_logits = a_logits.view(-1, self.num_embeddings)
        v_logits = v_logits.view(-1, self.num_embeddings)
        t_logits = t_logits.view(-1, self.num_embeddings)

        a_label =  torch.argmax(a_onehot.view(-1, self.num_embeddings), dim=-1)
        v_label =  torch.argmax(v_onehot.view(-1, self.num_embeddings), dim=-1)
        t_label =  torch.argmax(t_onehot.view(-1, self.num_embeddings), dim=-1)

        print(v_label)

        av_loss = 0.5 * (ce_loss(a_logits, t_label) + ce_loss(v_logits, t_label))
        at_loss = 0.5 * (ce_loss(a_logits, v_label) + ce_loss(t_logits, v_label))
        vt_loss = 0.5 * (ce_loss(v_logits, a_label) + ce_loss(t_logits, a_label))

        av_top1 = 0.5 * (compute_accuracy(a_onehot, v_label , top_k=1) +  compute_accuracy(v_onehot, a_label, top_k=1))
        vt_top1 = 0.5 * (compute_accuracy(v_onehot, t_label , top_k=1) +  compute_accuracy(t_onehot, v_label, top_k=1))

        a_prob = softmax(a_logits.view(self.batch_size, self.seqlen, -1).transpose(0,1).sum(dim=1))
        v_prob = softmax(v_logits.view(self.batch_size, self.seqlen, -1).transpose(0,1).sum(dim=1))
        t_prob = softmax(t_logits.view(self.batch_size, self.seqlen, -1).transpose(0,1).sum(dim=1))


        a_entropy = (-a_prob * torch.log(a_prob)).sum(dim=-1).mean()
        v_entropy = (-v_prob * torch.log(v_prob)).sum(dim=-1).mean()
        t_entropy = (-t_prob * torch.log(t_prob)).sum(dim=-1).mean()

        # loss = 10*(av_loss + at_loss + vt_loss) - a_entropy - v_entropy - t_entropy
        loss = av_loss + at_loss + vt_loss + a_entropy + v_entropy + t_entropy
        # loss = av_loss + at_loss + vt_loss

        # metrics = {
        #     'loss': loss.item(),
        #     'av_loss': av_loss.item(),
        #     'at_loss': at_loss.item(),
        #     'vt_loss': vt_loss.item(),
        #     'a_entropy': a_entropy.item(),
        #     'v_entropy': v_entropy.item(),
        #     't_entropy': t_entropy.item(),
        #     'temp': self.temperature,
        #     'vt_matrix': (v[:, 0] @ t[:, 0].T).detach()
        # }

        metrics = {
            'loss': loss.item(),
            'av_top1': av_top1.item(),
            'vt_top': vt_top1.item(),
            'av_loss': av_loss.item(),
            'at_loss': at_loss.item(),
            'vt_loss': vt_loss.item(),
            'a_entropy': a_entropy.item(),
            'v_entropy': v_entropy.item(),
            't_entropy': t_entropy.item(),
            'temp': self.temperature,
            'vt_matrix': (v[:, 0] @ t[:, 0].T).detach()
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
        self.num_classes = num_classes
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

        self.model.encoder.eval()
   
        self.fc = torch.nn.Linear(self.num_modalities * self.model_dimension, self.num_classes)

    
    def forward(self, a, v, t):
        with torch.no_grad():
            v = self.model.encoder.v_encoder(v)
            t = self.model.encoder.t_encoder(t)

        if self.pretrained_text:
            t = t.squeeze()
          
        similarity = (t @ v.T).detach()
        pred = self.fc(v)
        return pred, similarity
class VolqanoLightning(pl.LightningModule):
    def __init__(self,
        model_dimension=64, 
        batch_size=12, 
        learning_rate=2e-5,
        optimizer = 'adamW',
        scheduler = 'cosine',
        max_lr = 2e-5,
        min_lr = 4e-6,
    ):

        super().__init__()

        self.model_dimension = model_dimension
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.encoder = Volqano(
                        model_dimension=self.model_dimension, 
                        batch_size=self.batch_size, 
                        learning_rate=self.learning_rate,)

    def training_step(self, batch, batch_idx):
        a, v, t, urls, t_mask = batch
        # a, v, t, i, urls, t_mask = batch
        if a.shape[0] < self.batch_size:
            a = pad_batch(a, self.batch_size)
        if v.shape[0] < self.batch_size:
            v = pad_batch(v, self.batch_size)
        if t.shape[0] < self.batch_size:
            t = pad_batch(t, self.batch_size)
            t_mask = pad_batch(t_mask, self.batch_size)
        # if i.shape[0] < self.batch_size:
        #     i = pad_batch(i, self.batch_size)

        # a, v, t, i = self.encoder(a, v, t, i)
        # loss, metrics = self.encoder.loss(a, v, t, i, t_mask.to(dtype=torch.float32))

        a, v, t = self.encoder(a, v, t)
        loss, metrics = self.encoder.loss(a, v, t, t_mask.to(dtype=torch.float32))
            
        for k in metrics:
            if 'matrix' not in k:
                prog_bar = 'av_top1' in k
                self.log('train/{}'.format(k), metrics[k], prog_bar=prog_bar)

        torch.cuda.empty_cache()

        try:
            if batch_idx % 100 == 0:
                q = 'st' if self.pretrained_text else 'c'
                q += 'vq'
                if 'vt_matrix' in metrics:
                    visualize_batch(urls, metrics['vt_matrix'], prefix="train", qualifier=q, mode='vt')
                else: 
                    visualize_batch(urls, metrics['av_matrix'], prefix="train", qualifier=q, mode='av')
        except: 
            a = batch_idx % 100

        return {'loss': loss,
                'logs': metrics}

    def validation_step(self, batch, batch_idx):
        a, v, t, urls, t_mask = batch
        # a, v, t, i, urls, t_mask = batch
        if a.shape[0] < self.batch_size:
            a = pad_batch(a, self.batch_size)
        if v.shape[0] < self.batch_size:
            v = pad_batch(v, self.batch_size)
        if t.shape[0] < self.batch_size:
            t = pad_batch(t, self.batch_size)
            t_mask = pad_batch(t_mask, self.batch_size)
        # if i.shape[0] < self.batch_size:
        #     i = pad_batch(i, self.batch_size)

        # a, v, t, i = self.encoder(a, v, t, i)
        # loss, metrics = self.encoder.loss(a, v, t, i, t_mask.to(dtype=torch.float32))

        a, v, t = self.encoder(a, v, t)
        loss, metrics = self.encoder.loss(a, v, t, t_mask.to(dtype=torch.float32))

        for k in metrics:
            if 'matrix' not in k:
                prog_bar = 'av_top1' in k
                self.log('val/{}'.format(k), metrics[k], prog_bar=prog_bar)

        torch.cuda.empty_cache()

        try:
            if batch_idx % 100 == 0:
                q = 'st' if self.pretrained_text else 'c'
                q += 'vq'
                if 'vt_matrix' in metrics:
                    visualize_batch(urls, metrics['vt_matrix'], prefix="val", qualifier=q, mode='vt')
                else: 
                    visualize_batch(urls, metrics['av_matrix'], prefix="val", qualifier=q, mode='av')
        except:
                a = batch_idx % 100

        return {'val_total_loss': loss}


    def test_step(self, batch, batch_idx):
        if self.multiple_video:
            a, v1, v2, t, urls, t_mask = batch
            if a.shape[0] < self.batch_size:
                a = pad_batch(a, self.batch_size)
            if v1.shape[0] < self.batch_size:
                    v1 = pad_batch(v1, self.batch_size)
            if v2.shape[0] < self.batch_size:
                    v2 = pad_batch(v2, self.batch_size)
            if t.shape[0] < self.batch_size:
                t = pad_batch(t, self.batch_size)
                t_mask = pad_batch(t_mask, self.batch_size)
            a, v1, v2, t = self.encoder( a, v1, v2, t)
            loss, metrics = self.encoder.loss( a, v1, v2, t, t_mask.to(dtype=torch.float32))
        else:

            a, v, t, urls, t_mask = batch
            if a.shape[0] < self.batch_size:
                a = pad_batch(a, self.batch_size)
            if v.shape[0] < self.batch_size:
                v = pad_batch(v, self.batch_size)
            if t.shape[0] < self.batch_size:
                t = pad_batch(t, self.batch_size)
                t_mask = pad_batch(t_mask, self.batch_size)

            a, v, t = self.encoder(a, v, t)
            loss, metrics = self.encoder.loss(a, v, t, t_mask.to(dtype=torch.float32))

        return {'test_total_loss': metrics['total_loss'],}
                
    
    def training_epoch_end(self, outputs):
        avg_total_loss = torch.stack([m['loss'] for m in outputs]).mean()

        self.log('train_avg_loss', avg_total_loss)

    def validation_epoch_end(self, outputs):
        avg_total_loss = torch.stack([m['val_total_loss'] for m in outputs]).mean()

        self.log('val_avg_loss', avg_total_loss)

    def train_dataloader(self):
        dataset = LAVAData(prefix='train')
        return torch.utils.data.DataLoader(
                                    dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=8)

    def val_dataloader(self):
        dataset = LAVAData(prefix='val')
        return torch.utils.data.DataLoader(
                                dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=8)


    def configure_optimizers(self):
        if self.optimizer=="adamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        else: 
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if self.scheduler=="cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=0)
        elif self.scheduler=="multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1, verbose=True)
        elif self.scheduler=="exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1, verbose=True)
        else:
            return optimizer
        lr_scheduler = {'scheduler': scheduler,
                        'name': 'lr_scheduler'}

        return [optimizer], [lr_scheduler]


if __name__ == "__main__":
    logging = False
    lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

    modes = ['train', 'eval']


    hparams = {'gpus':[0,1], 
            'max_epochs': 100, 
            'auto_lr_find': False,
            'learning_rate': 5e-6,
            'max_lr': 5e-6,
            'min_lr': 5e-6,
            'video augmentation': ['random crop, temporal sampling', 'hflip', 'color jitter'],
            'auto_scale_batch_size': None,
            'batch_size': 32,
            'accumulate_grad_batches': 1,
            'model_dimension': 1024,
            'model_path': "/home/sgurram/Desktop/video_cvrl/checkpoints/epoch=7-step=57279.ckpt",
            'overfit_batches': 0,
            'amp_backend': 'native',
            'amp_level': 'O2',
            'precision': 16,
            'log_gpu_memory': 'all',
            'optimizer': 'adamW',
            'scheduler': 'None',
            'cosine_steps_period': 4000,
            'warmup': None,
            'profiler': 'simple',
            'distributed_backend': 'ddp',
            'callbacks': [lr_monitor_callback] if logging else None,
            'default_root_dir': '/home/sgurram/Desktop/volqano',
            'gradient_clip_val': 0.005}

    model = VolqanoLightning(
            model_dimension=hparams['model_dimension'], 
            batch_size=hparams['batch_size'], 
            learning_rate=hparams['learning_rate'],
            min_lr=hparams['min_lr'],
            max_lr=hparams['max_lr'],
            optimizer=hparams['optimizer'],
            # scheduler=hparams['scheduler'],
        )
        

    # model.load_state_dict(torch.load(hparams['model_path'], map_location='cuda:1')['state_dict'], strict=True)

    if logging:
            wandb_logger = WandbLogger(name='run',project='lava')
            wandb_logger.log_hyperparams(hparams)
            wandb_logger.watch(model, 
                    log='gradients', 
                    log_freq=10)
    else:
            wandb_logger = None

    trainer = pl.Trainer(
            default_root_dir=hparams['default_root_dir'], 
            gpus=hparams['gpus'], 
            max_epochs=hparams['max_epochs'],
            auto_scale_batch_size=hparams['auto_scale_batch_size'],
            auto_lr_find=hparams['auto_lr_find'],
            accumulate_grad_batches=hparams['accumulate_grad_batches'],
            overfit_batches=hparams['overfit_batches'],
            logger=wandb_logger,
            profiler=hparams['profiler'],
            amp_backend=hparams['amp_backend'],
            amp_level=hparams['amp_level'],
            log_gpu_memory=hparams['log_gpu_memory'],
            callbacks=hparams['callbacks'],
            precision=hparams['precision'],
            distributed_backend=hparams['distributed_backend'],
            gradient_clip_val=hparams['gradient_clip_val']
            ) 
            

    trainer.fit(model)
    
if __name__ == "__main__":
    # ones = (torch.nn.parameter.Parameter(torch.ones(8, 4)) * torch.arange(4)).T
    # twos = (2*torch.nn.parameter.Parameter(torch.ones(8, 4)) * torch.arange(4)).T
    # latents = torch.stack([ones, twos]).permute(0, 2, 1)
    # data = 1 * torch.ones(3, 2, 1, 8)

    # data = torch.rand(64, 28, 512).unsqueeze(2)
    # latents = torch.rand(28, 32, 512).permute(0,2,1)

    # res = data @ latents
    # print(res.shape)

    a = torch.rand(2, 2, 8)
    b = torch.nn.Softmax(dim=-1)