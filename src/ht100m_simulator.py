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
from aai.experimental.sgurram.lava.src.utils import visualize_batch_downstream, pad_batch
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

        self.token_embedding = nn.Embedding(int(2e4), model_dimension)

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

        x = self.token_embedding(x)

        feat_token = self.feat_token.repeat(x.shape[0], 1).unsqueeze(1)
        
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
        self.token_embedding = nn.Embedding(int(2e4), model_dimension)
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
                space_patch_size=14,
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

        self.token_embedding = nn.Embedding(int(2e4), model_dimension)

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
        n, t, h, w = x.shape 

        f, d = self.feature_dimension, self.model_dimension
        sp, tp = self.space_patch_size, self.time_patch_size

        x = self.token_embedding(x.view(n, -1))

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

    def forward(self, a, v, t):
        a = self.encode_audio(a)
        v = self.encode_video(v)
        t = self.encode_text(t)

        a = nn.functional.normalize(a, p=2, dim=-1)
        v = nn.functional.normalize(v, p=2, dim=-1)
        t = nn.functional.normalize(t, p=2, dim=-1)

        return a, v, t

    def norm(self, x):
        return nn.functional.normalize(x, p=2, dim=-1)

    def loss(self, a, v, t):

        a = self.norm(self.a_proj(a))
        v = self.norm(self.v_proj(v))
        t = self.norm(self.t_proj(t))

        # approximate centroid vector
        centroid = (a + v + t )/3
        centroid = self.norm(centroid)

        avt_loss = nce_loss(a, centroid, temp=self.temperature)
        avt_loss += nce_loss(v, centroid, temp=self.temperature)
        avt_loss += nce_loss(t, centroid, temp=self.temperature)

        av_loss = nce_loss(a, v, temp=self.temperature)
        vt_loss = nce_loss(v, t, temp=self.temperature)

        loss = avt_loss + av_loss   + vt_loss

        av = a @ v.T
        vt = v @ t.T

        labels = torch.arange(self.batch_size).to(device=v.device)
        av_top1 = 0.5 * (compute_accuracy(av, labels , top_k=1) +  compute_accuracy(av.T, labels , top_k=1))
        vt_top1 = 0.5 * (compute_accuracy(vt, labels , top_k=1) +  compute_accuracy(vt.T, labels , top_k=1))

        metrics = {
            'loss': loss.item(),
            'av_loss': av_loss.item(),
            'vt_loss': vt_loss.item(),
            'av_top1': av_top1.item(),
            'vt_top1': vt_top1.item(),
            'temp': self.temperature,
            'vt_matrix': vt.T.detach().to(dtype=torch.float32),
        }
        
        return loss, metrics

class LAVALightning(pl.LightningModule):

    def __init__(self,
        model_dimension=1024, 
        feature_dimension=512,
        seqlen=256,
        batch_size=12, 
        num_heads=8, 
        num_layers=8,
        learning_rate=2e-5,
        optimizer = 'adamW',
        scheduler = 'cosine',
        max_lr = 2e-5,
        min_lr = 4e-6,
        warmup_mode = None,     
        warmup_steps = 4000,
        cooldown_steps = 2000,
        warm_gamma = 1.147,
        cool_gamma = 0.9977,
        cosine_steps_period=4000,
        dropout=0.0,
        mode='avt',
        multiple_video=False,
        pretrained_text=False,):

        super().__init__()

        self.model_dimension = model_dimension
        self.feature_dimension = feature_dimension
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_mode = warmup_mode   
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.warm_gamma = warm_gamma
        self.cool_gamma = cool_gamma
        self.cosine_steps_period = int(cosine_steps_period / (2*np.pi))
        self.loggable_lr = learning_rate
        self.multiple_video = multiple_video
        self.pretrained_text = pretrained_text

        self.encoder = LAVA(dropout=self.dropout,
                        model_dimension=self.model_dimension, 
                        feature_dimension=self.feature_dimension,
                        batch_size=self.batch_size, 
                        learning_rate=self.learning_rate,
                        num_heads=self.num_heads, 
                        num_layers=self.num_layers,
                        pretrained_text=pretrained_text)
  
    def training_step(self, batch, batch_idx):
        a, v, t = batch
        if a.shape[0] < self.batch_size:
            a = pad_batch(a, self.batch_size)
        if v.shape[0] < self.batch_size:
            v = pad_batch(v, self.batch_size)
        if t.shape[0] < self.batch_size:
            t = pad_batch(t, self.batch_size)
            t_mask = pad_batch(t_mask, self.batch_size)

        a, v, t = self.encoder(a, v, t)
        loss, metrics = self.encoder.loss(a, v, t)
            
        for k in metrics:
            if 'matrix' not in k:
                prog_bar = 'av_top1' in k
                self.log('train/{}'.format(k), metrics[k], prog_bar=prog_bar)

        self.log('lr', self.loggable_lr)

        torch.cuda.empty_cache()

        return {'loss': loss,
                'logs': metrics}

    def validation_step(self, batch, batch_idx):
        a, v, t = batch
        if a.shape[0] < self.batch_size:
            a = pad_batch(a, self.batch_size)
        if v.shape[0] < self.batch_size:
            v = pad_batch(v, self.batch_size)
        if t.shape[0] < self.batch_size:
            t = pad_batch(t, self.batch_size)
            t_mask = pad_batch(t_mask, self.batch_size)

        a, v, t = self.encoder(a, v, t)
        loss, metrics = self.encoder.loss(a, v, t)

        for k in metrics:
            if 'matrix' not in k:
                prog_bar = 'av_top1' in k
                self.log('val/{}'.format(k), metrics[k], prog_bar=prog_bar)

        torch.cuda.empty_cache()

        return {'val_total_loss': loss}        
    
    def training_epoch_end(self, outputs):
        avg_total_loss = torch.stack([m['loss'] for m in outputs]).mean()

        self.log('train_avg_loss', avg_total_loss)

    def validation_epoch_end(self, outputs):
        avg_total_loss = torch.stack([m['val_total_loss'] for m in outputs]).mean()

        self.log('val_avg_loss', avg_total_loss)

    def train_dataloader(self):
        dataset = HT100MData(prefix='train')
        return torch.utils.data.DataLoader(
                                    dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=8)

    def val_dataloader(self):
        dataset = HT100MData(prefix='val')
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
    mode = 1

    hparams = {'gpus':[0,1], 
            'max_epochs': 100, 
            'auto_lr_find': False,
            'learning_rate': 6e-5,
            'max_lr': 6e-5,
            'min_lr': 6e-6,
            'video augmentation': ['random crop, temporal sampling', 'hflip', 'color jitter'],
            'auto_scale_batch_size': None,
            'batch_size': 16,
            'accumulate_grad_batches': 2,
            'model_dimension': 128,
            'feature_dimension': 1024,
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
            'default_root_dir': '/home/sgurram/Desktop/video_cvrl',}

    model = LAVALightning(
            model_dimension=hparams['model_dimension'], 
            feature_dimension=hparams['feature_dimension'],
            batch_size=hparams['batch_size'], 
            learning_rate=hparams['learning_rate'],
            min_lr=hparams['min_lr'],
            max_lr=hparams['max_lr'],
            optimizer=hparams['optimizer'],
            scheduler=hparams['scheduler'],
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
            # resume_from_checkpoint=hparams['model_path'],
            precision=hparams['precision'],
            distributed_backend=hparams['distributed_backend'],
            # limit_train_batches=2,
            # limit_val_batches=2,
            ) 
        

    trainer.fit(model)