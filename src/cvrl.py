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
from aai.experimental.sgurram.lava.src.encoder import*
from aai.experimental.sgurram.lava.src.data import*

class CVRL(nn.Module):
    
    def __init__(self, 
                model_dimension=128, 
                feature_dimension=1024,
                batch_size=20, 
                learning_rate=3e-4,
                num_heads=8, 
                num_layers=8,
                dropout=0.0,
            ):

        super(CVRL, self).__init__()

        self.model_dimension = model_dimension
        self.feature_dimension = feature_dimension
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate


        self.pairwise_temp = nn.Parameter(torch.tensor([14.285]), requires_grad=False)
        
        # self.v_encoder = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
        # self.v_encoder = torchvision.models.video.r3d_18(pretrained=False)

        self.v_encoder = VisionEncoder(
                model_dimension=1024,
                batch_size=8,
                frame_size=112,
                space_patch_size=14,
                time_patch_size=4,
                num_frames=16,
                num_patches=8,
                max_seqlen=256,
                num_channels=3,
                num_heads=8, 
                num_layers=8,
                dropout=0.0,
                pool="token",)

        
        self.v_proj = nn.Sequential(
            nn.Linear(self.feature_dimension, self.model_dimension),
            nn.BatchNorm1d(self.model_dimension),
            nn.GELU(),
            nn.Linear(self.model_dimension, self.model_dimension),
            nn.BatchNorm1d(self.model_dimension),
            nn.GELU(),
            nn.Linear(self.model_dimension, self.model_dimension),
        )
    def encode_video(self, x):
        x = x.permute(0, 1, 3, 4, 2)
        x = self.v_encoder(x)
        x = self.v_proj(x)
        return x

    def forward(self, v1, v2):
        v1 = self.encode_video(v1)
        v2 = self.encode_video(v2)
        v1 = nn.functional.normalize(v1, p=2, dim=-1)
        v2 = nn.functional.normalize(v2, p=2, dim=-1)
        return v1, v2
    
    def loss(self, v1, v2):
        pairwise_temp = self.pairwise_temp
        total_loss = nce_loss(v1, v2, temp=pairwise_temp)

        v = v1 @ v2.T
        v_cos_sim = torch.diag(v).mean()
        v_cos_sim_neg = torch.triu(v, diagonal=1).mean()


        metrics = {
                'loss': total_loss.item(),
                'cos_sim_v': v_cos_sim.item(),
                'cos_sim_v_neg': v_cos_sim_neg.item(),
                'v_matrix': v.T.detach().to(dtype=torch.float32),
        }
        return total_loss, metrics

class CVRLLightning(pl.LightningModule):

    def __init__(self,
        model_dimension=128, 
        feature_dimension=400,
        batch_size=12, 
        learning_rate=2e-5,
        optimizer = 'adamW',
        scheduler = 'cosine',
        max_lr = 2e-5,
        min_lr = 4e-6,
    ):

        super().__init__()

        self.model_dimension = model_dimension
        self.feature_dimension = feature_dimension
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.encoder = CVRL(
                        model_dimension=self.model_dimension, 
                        feature_dimension=self.feature_dimension,
                        batch_size=self.batch_size, 
                        learning_rate=self.learning_rate,)

    def training_step(self, batch, batch_idx):
        v1, v2 = batch
        if v1.shape[0] < self.batch_size:
            v1 = pad_batch(v1, self.batch_size)
        if v2.shape[0] < self.batch_size:
            v2 = pad_batch(v2, self.batch_size)

        v1, v2 = self.encoder(v1, v2)
        loss, metrics = self.encoder.loss(v1, v2,)

        for k in metrics:
            if 'matrix' not in k:
                prog_bar = True if "temp" in k else False
                self.log('train/{}'.format(k), metrics[k], prog_bar=prog_bar)

        torch.cuda.empty_cache()

        try:
            if batch_idx % 100 == 0:
                q = 'cvrl'
                if 'v_matrix' in metrics:
                    visualize_batch_downstream(metrics['v_matrix'], prefix="train", dataset=q)
        except: 
            q = 'cvrl'

        return {'loss': loss,
                'logs': metrics}

    def validation_step(self, batch, batch_idx):
        v1, v2 = batch
        if v1.shape[0] < self.batch_size:
            v1 = pad_batch(v1, self.batch_size)
        if v2.shape[0] < self.batch_size:
            v2 = pad_batch(v2, self.batch_size)

        v1, v2 = self.encoder(v1, v2)
        loss, metrics = self.encoder.loss(v1, v2,)

        for k in metrics:
            if 'matrix' not in k:
                prog_bar = True if "temp" in k else False
                self.log('train/{}'.format(k), metrics[k], prog_bar=prog_bar)

        torch.cuda.empty_cache()

        try:
            if batch_idx % 100 == 0:
                q = 'cvrl'
                if 'v_matrix' in metrics:
                    visualize_batch_downstream(metrics['v_matrix'], prefix="val", dataset=q)
        except: 
            q = 'cvrl'

        return {'val_total_loss': loss}
    
    def training_epoch_end(self, outputs):
        avg_total_loss = torch.stack([m['loss'] for m in outputs]).mean()

        self.log('train_avg_loss', avg_total_loss)

    def validation_epoch_end(self, outputs):
        avg_total_loss = torch.stack([m['val_total_loss'] for m in outputs]).mean()

        self.log('val_avg_loss', avg_total_loss)

    def train_dataloader(self):
        dataset = CVRLData(prefix='train')
        return torch.utils.data.DataLoader(
                                    dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=8)

    def val_dataloader(self):
        dataset = CVRLData(prefix='val')
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
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, eta_min=0)
        elif self.scheduler=="multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1, verbose=True)
        elif self.scheduler=="exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1, verbose=True)
        else:
            return optimizer
        lr_scheduler = {'scheduler': scheduler,
                        'name': 'lr_scheduler'}

        return [optimizer], [lr_scheduler]

class CVRLProbeLightning(pl.LightningModule):
    def __init__(self,
                data=Kinetics700Data,
                num_classes=700,
                feature_dimension=400,
                model_dimension=128,
                batch_size=32,
                learning_rate=1e-3,
                model_path="/home/sgurram/Desktop/video_cvrl/checkpoints/epoch=7-step=57279.ckpt",
                # model_path="/home/sgurram/Desktop/video_lava/checkpoints/epoch=11-step=42959.ckpt",
            ):

        super().__init__()

        self.data = data
        self.num_classes = num_classes
        self.model_dimension = model_dimension
        self.feature_dimension = feature_dimension
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_path = model_path

     
        self.model = CVRLLightning(
                model_dimension=model_dimension, 
                feature_dimension=feature_dimension,
                batch_size=batch_size,)

        # self.model.load_state_dict(torch.load(model_path, map_location='cuda:1')['state_dict'], strict=True)
        self.fc = nn.Linear(feature_dimension, num_classes)
        self.model.encoder.eval()
        self.loss = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        v, _, label = batch
        v = v.permute(0, 2, 1, 3, 4)

        if v.shape[0] < self.batch_size:
            v = pad_batch(v, self.batch_size)
            label = pad_batch(label, self.batch_size)

        with torch.no_grad():
            features = self.model.encoder.v_encoder(v)
        logits = self.fc(features)
        
        loss = self.loss(logits, label)
        top_1_accuracy = compute_accuracy(logits, label, top_k=1)
        top_5_accuracy = compute_accuracy(logits, label, top_k=5)

        logs = {
            'loss': loss,
            'train_top_1': top_1_accuracy,
            'train_top_5': top_5_accuracy}

        for k in logs:
            prog_bar = True if "top" in k else False
            self.log('train/{}'.format(k), logs[k], prog_bar=prog_bar)

        return {'loss': loss,
                'logs': logs}

    def validation_step(self, batch, batch_idx):
        v, v2, label = batch
        v = v.permute(0, 2, 1, 3, 4)
        v2 = v2.permute(0, 2, 1, 3, 4)

        if v.shape[0] < self.batch_size:
            v = pad_batch(v, self.batch_size)
            label = pad_batch(label, self.batch_size)


        # with torch.no_grad():
        features = self.model.encoder.v_encoder(v)   
        logits = self.fc(features)

        f1, f2 = self.model.encoder(v, v2)
        similarity = f1 @ f2.T
        visualize_batch_downstream(similarity, prefix="val", dataset="cvrl_ucf")


        loss = self.loss(logits, label)
        top_1_accuracy = compute_accuracy(logits, label, top_k=1)
        top_5_accuracy = compute_accuracy(logits, label, top_k=5)

        logs = {
            'val_loss': loss,
            'val_top_1': top_1_accuracy,
            'val_top_5': top_5_accuracy}

        for k in logs:
            prog_bar = True if "top" in k else False
            self.log('val/{}'.format(k), logs[k], prog_bar=prog_bar)

        return logs

    def test_step(self, batch, batch_idx):
        v, label = batch
        v = v.permute(0, 2, 1, 3, 4)

        if v.shape[0] < self.batch_size:
            v = pad_batch(v, self.batch_size)
            label = pad_batch(label, self.batch_size)


        features = self.model.encoder.v_encoder(v)
        logits = self.fc(features)

        loss = self.loss(logits, label)
        top_1_accuracy = compute_accuracy(logits, label, top_k=1)
        top_5_accuracy = compute_accuracy(logits, label, top_k=5)

        logs = {
            'test_loss': loss,
            'test_top_1': top_1_accuracy,
            'test_top_5': top_5_accuracy}
            
        for k in logs:
            self.log('test/{}'.format(k), logs[k], prog_bar=False)

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([m['logs']['loss'] for m in outputs]).mean()

        self.log('train_avg_loss', avg_loss)


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([m['val_loss'] for m in outputs]).mean()

        self.log('val_avg_loss', avg_loss)

    def train_dataloader(self):
        dataset = CVRLKinetics700Data(prefix='train')
        # dataset = UCF101Dataset(prefix='train')
        return torch.utils.data.DataLoader(
                                    dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=8)

    def val_dataloader(self):
        dataset = CVRLKinetics700Data(prefix='val')
        # dataset = UCF101Dataset(prefix='val')

        return torch.utils.data.DataLoader(
                                  dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=8)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer



if __name__ == "__main__":
    logging = False
    lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

    modes = ['train', 'eval']
    mode = 1

    if mode == 1:

        hparams = {'gpus':[0,1], 
                'max_epochs': 100, 
                'auto_lr_find': False,
                'learning_rate': 6e-5,
                'max_lr': 6e-5,
                'min_lr': 6e-6,
                'video augmentation': ['random crop, temporal sampling', 'hflip', 'color jitter'],
                'auto_scale_batch_size': None,
                'batch_size': 32,
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

        model = CVRLLightning(
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
    elif mode == 2:

        model = CVRLProbeLightning()

        trainer = pl.Trainer(
                # default_root_dir=hparams['default_root_dir'], 
                gpus=[1], 
                max_epochs=25,
                # auto_scale_batch_size=hparams['auto_scale_batch_size'],
                # auto_lr_find=hparams['auto_lr_find'],
                accumulate_grad_batches=2,
                overfit_batches=0,
                # logger=wandb_logger,
                # profiler=hparams['profiler'],
                amp_backend='native',
                amp_level='02',
                # log_gpu_memory=hparams['log_gpu_memory'],
                precision=16,
                # distributed_backend=ddp,
                # limit_train_batches=2,
                # limit_val_batches=2,
            )
        trainer.fit(model)
 
            
       