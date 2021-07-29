import torch
import torchaudio
import torchvision
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet
import pytorch_warmup as warmup

import numpy as np
import pandas as pd 
import warnings
import glob
from tqdm import tqdm
import pickle
from collections import Counter
import copy
import os

from aai.experimental.sgurram.lava.src.encoder import LAVA, LinearClassifierAVT
from aai.experimental.sgurram.lava.src.data import*
from aai.experimental.sgurram.lava.src.metrics import compute_accuracy
from aai.experimental.sgurram.lava.src.utils import visualize_batch, visualize_batch_downstream, pad_batch

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

            # a, v, t, urls, t_mask = batch
            a, v, t, i, urls, t_mask = batch
            if a.shape[0] < self.batch_size:
                a = pad_batch(a, self.batch_size)
            if v.shape[0] < self.batch_size:
                v = pad_batch(v, self.batch_size)
            if t.shape[0] < self.batch_size:
                t = pad_batch(t, self.batch_size)
                t_mask = pad_batch(t_mask, self.batch_size)
            if i.shape[0] < self.batch_size:
                i = pad_batch(i, self.batch_size)

            a, v, t, i = self.encoder(a, v, t, i)
            loss, metrics = self.encoder.loss(a, v, t, i, t_mask.to(dtype=torch.float32))

            # a, v, t = self.encoder(a, v, t)
            # loss, metrics = self.encoder.loss(a, v, t, t_mask.to(dtype=torch.float32))
            
        for k in metrics:
            if 'matrix' not in k:
                prog_bar = 'av_top1' in k
                self.log('train/{}'.format(k), metrics[k], prog_bar=prog_bar)

        self.log('lr', self.loggable_lr)

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

            # a, v, t, urls, t_mask = batch
            a, v, t, i, urls, t_mask = batch
            if a.shape[0] < self.batch_size:
                a = pad_batch(a, self.batch_size)
            if v.shape[0] < self.batch_size:
                v = pad_batch(v, self.batch_size)
            if t.shape[0] < self.batch_size:
                t = pad_batch(t, self.batch_size)
                t_mask = pad_batch(t_mask, self.batch_size)
            if i.shape[0] < self.batch_size:
                i = pad_batch(i, self.batch_size)

            a, v, t, i = self.encoder(a, v, t, i)
            loss, metrics = self.encoder.loss(a, v, t, i, t_mask.to(dtype=torch.float32))

            # a, v, t = self.encoder(a, v, t)
            # loss, metrics = self.encoder.loss(a, v, t, t_mask.to(dtype=torch.float32))

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
        dataset = LAVAData(prefix='train', pretrained_text=self.pretrained_text)
        return torch.utils.data.DataLoader(
                                    dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=8)

    def val_dataloader(self):
        dataset = LAVAData(prefix='val', pretrained_text=self.pretrained_text)
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

class EvalLightning(pl.LightningModule):

    def __init__(self,
                classifier=LinearClassifierAVT,
                data=K600Dataset, # Kinetics700Data # UCF101Dataset # K600Dataset
                num_classes=600,
                feature_dimension=512,
                model_dimension=1024,
                num_modalities=3,
                batch_size=5,
                learning_rate=1e-3,
                model_path=None,
                model=None,
                pretrained_text=True):

        super().__init__()

        self.data = data
        self.num_classes = num_classes
        self.model_dimension = model_dimension
        self.feature_dimension = feature_dimension
        self.num_modalities = num_modalities
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.model = model
        self.pretrained_text = pretrained_text

        self.classifier = classifier(data=self.data,
                            num_classes=self.num_classes,
                            feature_dimension=self.feature_dimension,
                            model_dimension=self.model_dimension,
                            num_modalities=self.num_modalities,
                            batch_size=self.batch_size,
                            learning_rate=self.learning_rate,
                            model_path=self.model_path,
                            model=model,
                            pretrained_text=pretrained_text)

        self.loss = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        a, v, t, label, urls = batch
        
        if a.shape[0] < self.batch_size:
            a = pad_batch(a, self.batch_size)
            v = pad_batch(v, self.batch_size)
            t = pad_batch(t, self.batch_size)
            label = pad_batch(label, self.batch_size)


        logits, similarity = self.classifier(a, v, t)
        
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


        if similarity is not None and batch_idx % 100 == 0:
            visualize_batch_downstream(similarity, prefix="train", dataset="k700")

        return {'loss': loss,
                'logs': logs}

    def validation_step(self, batch, batch_idx):
        a, v, t, label, urls = batch

        if a.shape[0] < self.batch_size:
            a = pad_batch(a, self.batch_size)
            v = pad_batch(v, self.batch_size)
            t = pad_batch(t, self.batch_size)
            label = pad_batch(label, self.batch_size)

        logits, similarity = self.classifier(a, v, t)

        loss = self.loss(logits, label)
        top_1_accuracy = compute_accuracy(logits, label, top_k=1)
        top_5_accuracy = compute_accuracy(logits, label, top_k=5)

        logs = {
            'val_loss': loss,
            'val_top_1': top_1_accuracy,
            'val_top_5': top_5_accuracy}

        if similarity is not None and batch_idx % 100 == 0:
            visualize_batch_downstream(similarity, prefix="val", dataset="k700")

        for k in logs:
            prog_bar = True if "top" in k else False
            self.log('val/{}'.format(k), logs[k], prog_bar=prog_bar)

        return logs

    def test_step(self, batch, batch_idx):
        a, v, t, label = batch
        logits, similarity = self.classifier(a, v, t)

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
        dataset = self.data(prefix='train')
        return torch.utils.data.DataLoader(
                                    dataset,
                                    batch_size=self.classifier.batch_size,
                                    shuffle=True,
                                    num_workers=8)

    def val_dataloader(self):
          dataset = self.data(prefix='val')
          return torch.utils.data.DataLoader(
                                  dataset,
                                  batch_size=self.classifier.batch_size,
                                  shuffle=False,
                                  num_workers=8)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
