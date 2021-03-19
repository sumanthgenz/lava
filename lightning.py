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

from aai.experimental.sgurram.lava.src.encoder import*
from aai.experimental.sgurram.lava.src.data import LAVAData
from aai.experimental.sgurram.lava.src.metrics import compute_accuracy

torchaudio.set_audio_backend("sox_io") 
warnings.filterwarnings("ignore")

class LAVALightning(pl.LightningModule):

    def __init__(self,
        model_dimension=1024, 
        feat_dimension=512,
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
        dropout=0.1,):

        super().__init__()

        self.model_dimension = model_dimension
        self.feature_dimension = feat_dimension
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

        self.encoder = LAVA(dropout=self.dropout,
                        model_dimension=self.model_dimension, 
                        feat_dimension=self.feature_dimension,
                        seqlen=self.seqlen,
                        batch_size=self.batch_size, 
                        learning_rate=self.learning_rate,
                        num_heads=self.num_heads, 
                        num_layers=self.num_layers,)
  
    def training_step(self, batch, batch_idx):
        audio, video, text = batch
        audio_encoded, video_encoded, text_encoded = self.encoder(audio, video, text)
        metrics = self.encoder.loss(audio_encoded, video_encoded, text_encoded)
        loss = metrics['loss']

        for k in metrics:
            if k == 'loss':
                self.log('train/{}'.format(k), metrics[k].item(), prog_bar=False)
            else:
                self.log('train/{}'.format(k), metrics[k], prog_bar=False)

        self.log('lr', self.loggable_lr)

        torch.cuda.empty_cache()

        return {'loss': loss,
                'logs': metrics}

    def validation_step(self, batch, batch_idx):
        audio, video, text = batch
        audio_encoded, video_encoded, text_encoded = self.encoder(audio, video, text)
        metrics = self.encoder.loss(audio_encoded, video_encoded, text_encoded)
        loss = metrics['loss']

        for k in metrics:
            if k == 'loss':
                self.log('val/{}'.format(k), metrics[k].item(), prog_bar=False)
            else:
                self.log('val/{}'.format(k), metrics[k], prog_bar=False)

        torch.cuda.empty_cache()

        return {'val_total_loss': loss}


    def test_step(self, batch, batch_idx):
        audio, video, text = batch
        audio_encoded, video_encoded, text_encoded = self.encoder(audio, video, text)
        metrics = self.encoder.loss(audio_encoded, video_encoded, text_encoded)

        for k in metrics:
            self.log('val/{}'.format(k), metrics[k], prog_bar=False)
            
        return {'test_total_loss': metrics['total_loss'],}
                
    
    def training_epoch_end(self, outputs):
        avg_total_loss = torch.stack([m['logs']['loss'] for m in outputs]).mean()

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
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, eta_min=self.min_lr)
            # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=self.min_lr, verbose=True)
        elif self.scheduler=="multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
        else:
            return optimizer
        lr_scheduler = {'scheduler': scheduler,
                        'name': 'lr_scheduler'}

        return [optimizer], [lr_scheduler]

    # def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):        
    #     lr = (self.max_lr / 2) * (np.cos(self.trainer.global_step / self.cosine_steps_period) + 1) + self.min_lr
    #     for pg in optimizer.param_groups:
    #         pg['lr'] = lr  

    #     self.loggable_lr = lr
    #     optimizer.step()
    #     optimizer.zero_grad()

class EvalLightning(pl.LightningModule):

    def __init__(self,
                classifier=LinearClassifierAVT,
                data=Kinetics700Data,
                num_classes=700,
                feature_dimension=512,
                model_dimension=1024,
                num_modalities=3,
                batch_size=32,
                learning_rate=1e-3,
                model_path=None):

        super().__init__()

        self.data = data
        self.num_classes = num_classes
        self.model_dimension = model_dimension
        self.feature_dimension = feature_dimension
        self.num_modalities = num_modalities
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_path = model_path

        self.classifier = classifier(data=self.data,
                            num_classes=self.num_classes,
                            feature_dimension=self.feature_dimension,
                            model_dimension=self.model_dimension,
                            num_modalities=self.num_modalities,
                            batch_size=self.batch_size,
                            learning_rate=self.learning_rate,
                            model_path=self.model_path,)

        self.loss = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        a, v, t, label = batch
        logits = self.classifier(a, v, t)
        
        loss = self.loss(logits, label)
        top_1_accuracy = compute_accuracy(logits, label, top_k=1)
        top_5_accuracy = compute_accuracy(logits, label, top_k=5)

        logs = {
            'loss': loss,
            'train_top_1': top_1_accuracy,
            'train_top_5': top_5_accuracy}

        for k in logs:
            self.log('train/{}'.format(k), logs[k], prog_bar=False)

        return {'loss': loss,
                'logs': logs}

    def validation_step(self, batch, batch_idx):
        a, v, t, label = batch
        logits = self.classifier(a, v, t)

        loss = self.loss(logits, label)
        top_1_accuracy = compute_accuracy(logits, label, top_k=1)
        top_5_accuracy = compute_accuracy(logits, label, top_k=5)

        logs = {
            'val_loss': loss,
            'val_top_1': top_1_accuracy,
            'val_top_5': top_5_accuracy}

        for k in logs:
            self.log('val/{}'.format(k), logs[k], prog_bar=False)

        return logs

    def test_step(self, batch, batch_idx):
        a, v, t, label = batch
        logits = self.classifier(a, v, t)

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
        dataset = self.classifier.data(prefix='train')
        return torch.utils.data.DataLoader(
                                    dataset,
                                    batch_size=self.classifier.batch_size,
                                    shuffle=True,
                                    num_workers=8)

    def val_dataloader(self):
          dataset = self.classifier.data(prefix='val')
          return torch.utils.data.DataLoader(
                                  dataset,
                                  batch_size=self.classifier.batch_size,
                                  shuffle=False,
                                  num_workers=8)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
