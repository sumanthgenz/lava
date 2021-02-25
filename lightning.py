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

from encoder import*
from data import LAVAData
from metrics import compute_accuracy

torchaudio.set_audio_backend("sox_io") 
warnings.filterwarnings("ignore")

class LAVALightning(pl.LightningModule):

    def __init__(self,logger):
        super().__init__()

        self.encoder = LAVA()
        self.logger = logger

    def training_step(self, batch, batch_idx):
        audio, video, text = batch
        audio_encoded, video_encoded, text_encoded = self.encoder(audio, video, text)
        metrics = self.encoder.loss(audio_encoded, video_encoded, text_encoded)
        loss = metrics['total_loss']
        metrics['loss'] = loss 

        return {'loss': loss,
                'logs': metrics}

    def validation_step(self, batch, batch_idx):
        audio, video, text = batch
        audio_encoded, video_encoded, text_encoded = self.encoder(audio, video, text)
        metrics = self.encoder.loss(audio_encoded, video_encoded, text_encoded)

        return {'val_total_loss': metrics['total_loss'],
                'val_a_loss': metrics['a_loss'],
                'val_v_loss': metrics['v_loss'],
                'val_t_loss': metrics['t_loss'],
                'val_av_loss': metrics['av_loss'],
                'val_at_loss': metrics['at_loss'],
                'val_vt_loss': metrics['vt_loss'],
                'val_avt_loss': metrics['avt_loss'],}

    def test_step(self, batch, batch_idx):
        audio, video, text = batch
        audio_encoded, video_encoded, text_encoded = self.encoder(audio, video, text)
        metrics = self.encoder.loss(audio_encoded, video_encoded, text_encoded)

        return {'test_total_loss': metrics['total_loss'],
                'test_avt_loss': metrics['avt_loss'],
                'test_av_loss': metrics['av_loss'],
                'test_at_loss': metrics['at_loss'],
                'test_vt_loss': metrics['vt_loss'],}
                
    
    def training_epoch_end(self, outputs):
        avg_a_loss = torch.stack([m['logs']['a_loss'] for m in outputs]).mean()
        avg_v_loss = torch.stack([m['logs']['v_loss'] for m in outputs]).mean()
        avg_t_loss = torch.stack([m['logs']['t_loss'] for m in outputs]).mean()
        avg_av_loss = torch.stack([m['logs']['av_loss'] for m in outputs]).mean()
        avg_at_loss = torch.stack([m['logs']['at_loss'] for m in outputs]).mean()
        avg_vt_loss = torch.stack([m['logs']['vt_loss'] for m in outputs]).mean()
        avg_avt_loss = torch.stack([m['logs']['avt_loss'] for m in outputs]).mean()
        avg_total_loss = torch.stack([m['logs']['loss'] for m in outputs]).mean()

  
        logs = {'train_total_loss': avg_total_loss,
                'train_a_loss': avg_a_loss,
                'train_v_loss': avg_v_loss,
                'train_t_loss': avg_t_loss,
                'train_av_loss': avg_av_loss,
                'train_at_loss': avg_at_loss,
                'train_vt_loss': avg_vt_loss,
                'train_avt_loss': avg_avt_loss,}

        self.logger.log_metrics(logs)

    def validation_epoch_end(self, outputs):
        avg_a_loss = torch.stack([m['val_a_loss'] for m in outputs]).mean()
        avg_v_loss = torch.stack([m['val_v_loss'] for m in outputs]).mean()
        avg_t_loss = torch.stack([m['val_t_loss'] for m in outputs]).mean()
        avg_av_loss = torch.stack([m['val_av_loss'] for m in outputs]).mean()
        avg_at_loss = torch.stack([m['val_at_loss'] for m in outputs]).mean()
        avg_vt_loss = torch.stack([m['val_vt_loss'] for m in outputs]).mean()
        avg_avt_loss = torch.stack([m['val_avt_loss'] for m in outputs]).mean()
        avg_total_loss = torch.stack([m['val_total_loss'] for m in outputs]).mean()

        logs = {'val_total_loss': avg_total_loss,
                'val_a_loss': avg_a_loss,
                'val_v_loss': avg_v_loss,
                'val_t_loss': avg_t_loss,
                'val_av_loss': avg_av_loss,
                'val_at_loss': avg_at_loss,
                'val_vt_loss': avg_vt_loss,
                'val_avt_loss': avg_avt_loss,}
        
        return {'val_total_loss': avg_total_loss, 'log': logs}

    def train_dataloader(self):
        dataset = LAVAData(prefix='train')
        return torch.utils.data.DataLoader(
                                    dataset,
                                    batch_size=self.encoder._batch_size,
                                    shuffle=True,
                                    num_workers=8)

    def val_dataloader(self):
          dataset = LAVAData(prefix='val')
          return torch.utils.data.DataLoader(
                                  dataset,
                                  batch_size=self.encoder._batch_size,
                                  shuffle=False,
                                  num_workers=8)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.encoder._learning_rate)
        return optimizer


class EvalLightning(pl.LightningModule):

    def __init__(self,
                logger=None,
                classifier=LinearClassifierAVT,):

        super().__init__()

        self.classifier = classifier()
        self.loss = torch.nn.CrossEntropyLoss()
        self.logger = logger

    def training_step(self, batch, batch_idx):
        a, v, t, label = batch
        logits = self.classifier(a, v, t)
        # print(self.classifer.training())
        
        loss = self.loss(logits, label)
        top_1_accuracy = compute_accuracy(logits, label, top_k=1)
        top_5_accuracy = compute_accuracy(logits, label, top_k=5)

        logs = {
            'loss': loss,
            'train_top_1': top_1_accuracy,
            'train_top_5': top_5_accuracy}

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

        return logs

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([m['logs']['loss'] for m in outputs]).mean()
        avg_top1 = torch.stack([m['logs']['train_top_1'] for m in outputs]).mean()
        avg_top5 = torch.stack([m['logs']['train_top_5'] for m in outputs]).mean()

        logs = {
        'train_loss': avg_loss,
        'train_top_1': avg_top1,
        'train_top_5': avg_top5}

        self.logger.log_metrics(logs)


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([m['val_loss'] for m in outputs]).mean()
        avg_top1 = torch.stack([m['val_top_1'] for m in outputs]).mean()
        avg_top5 = torch.stack([m['val_top_5'] for m in outputs]).mean()

        logs = {
        'val_loss': avg_loss,
        'val_top_1': avg_top1,
        'val_top_5': avg_top5}

        return {'val_loss': avg_loss, 'log': logs}

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
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.classifier.learning_rate)
        return optimizer
