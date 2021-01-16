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

from encoder import *

class VideoBYOLightning(pl.LightningModule):

    def __init__(self,):
        super().__init__()

        self.encoder = CAVE()

    # Handle BYOL component: Implemetation from https://github.com/CannyLab/aai/blob/main/aai/research/gptcaptions/driver.py
    def on_before_zero_grad(self, _):
        self.model.update_moving_average()


    def training_step(self, batch, batch_idx):
        audio, video = batch
        audio_encoded, video_encoded = self.encoder(audio, video)
        metrics = self.encoder.loss(audio_encoded, video_encoded)

        return {'loss': metrics['total_loss'],
                'logs': metrics}

    def validation_step(self, batch, batch_idx):
        audio, video = batch
        audio_encoded, video_encoded = self.encoder(audio, video)
        metrics = self.encoder.loss(audio_encoded, video_encoded)

        return {'val_total_loss': metrics['total_loss'],
                'val_audio_loss': metrics['audio_loss'],
                'val_video_loss': metrics['video_loss'],}
    
    def test_step(self, batch, batch_idx):
        audio, video = batch
        audio_encoded, video_encoded = self.encoder(audio, video)
        metrics = self.encoder.loss(audio_encoded, video_encoded)

        return {'test_total_loss': metrics['total_loss'],
                'test_audio_loss': metrics['audio_loss'],
                'test_video_loss': metrics['video_loss'],}

    
    def validation_epoch_end(self, outputs):
        avg_total_loss = torch.stack([m['val_total_loss'] for m in outputs]).mean()
        avg_audio_loss = torch.stack([m['val_audio_loss'] for m in outputs]).mean()
        avg_video_loss = torch.stack([m['val_video_loss'] for m in outputs]).mean()

        logs = {'val_total_loss': avg_total_loss,
                'val_audio_loss': avg_audio_loss,
                'val_video_loss': avg_video_loss,}
        
        return {'val_total_loss': avg_total_loss, 'log': logs}

    def train_dataloader(self):
        dataset = AudioVisualData(data_type='train')
        return torch.utils.data.DataLoader(
                                    dataset,
                                    batch_size=self.encoder._batch_size,
                                    shuffle=True,
                                    num_workers=8)

    def val_dataloader(self):
          dataset = AudioVisualData(data_type='val')
          return torch.utils.data.DataLoader(
                                  dataset,
                                  batch_size=self.encoder._batch_size,
                                  shuffle=False,
                                  collate_fn=self.collate_fn,
                                  num_workers=8)

    #test clipped folder does not exist on stout
    # def test_dataloader(self):
    #     dataset = AudioVisualData(data_type='test')
    #     return torch.utils.data.DataLoader(
    #                             dataset,
    #                             batch_size=self.encoder._batch_size,
    #                             shuffle=False,
    #                             num_workers=8)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.encoder._learning_rate)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=29, epochs=self.num_epochs)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.02)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70,80, 85, 90, 95], gamma=0.1)
        # return [optimizer], [scheduler]
        return optimizer
