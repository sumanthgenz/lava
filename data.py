import torch
import torch.nn as nn
import torchvision
import torchaudio
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle
from tqdm import tqdm

import warnings
import glob
import gc 
import os
import socket

from augment import *
from metrics import *
from encoder import *
# from torchaudio_transforms import *

torchaudio.set_audio_backend("sox_io") 
os.environ["IMAGEIO_FFMPEG_EXE"] = "/home/sgurram/anaconda3/bin/ffmpeg"
warnings.filterwarnings("ignore")

data = ""
host = socket.gethostname()
if host == "stout":
    data = "big"
elif socket.gethostname() == "greybeard":
    data = "ssd"

class AudioData(Dataset):

    def __init__(self, dataType):
        self.dataType = dataType
        self.dir = "/{}/kinetics_audio/{}".format(data, dataType)
        self.num_classes = 700
        self.downsamp_factor = 2
        self.samp_freq = 22050*4
        self.seq_len = 500
        self.wav_paths = self.get_all_files()
        
    def get_all_files(self):
        wav_paths = []
        for path in glob.glob(f'{self.dir}/**/*.wav'):
            wav_paths.append(path)
        return wav_paths

    def get_pickle(self, classPath):
        with open('Desktop/kinetics_{}.pickle'.format(classPath), 'rb') as handle:
            result = pickle.load(handle)
        return result
    
    def __len__(self):
        return len(self.wav_paths)

    def getNumClasses(self):
        return self.num_classes

    def __getitem__(self, idx):
        try:
            filePath = self.wav_paths[idx]
            # num_label = int((filePath.split('/')[4]).split('_')[0]) - 1
            # wav, samp_freq = torchaudio.load(filePath)
            # feat = np.transpose(np.array(torchaudio.compliance.kaldi.mfcc(wav, sample_frequency=self.samp_freq)))
            # return feat, num_label, self.seq_len

            view1, view2, t1, t2 = get_augmented_views(filePath)
            # return view1.type(torch.FloatTensor), view2.type(torch.FloatTensor), t1, t2
            return view1, view2, t1, t2

        except:
            return None, None, None, None

class TemporalData(Dataset):

    def __init__(self, dataType):
        self.dataType = dataType
        self.dir = "/{}/kinetics_audio/{}".format(data, dataType)
        self.num_classes = 700
        self.downsamp_factor = 2
        self.samp_freq = 22050*4
        self.seq_len = 500
        self.wav_paths = self.get_all_files()
        
    def get_all_files(self):
        wav_paths = []
        for path in glob.glob(f'{self.dir}/**/*.wav'):
            wav_paths.append(path)
        return wav_paths

    def get_pickle(self, classPath):
        with open('Desktop/kinetics_{}.pickle'.format(classPath), 'rb') as handle:
            result = pickle.load(handle)
        return result
    
    def __len__(self):
        return len(self.wav_paths)

    def getNumClasses(self):
        return self.num_classes

    def __getitem__(self, idx):
        try:
            filePath = self.wav_paths[idx]
            # num_label = int((filePath.split('/')[4]).split('_')[0]) - 1
            # wav, samp_freq = torchaudio.load(filePath)
            # feat = np.transpose(np.array(torchaudio.compliance.kaldi.mfcc(wav, sample_frequency=self.samp_freq)))
            # return feat, num_label, self.seq_len

            anchor, permutes = get_temporal_shuffle_views(filePath)
            # return view1.type(torch.FloatTensor), view2.type(torch.FloatTensor), t1, t2
            return anchor, permutes

        except:
            return None, None, None, None


class AudioVisualData(Dataset):

    def __init__(self, dataType):
        self.dataType = dataType
        self.dir = "/big/davidchan/kinetics/kinetics_{}_clipped".format(dataType)
        self.num_classes = 700
        self.downsamp_factor = 2
        self.samp_freq = 22050*4
        self.seq_len = 500
        self.wav_paths = self.get_all_files()
        
    def get_all_files(self):
        wav_paths = []
        for path in glob.glob(f'{self.dir}/*.mp4'):
            wav_paths.append(path)
        return wav_paths

    def get_pickle(self, classPath):
        with open('Desktop/kinetics_{}.pickle'.format(classPath), 'rb') as handle:
            result = pickle.load(handle)
        return result
    
    def __len__(self):
        return len(self.wav_paths)

    def getNumClasses(self):
        return self.num_classes

    def __getitem__(self, idx):
        filePath = self.wav_paths[idx]

        return get_audiovisual(filePath)



if __name__ == '__main__':


    video_feat_dim = 14
    audio_feat_dim = 512

    #audio convnet 
    conv1 = torch.nn.Conv1d(
                in_channels=128, 
                out_channels=audio_feat_dim, 
                kernel_size=2, 
                stride=2,
    )

    conv2 = torch.nn.Conv1d(
                in_channels=audio_feat_dim, 
                out_channels=audio_feat_dim, 
                kernel_size=2,
                stride=2,
    )

    pool1 = nn.MaxPool1d(
            kernel_size=2,
            stride=2
    )

    audio_conv = nn.Sequential(
            conv1,
            conv2,
            pool1,
    )



    #video convnet 
    conv3 = torch.nn.Conv3d(
                in_channels=3, 
                out_channels=64, 
                kernel_size=[1,4,4],         
    )

    conv4 = torch.nn.Conv3d(
                in_channels=64, 
                out_channels=32, 
                kernel_size=[1,4,4], 
    )

    conv5 = torch.nn.Conv3d(
                in_channels=32, 
                out_channels=1, 
                kernel_size=[1,4,4], 
    )

    pool3 = nn.MaxPool3d(
                kernel_size=[1,5,5], 
    )
    
    pool4 = nn.MaxPool3d(
                kernel_size=[1,4,4], 
                stride=1,
    )

    pool5 = nn.MaxPool3d(
                kernel_size=[1,3,3], 
                stride=1,
    )

    video_conv = nn.Sequential(
            conv3,
            pool3,
            conv4,
            pool4,
            conv5,
            pool5,
    )

    fc = nn.Linear(video_feat_dim**2, audio_feat_dim)

    resnet_model = torchvision.models.resnet18(pretrained=True)

    feature_model = torch.nn.Sequential(
            resnet_model.conv1,
            resnet_model.bn1,
            resnet_model.relu,
            resnet_model.maxpool,
            resnet_model.layer1,
            resnet_model.layer2,
            resnet_model.layer3,
            resnet_model.layer4,
            )

    audio_model = AudioFeatureModel()
    video_model = VideoFeatureModel()

    model = CAVE(batch_size=5)

    ad = AudioVisualData("val")
    for i in tqdm(range(1)):
        # a, v = ad.__getitem__(1)

        a1, v1 = ad.__getitem__(0)
        a2, v2 = ad.__getitem__(25)
        a3, v3 = ad.__getitem__(500)
        a4, v4 = ad.__getitem__(1000)
        a5, v5 = ad.__getitem__(2500)

        a = torch.stack((a1, a2, a3, a4, a5))
        v = torch.stack((v1, v2, v3, v4, v5))

        view1, view2 = model(a,v)
        loss = model.loss(view1, view2)

        print(view1)
        print(view2)
        print(view1.shape)
        print(view2.shape)
        print(loss)

