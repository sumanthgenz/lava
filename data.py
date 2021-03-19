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

from aai.experimental.sgurram.lava.src.augment import get_npy_paths
from aai.experimental.sgurram.lava.src.utils import create_kinetics_labels, get_kinetics_labels
from references import pickle_root_dir
# from encoder import*

torchaudio.set_audio_backend("sox_io")
os.environ["IMAGEIO_FFMPEG_EXE"] = "/home/sgurram/anaconda3/bin/ffmpeg"
warnings.filterwarnings("ignore")

class LAVAData(Dataset):
    def __init__(self, prefix):
        self.prefix = prefix
        self.num_classes = 700

        # Fetch paths to audio, video, text features for samples
        self.a_paths, self.v_paths, self.t_paths = get_npy_paths(prefix)

    def __len__(self):
        return len(self.a_paths)

    def getNumClasses(self):
        return self.num_classes

    def __getitem__(self, idx):
        """ Return audio, video, text features from npy files"""
        try:
            a, v, t = np.load(self.a_paths[idx]), np.load(self.v_paths[idx]), np.load(self.t_paths[idx])
            a, v, t = torch.from_numpy(a), torch.from_numpy(v).to(dtype=torch.float), torch.from_numpy(t)
        except:
            print("Error at Idx: %d" % idx)
            return torch.zeros(128, 2048), torch.zeros(16, 128, 128, 3), torch.zeros(1, 512)

        assert a.shape == (128, 2048), "audio shape is {}".format(a.shape)
        assert v.shape == (16, 128, 128, 3), "video shape is {}".format(a.shape)
        assert t.shape == (1, 512), "text shape is {}".format(a.shape)
        return a, v, t



class Kinetics700Data(Dataset):
    def __init__(self,
                prefix,
                num_classes=700,
                zero_shot=False):

        self.prefix = prefix
        self.num_classes = num_classes
        self.zero_shot = zero_shot

        # Fetch paths to audio, video, text features for samples
        self.a_paths, self.v_paths, self.t_paths = get_npy_paths(prefix)
        self.labels = pickle.load(open("{}/{}.pickle".format(pickle_root_dir, prefix), "rb"))

    def __len__(self):
        return len(self.a_paths)

    def getNumClasses(self):
        return self.num_classes

    def get_label(self, path):
        filename = path.split('/')[-1][:-4]
        if self.zero_shot:
            #return encoded text
            return self.labels[filename][0]
        #return label in [0,699]
        return self.labels[filename][1]

    def __getitem__(self, idx):
        """ Return audio, video, text and the Kinetics label"""
        a, v, t = np.load(self.a_paths[idx]), np.load(self.v_paths[idx]), np.load(self.t_paths[idx])
        a, v, t = torch.from_numpy(a), torch.from_numpy(v).to(dtype=torch.float), torch.from_numpy(t)
        label = self.get_label(self.a_paths[idx])
        return a, v, t, label
