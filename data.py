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
            return a, v, t

        except:
            print("Error at Idx: %d" % idx)
            return torch.zeros(128, 2048), torch.zeros(16, 128, 128, 3), torch.zeros(1, 512)

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
        self.labels = pickle.load(open("/home/sgurram/Desktop/kinetics/{}.pickle".format(prefix), "rb"))

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


if __name__ == '__main__':

    a, v, t = torch.ones(32, 128), torch.ones(32, 128), torch.ones(32, 128)
    res = torch.stack((a, v, t))
    print(res.mean(dim=0).squeeze().shape)

    """ Evaluate Model Forward and Losses """
    # batch_size = 5
    # lava_data = LAVAData("val")
    # model = LAVA(batch_size=batch_size)

    # a1, v1 = lava_data[0]
    # a2, v2 = lava_data[250]
    # a3, v3 = lava_data[500]
    # a4, v4 = lava_data[1000]
    # a5, v5 = lava_data[2500]


    # a = torch.stack((a1, a2, a3, a4, a5))
    # v = torch.stack((v1, v2, v3, v4, v5))

    # x, y, z = torch.rand(5, 100), torch.rand(5, 100), torch.rand(5, 100)
    # dist = centroid_loss(x, y, z)
    # print(dist)

        # a, v = audio_model(a), video_model(v)
        # print(a.shape)
        # print(v.shape)

        # view1, view2 = model(a,v)
        # loss = model.loss(view1, view2)

        # print(view1)
        # print(view2)
        # print(view1.shape)
        # print(view2.shape)
        # print(loss)


    """Benchmark Model Pipeline Speeds"""
    # batch_size = 64
    # lava_data = LAVAData("val")
    # model = LAVA()
    # optimizer = torch.optim.Adam(model.parameters(),
    #                             lr=model._learning_rate)
    # dataloader = torch.utils.data.DataLoader(lava_data,
    #                                         batch_size=batch_size,
    #                                         shuffle=False,
    #                                         num_workers=8)


    # start_data_load = time.time()
    # batch = next(iter(dataloader))
    # a, v = batch
    # end_data_load = time.time()

    # start_forward = time.time()
    # a_encode, v_encode = model(a, v)
    # end_forward = time.time()

    # start_loss = time.time()
    # loss = model.loss(a_encode,v_encode)['total_loss']
    # end_loss = time.time()

    # start_backward = time.time()
    # loss.backward()
    # end_backward = time.time()

    # start_optimizer = time.time()
    # optimizer.step()
    # optimizer.zero_grad()
    # end_optimizer = time.time()

    # timings = {"load_batch": end_data_load - start_data_load,
    #           "forward": end_forward - start_forward,
    #           "loss": end_loss - start_loss,
    #           "backward": end_backward - start_backward,
    #           "optimizer": end_optimizer - start_optimizer,
    # }

    # print(timings)


    """ Create and debug kinetics labels for linear probe experiments"""
    # create_kinetics_labels()

    # prefix = "train"
    # labels = pickle.load(open("/home/sgurram/Desktop/kinetics/{}.pickle".format(prefix), "rb"))

    # for k in labels:
    #     if not (0 <= labels[k][1] < 700):
    #         print(k, labels[k])

    # train =  pickle.load(open("/home/sgurram/Desktop/kinetics/kinetics_train.pickle", "rb"))
    # print(train["27RMXJm71A8"])
