import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import os
# from aai.utils.video.file import load_video_to_numpy
from aai.experimental.sgurram.lava.src.augment import augment_video
from aai.experimental.ilianherzi.augmented_video_learning.video_transforms import CenterCrop

class UCF101Dataset(Dataset):
    def __init__(self, 
                 filepath,
                 loader='npy'
                 ):
        super().__init__()
        with open(filepath, 'r') as f:
            self.data = f.readlines()
        print("Peeking files ", self.data[0])
        self.size = len(self.data)

        with open('/big/iherzi/ucfTrainTestlist/classInd.txt', 'r') as f:
            class_file = f.readlines()
        self.classes = {}
        for i in range(len(class_file)):
            v, k = class_file[i].split(' ')
            k = k.replace("Handstand", "HandStand")
            self.classes[k[:-1]] = v

        # print(os.path.exists('/big/sgurram/UCF-101-raw-npy/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01_v.npy'))
        # print(os.path.exists('/big/sgurram/UCF-101-raw-npy/CricketBowling/v_CricketBowling_g20_c02_v.npy'))
        # print(os.path.exists('/big/sgurram/UCF-101-raw-npy/Diving/v_Diving_g04_c05_v.npy'))

    def __getitem__(self, i):
        try:
            path, label_raw = self.data[i].split(' ')
            if '\n' in label_raw:
                label_raw = label_raw[:-1]
            label = int(label_raw) - 1
            path = path.replace('iherzi', 'sgurram')
            path = path.replace('UCF-101-LAVA-npy/', 'UCF-101-raw-npy/')
            path = path.replace('.avi', '.npy')
            path = path.replace('.npy', '_v.npy')
            if 'big' not in path:
                path = '/big/sgurram/UCF-101-raw-npy/' + path
            v = np.load(path, allow_pickle=True).astype(np.float32)
        except:
            path = '/big/sgurram/UCF-101-raw-npy/' + self.data[i].replace('.avi\n', '_v.npy')
            class_name = path.split('_')[1].replace("Handstand", "HandStand")
            label_raw = int(self.classes[class_name]) - 1
            label = int(label_raw)
            v = np.load(path, allow_pickle=True).astype(np.float32)
        # a = np.load(path.replace('.npy', '_a.npy'), allow_pickle=True).astype(np.float32)
        # a = torch.from_numpy(a).reshape(-1).to(dtype=torch.float32)

        # v, _ = augment_video(v)
        subsample_rate = max(1, int(v.shape[0]) // 16)
        v = v[::subsample_rate][:16]
        v = CenterCrop()(v)
        v = torch.from_numpy(v).to(dtype=torch.float32)
        if v.shape != (16, 128, 128, 3):
            v = torch.zeros((16, 128, 128, 3))
        return v, label
    
    def __len__(self):
        return self.size



    