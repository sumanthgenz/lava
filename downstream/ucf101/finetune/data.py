import torch
from torchvision.transforms import CenterCrop, RandomCrop
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import os
# from aai.utils.video.file import load_video_to_numpy
from aai.experimental.sgurram.lava.src.augment import augment_video
from aai.experimental.sgurram.lava.src.utils import pad_video
# from aai.experimental.ilianherzi.augmented_video_learning.video_transforms import CenterCrop
import matplotlib.pyplot as plt
import pickle

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
        
    def __getitem__(self, i):
        frame_size=224
        try:
            try:
                path, label_raw = self.data[i].split(' ')
                if '\n' in label_raw:
                    label_raw = label_raw[:-1]
                label = int(label_raw) - 1
                path = path.replace('iherzi', 'sgurram')
                path = path.replace('UCF-101-LAVA-npy/', 'UCF-101-raw-npy/')
                path = path.replace('.avi', '.npy')
                # path = path.replace('.npy', '_v.npy')
                if 'big' not in path:
                    path = '/big/sgurram/UCF-101-raw-npy/' + path
                v = np.load(path, allow_pickle=True).astype(np.float32)
            except:
                # path = '/big/sgurram/UCF-101-raw-npy/' + self.data[i].replace('.avi\n', '_v.npy')
                path = '/big/sgurram/UCF-101-raw-npy/' + self.data[i].replace('.avi\n', '.npy').replace('.avi', '.npy')
                path = path.split(" ")[0]
                class_name = path.split('_')[1].replace("Handstand", "HandStand")
                label_raw = int(self.classes[class_name]) - 1
                label = int(label_raw)
                v = np.load(path, allow_pickle=True).astype(np.float32)
            # a = np.load(path.replace('.npy', '_a.npy'), allow_pickle=True).astype(np.float32)
            # a = torch.from_numpy(a).reshape(-1).to(dtype=torch.float32)

            # v, _ = augment_video(v)
            subsample_rate = max(1, int(v.shape[0]) // 16)
            v = v[::subsample_rate][:16]
            v = (torch.from_numpy(v)).to(dtype=torch.float32)
            v = CenterCrop(frame_size)(v.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            if v.shape != (16, frame_size, frame_size, 3):
                v = pad_video(v, 16)
            return v, label
        except:
            print("bad path")
            return torch.zeros(16, frame_size, frame_size, 3), 0
    
    def __len__(self):
        return self.size

if __name__ == "__main__":
    confusion_dict = pickle.load(open('ucf101_confusion_dictionary.pickle', "rb"))
    for k in confusion_dict:
        print(k, confusion_dict[k]) 
