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
import glob
from pytorchvideo.data import Hmdb51
import pickle

class HMDB51Dataset(Dataset):
    def __init__(self, 
                 filepath,
                 loader='npy',
                 split=1,
                 prefix="train",
                 ):
        super().__init__()
        
        self.data = []
        self.classes = [s.split('/')[-1] for s in list(glob.glob('/big/sgurram/HMDB-raw-npy/*'))]

        root = '/big/sgurram/HMDB-raw-npy'
  
        for split_class in glob.glob(f'/big/sgurram/HMDB-splits/*_split{int(split)}.txt'):
            label = split_class.split("/")[-1].split("_test")[0]
            all_files = open(split_class, 'r').readlines()
            cutoff = int(len(all_files)*0.7)
            if prefix == "train":
                files = all_files[:cutoff]
            else:
                files = all_files[cutoff:]
            for i, s in enumerate(files):
                name = s.split(" ")[0].replace('.avi', '.npy')
                self.data.append(f'{root}/{label}/{name}')
        
    def __getitem__(self, i):
        try:
            path = self.data[i]
            label = self.classes.index(path.split('/')[-2])

            v = np.load(path)
            v = augment_video(v)
            v = v.to(dtype=torch.float)
            return v, label
        except:
            return torch.zeros(16, 224, 224, 3), 0
        
    
    def __len__(self):
        return len(self.data)

    # if __name__ == "__main__":
        # root = '/big/sgurram/HMDB-raw-npy'
        # all_paths = []
        # for f in glob.glob('/big/sgurram/HMDB-raw-npy/*/*.npy'):
        #     all_paths.append(f)

        # train_split1 = []
        # test_split1 = []
        # for split_class in glob.glob('/big/sgurram/HMDB-splits/*_split1.txt'):
        #     label = split_class.split("/")[-1].split("_")[0]
        #     files = open(split_class, 'r').readlines()
        #     path = f'/big/sgurram/HMDB/{label}'
        #     for i, s in enumerate(files):
        #         if i < 70:
        #             train_split1.append(f'{root}/{label}/{s.split(" ")[0]}')
        #         else:
        #             test_split1.append(f'{root}/{label}/{s.split(" ")[0]}')


        # train_split2 = []
        # test_split2 = []        
        # for split_class in glob.glob('/big/sgurram/HMDB-splits/*_split2.txt'):
        #     label = split_class.split("/")[-1].split("_")[0]
        #     files = open(split_class, 'r').readlines()
        #     for i, s in enumerate(files):
        #         if i < 70:
        #             train_split2.append(f'{root}/{label}/{s.split(" ")[0]}')
        #         else:
        #             test_split2.append(f'{root}/{label}/{s.split(" ")[0]}')


        # train_split3 = []
        # test_split3 = []        
        # for split_class in glob.glob('/big/sgurram/HMDB-splits/*_split3.txt'):
        #     label = split_class.split("/")[-1].split("_")[0]
        #     files = open(split_class, 'r').readlines()
        #     for i, s in enumerate(files):
        #         if i < 70:
        #             train_split3.append(f'{root}/{label}/{s.split(" ")[0]}')
        #         else:
        #             test_split3.append(f'{root}/{label}/{s.split(" ")[0]}')



if __name__ == "__main__":
    confusion_dict = pickle.load(open('hmdb51_confusion_dictionary.pickle', "rb"))
    for k in confusion_dict:
        print(k, confusion_dict[k]) 