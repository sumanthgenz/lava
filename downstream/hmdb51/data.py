import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
# from aai.utils.video.file import load_video_to_numpy

class HMDB51Dataset(Dataset):
    def __init__(self, 
                 filepath,
                 loader='npy'
                 ):
        super().__init__()
        with open(filepath, 'r') as f:
            self.data = f.readlines()
        print("Peeking files ", self.data[0])
        self.size = len(self.data)
    
    def __getitem__(self, i):
        path, label_raw = self.data[i].split(' ')
        label = int(label_raw)
        path = path.replace('iherzi', 'sgurram')
        # path = path.replace('UCF-101-LAVA-npy/', 'UCF-101-LAVA-guse-npy/')
        v = np.load(path.replace('.npy', '_v.npy'), allow_pickle=True).astype(np.float32)


        v = torch.from_numpy(v).reshape(-1).to(dtype=torch.float32)
        v = nn.functional.normalize(v, p=2, dim=-1)
        return v, label
    
    def __len__(self):
        return self.size



    