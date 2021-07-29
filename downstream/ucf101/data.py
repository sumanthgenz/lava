import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
# from aai.utils.video.file import load_video_to_numpy

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
    
    def __getitem__(self, i):
        path, label_raw = self.data[i].split(' ')
        label = int(label_raw)
        path = path.replace('iherzi', 'sgurram')
        # path = path.replace('UCF-101-LAVA-npy/', 'UCF-101-LAVA-guse-npy/')
        v = np.load(path.replace('.npy', '_v.npy'), allow_pickle=True).astype(np.float32)
        a = np.load(path.replace('.npy', '_a.npy'), allow_pickle=True).astype(np.float32)

        if a.shape != (1024,):
            a = np.zeros(1024)
            assert a.shape == v.shape, f"audio {a.shape}, video {v.shape}"

        a = torch.from_numpy(a).reshape(-1).to(dtype=torch.float32)
        v = torch.from_numpy(v).reshape(-1).to(dtype=torch.float32)
        a = nn.functional.normalize(a, p=2, dim=-1)
        v = nn.functional.normalize(v, p=2, dim=-1)
        av = torch.cat([a,v], dim=0)
        return av, label
        # return v, label
    
    def __len__(self):
        return self.size



    