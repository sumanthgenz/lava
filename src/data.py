import torch
import torch.nn as nn
import torchvision
import torchaudio
import torchtext
from torchtext.data.functional import generate_sp_model, load_sp_model, sentencepiece_numericalizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle
from tqdm import tqdm
import itertools

import warnings
import glob
import gc
import os
import socket

from aai.experimental.sgurram.lava.src.augment import augment_audio, augment_video, augment_cvrl, augment_views, get_audio_from_mp4, get_video_from_mp4, augment_visual_vocab, augment_video_for_visual_vocab
from aai.experimental.sgurram.lava.src.utils import create_kinetics_labels, get_kinetics_labels, process_tags, get_npy_paths
from aai.experimental.sgurram.lava.src.references import pickle_root_dir, sp_model_path, sp_vocab_size, raw_text_dir, downsample_root_dir, visual_vocab_dir
# from encoder import*

class LAVAData(Dataset):
    def __init__(self, prefix, pretrained_text=False):
        self.prefix = prefix
        self.pretrained_text = pretrained_text
        self.context_length = 128
        self.num_classes = 700
        self.vocab_size = sp_vocab_size
        self.sp_model_path = sp_model_path
        self.sp_model = load_sp_model(sp_model_path)
        self.sp_id_generator = sentencepiece_numericalizer(self.sp_model)
        self.start_token, self.end_token = self.sp_id_generator(["<|startoftext|>","<|endoftext|>"])
        self.text_dict = pickle.load(open(f"{raw_text_dir}/kinetics_{prefix}.pickle", "rb"))

        # Fetch paths to audio, video, text features for samples
        self.a_paths, self.v_paths, self.t_paths = get_npy_paths(prefix, pretrained_text=pretrained_text)
        print(len(self.t_paths))

        # length = len(self.a_paths)
        # limit = length // 3

        # self.a_paths, self.v_paths, self.t_paths = self.a_paths[:limit], self.v_paths[:limit], self.t_paths[:limit]

        self.tags = []

    def __len__(self):
        return len(self.a_paths)

    def getNumClasses(self):
        return self.num_classes

    def process_text(self, url):
        t =  self.text_dict[url]
        
        t_mask = 0
        if not (t == "A video of" or t == "A video of " or t == "A video " or t == "A video"):
            t_mask = 1

        tokens = list(itertools.chain(
            *[self.start_token] + list(self.sp_id_generator(t)) + [self.end_token]))[:self.context_length]
        t = torch.zeros(self.context_length, dtype=torch.long)
        t[:len(tokens)] = torch.tensor(tokens).flatten()
        return t, t_mask

    def __getitem__(self, idx):
        """ Return audio, video, text features from npy files"""

        # try:
        a, v = np.load(self.a_paths[idx]), np.load(self.v_paths[idx])
        a = augment_audio(a)
        v = augment_video(v)
        a, v = a.to(dtype=torch.float), v.to(dtype=torch.float)

        # v1, v2 = augment_views(v)
        # a, v1, v2 = a.to(dtype=torch.float), v1.to(dtype=torch.float), v2.to(dtype=torch.float)

        if self.pretrained_text:
            url = self.t_paths[idx].split('/')[-1].split('.')[0]
            t = torch.from_numpy(np.load(self.t_paths[idx])).to(dtype=torch.float)
        else:
            url = self.t_paths[idx]
            t = self.text_dict[url]

        # except:
        #     print(f"Error at Idx: {idx}")
        #     return torch.zeros(80, 512), torch.zeros(16, 128, 128, 3), torch.zeros(1, 512), self.t_paths[idx], 0

        t, t_mask = self.process_text(url) if not self.pretrained_text else (t, 1)
        # a = torch.rand(80, 512)
        # v = torch.rand(16, 256, 256, 3)

        return a, v, t, url, t_mask
        # return a, v1, v2, t, url, t_mask

class LAVADataVV(Dataset):
    def __init__(self, prefix, pretrained_text=False):
        self.prefix = prefix
        self.pretrained_text = pretrained_text
        self.context_length = 128
        self.num_classes = 700
        self.vocab_size = sp_vocab_size
        self.sp_model_path = sp_model_path
        self.sp_model = load_sp_model(sp_model_path)
        self.sp_id_generator = sentencepiece_numericalizer(self.sp_model)
        self.start_token, self.end_token = self.sp_id_generator(["<|startoftext|>","<|endoftext|>"])
        self.text_dict = pickle.load(open(f"{raw_text_dir}/kinetics_{prefix}.pickle", "rb"))
        self.visual_vocab = pickle.load(open('path_visual_vocab_dir.pickle', "rb"))

        # Fetch paths to audio, video, text features for samples
        self.a_paths, self.v_paths, self.t_paths = get_npy_paths(prefix, pretrained_text=pretrained_text)
        # print(len(self.t_paths))

        # length = len(self.a_paths)
        # limit = length // 3

        # self.a_paths, self.v_paths, self.t_paths = self.a_paths[:limit], self.v_paths[:limit], self.t_paths[:limit]

        self.tags = []

    def __len__(self):
        return len(self.a_paths)

    def getNumClasses(self):
        return self.num_classes

    def process_text(self, url):
        text =  self.text_dict[url]
        t_mask = 0
        if not (text == "A video of" or text == "A video of " or text == "A video " or text == "A video"):
            t_mask = 1

        tokens = list(itertools.chain(
            *[self.start_token] + list(self.sp_id_generator(text)) + [self.end_token]))[:self.context_length]
        t = torch.zeros(self.context_length, dtype=torch.long)
        t[:len(tokens)] = torch.tensor(tokens).flatten()

        return t, t_mask
    
    def get_visual_vocab(self, url):
        if url in self.visual_vocab:
            img_paths = self.visual_vocab[url]
            random.shuffle(img_paths)
            imgs = [augment_visual_vocab(i) for i in img_paths[:4]]
            return torch.stack(imgs)
        return None

    def __getitem__(self, idx):
        """ Return audio, video, text features from npy files"""

        # try:
        a, v = np.load(self.a_paths[idx]), np.load(self.v_paths[idx])
        a = augment_audio(a)
        v = augment_video(v)
        a, v = a.to(dtype=torch.float), v.to(dtype=torch.float)

        # v1, v2 = augment_views(v)
        # a, v1, v2 = a.to(dtype=torch.float), v1.to(dtype=torch.float), v2.to(dtype=torch.float)

        if self.pretrained_text:
            url = self.t_paths[idx].split('/')[-1].split('.')[0]
            t = torch.from_numpy(np.load(self.t_paths[idx])).to(dtype=torch.float)
        else:
            url = self.t_paths[idx]
            t = self.text_dict[url]
        
        imgs = self.get_visual_vocab(url)
        if imgs is None:
            imgs = augment_video_for_visual_vocab(v)
        imgs = imgs.to(dtype=torch.float)

        # except:
        #     print(f"Error at Idx: {idx}")
        #     return torch.zeros(80, 512), torch.zeros(16, 128, 128, 3), torch.zeros(1, 512), self.t_paths[idx], 0

        t, t_mask = self.process_text(url) if not self.pretrained_text else (t, 1)
        # a = torch.rand(80, 512)
        # v = torch.rand(16, 256, 256, 3)

        return a, v, t, imgs, url, t_mask
        # return a, v1, v2, t, url, t_mask

class LAVADataMP4(Dataset):
    def __init__(self, prefix, pretrained_text=False):
        self.prefix = prefix
        self.pretrained_text = pretrained_text
        self.context_length = 128
        self.num_classes = 700
        self.vocab_size = sp_vocab_size
        self.sp_model_path = sp_model_path
        self.sp_model = load_sp_model(sp_model_path)
        self.sp_id_generator = sentencepiece_numericalizer(self.sp_model)
        self.start_token, self.end_token = self.sp_id_generator(["<|startoftext|>","<|endoftext|>"])
        self.text_dict = pickle.load(open(f"{raw_text_dir}/kinetics_{prefix}.pickle", "rb"))

        # Fetch paths to audio, video, text features for samples
        self.paths = glob.glob(f'{downsample_root_dir}/{prefix}/*.mp4')
        print(len(self.paths))

    def __len__(self):
        return len(self.paths)

    def getNumClasses(self):
        return self.num_classes

    def process_text(self, url):
        try:
            t =  self.text_dict[url]
        except:
            return torch.zeros(128, dtype=torch.long), 0
        
        t_mask = 0
        if not (t == "A video of" or t == "A video of " or t == "A video " or t == "A video"):
            t_mask = 1

        tokens = list(itertools.chain(
            *[self.start_token] + list(self.sp_id_generator(t)) + [self.end_token]))[:self.context_length]
        t = torch.zeros(self.context_length, dtype=torch.long)
        t[:len(tokens)] = torch.tensor(tokens).flatten()
        return t, t_mask

    def __getitem__(self, idx):
        """ Return audio, video, text features from npy files"""

        path = self.paths[idx]
        url = path.split(f'{self.prefix}/')[-1][:-4]
        a, v = get_audio_from_mp4(path, save=False), get_video_from_mp4(path, save=False)

        a = augment_audio(a).to(dtype=torch.float)
        v = augment_video(v).to(dtype=torch.float)
        t, t_mask = self.process_text(url)
        return a, v, t, url, t_mask

class Kinetics700Data(Dataset):
    def __init__(self,
                prefix,
                num_classes=700,
                zero_shot=False,
                pretrained_text=False):

        self.prefix = prefix
        self.num_classes = num_classes
        self.zero_shot = zero_shot
        self.pretrained_text = pretrained_text

        self.context_length = 128
        self.vocab_size = 20000
        self.sp_model_path = sp_model_path
        self.sp_model = load_sp_model(sp_model_path)
        self.sp_id_generator = sentencepiece_numericalizer(self.sp_model)
        self.start_token, self.end_token = self.sp_id_generator(["<|startoftext|>","<|endoftext|>"])
        self.text_dict = pickle.load(open(f"{raw_text_dir}/kinetics_{prefix}.pickle", "rb"))

        # Fetch paths to audio, video, text features for samples
        self.a_paths, self.v_paths, self.t_paths = get_npy_paths(prefix, pretrained_text=pretrained_text)
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
        if self.pretrained_text:
            a, v, t = np.load(self.a_paths[idx]), np.load(self.v_paths[idx]), np.load(self.t_paths[idx])
            a,v = augment_audio(a), augment_video(v)
            a, v, t = torch.from_numpy(a), torch.from_numpy(v).to(dtype=torch.float), torch.from_numpy(t)
            label = self.get_label(self.a_paths[idx])                
            url = self.t_paths[idx].split('/')[-1].split('.')[0]
            return a, v, t, label, url
            # return 0, 0, t, label, url
        else:
            a, v, t = np.load(self.a_paths[idx]), np.load(self.v_paths[idx]), self.text_dict[self.t_paths[idx]]
            a,v = augment_audio(a), augment_video(v)
            # a, v = torch.from_numpy(a).to(dtype=torch.float), torch.from_numpy(v).to(dtype=torch.float)
            a, v = a.to(dtype=torch.float), v.to(dtype=torch.float)
            tokens = list(itertools.chain(
                *[self.start_token] + list(self.sp_id_generator(t)) + [self.end_token]))[:self.context_length]
            t = torch.zeros(self.context_length, dtype=torch.long)
            t[:len(tokens)] = torch.tensor(tokens).flatten()
            label = self.get_label(self.a_paths[idx])
        # print(a.shape)
        # print(v.shape)
        # print(t.shape)
        return a, v, t, label, self.t_paths[idx]
        # return 0, 0, t, label, self.t_paths[idx]

class K600Dataset(Dataset):
    def __init__(self, 
                 prefix,
                 loader='npy'
                 ):
        super().__init__()


        self.root = f'/big/sgurram/kinetics600/{prefix}'
        self.classes = []
        self.data = []
        for path in glob.glob(f'{self.root}/*'):
            self.classes.append(path.split('/')[-1])


        for path in glob.glob(f'{self.root}/*/*.npy'):
            self.data.append(path)

    def __len__(self):
        return len(self.data)

    def getNumClasses(self):
        return len(self.classes)
    
    def __getitem__(self, i):
        path = self.data[i]
        label = self.classes.index(path.split('/')[-2])

        v = np.load(path)
        v = augment_video(v)
        v = v.to(dtype=torch.float)
        return torch.zeros(80, 512), v, torch.zeros(128), label, ""

class CVRLData(Dataset):
    def __init__(self, prefix):
        self.prefix = prefix
        self.num_classes = 700

        _, self.v_paths, _ = get_npy_paths(prefix, pretrained_text=False)
        
        print(len(self.v_paths))

    def __len__(self):
        return len(self.v_paths)

    def getNumClasses(self):
        return self.num_classes

    def __getitem__(self, idx):
        v = np.load(self.v_paths[idx])
        v1, v2 = augment_cvrl(v)
        v1, v2 = v1.to(dtype=torch.float), v2.to(dtype=torch.float)
        # v1, v2 = torch.rand(16, 3, 128, 128), torch.rand(16, 3, 128, 128)
        return v1, v2

class CVRLKinetics700Data(Dataset):
    def __init__(self,
                prefix,
                num_classes=700,):

        self.prefix = prefix
        self.eval = prefix=="val"
        self.num_classes = num_classes

      
        _, self.v_paths, _ = get_npy_paths(prefix, pretrained_text=False)
        print(self.v_paths[0])
        self.labels = pickle.load(open("{}/{}.pickle".format(pickle_root_dir, prefix), "rb"))

    def __len__(self):
        return len(self.v_paths)

    def getNumClasses(self):
        return self.num_classes

    def get_label(self, path):
        filename = path.split('/')[-1][:-4]
        return self.labels[filename][1]

    def __getitem__(self, idx):
        """ Return audio, video, text and the Kinetics label"""
        v = np.load(self.v_paths[idx])
        v1, v2 = augment_cvrl(v, eval=self.eval)
        v1 = v1.to(dtype=torch.float)
        v2 = v2.to(dtype=torch.float)
        label = self.get_label(self.v_paths[idx])                
        return v1, v2, label

class UCF101Dataset(Dataset):
    def __init__(self, 
                 prefix,
                 loader='npy'
                 ):
        super().__init__()

        if prefix=="train":
            filepath = '/big/iherzi/ucfTrainTestlist/trainlist01.txt'
            self.eval = False
        else:
            filepath = '/big/iherzi/ucfTrainTestlist/testlist01.txt'
            self.eval = True
        with open(filepath, 'r') as f:
            self.data = f.readlines()
        print("Peeking files ", self.data[0])
        self.size = len(self.data)
    
    def __getitem__(self, i):
        path, label_raw = self.data[i].split(' ')
        label = int(label_raw)
        path = path.replace('iherzi', 'sgurram')
        path = path.replace('UCF-101-LAVA-npy/', 'UCF-101-raw-npy/')
        v = np.load(path.replace('.npy', '_v.npy'), allow_pickle=True)
        v1, v2 = augment_cvrl(v, eval=self.eval)
        # f, arr = plt.subplots(1,2)
        # arr[0].imshow(v1[0].permute(1, 2, 0))
        # arr[1].imshow(v2[0].permute(1, 2, 0))
        # plt.savefig("cvrl_ucf_sanity.png")
        v1 = v1.to(dtype=torch.float).permute(0, 2, 3, 1)
        # v2 = v2.to(dtype=torch.float)
        # return 1/0
        return torch.zeros(80, 512), v1, torch.zeros(128), label, ""
    
    def __len__(self):
        return self.size

if __name__=="__main__":
    # prefix = "train"
    # print(len(glob.glob(f'/big/sgurram/kinetics600/{prefix}/*/*.npy')))
    data = LAVAData(prefix="train")
    a,v,t,i,_,_ = data[5]
    plt.imshow( i.to(dtype=torch.int)[0])
    plt.savefig("sanity_checks/visual_vocab")

class HT100MData(Dataset):
    def __init__(self, 
                 prefix,
                 ):
        super().__init__()

        if prefix=="train":
            self.size = int(1.4e8)
        else:
            self.size = int(1e6)
    
    def __getitem__(self, i):
        return torch.ones(64).to(dtype=torch.int), torch.ones(4, 16, 16).to(dtype=torch.int), torch.ones(128).to(dtype=torch.int)
    
    def __len__(self):
        return self.size

if __name__=="__main__":
    # prefix = "train"
    # print(len(glob.glob(f'/big/sgurram/kinetics600/{prefix}/*/*.npy')))
    data = LAVAData(prefix="train")
    a,v,t,i,_,_ = data[5]
    plt.imshow( i.to(dtype=torch.int)[0])
    plt.savefig("sanity_checks/visual_vocab")
