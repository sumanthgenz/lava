import torch
import torch.nn as nn
import torchvision
import torchaudio
import torchtext
from torchtext.data.functional import generate_sp_model, load_sp_model, sentencepiece_numericalizer
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tqdm import tqdm
import av
import pickle
from typing import Tuple, Optional
import os
import warnings
import glob
import itertools

from aai.experimental.sgurram.lava.src.new_encoder2 import LAVA
from aai.experimental.sgurram.lava.src.utils import get_src_conditional_mask, position_embed, position_embed_3d
from aai.experimental.sgurram.lava.src.metrics import cosine_similarity
from aai.experimental.sgurram.lava.src.references import lava_weights_path, sp_model_path, raw_text_dir
from aai.experimental.sgurram.lava.src.lightning import LAVALightning
from aai.experimental.sgurram.lava.src.augment import get_npy_paths, get_audio_from_mp4, get_video_from_mp4, get_audio_from_wav, augment_video

class LAVAFeatures():

    def __init__(self,
                 lava_model_path=lava_weights_path,
                 guse_model_path="https://tfhub.dev/google/universal-sentence-encoder/4",
                 sp_model_path=sp_model_path,
                 prefix="train"):

        self.guse_model_path = guse_model_path
        self.lava_model_path = lava_model_path

        self.sp_model_path = sp_model_path
        self.sp_model = load_sp_model(sp_model_path)
        self.sp_id_generator = sentencepiece_numericalizer(self.sp_model)
        self.start_token, self.end_token = self.sp_id_generator(["<|startoftext|>","<|endoftext|>"])
        self.context_length = 128

        # self.guse_model = hub.load(self.guse_model_path)
        self.model = LAVALightning(pretrained_text=False, num_heads=4, num_layers=4, model_dimension=1024)
        self.model.load_state_dict(torch.load(lava_model_path, map_location='cpu')['state_dict'], strict=True)
        self.model.eval()

        self.lava_model = self.model.encoder

        # self.a_feature_model = self.lava_model._audio_feature_model
        # self.a_projection = self.lava_model._audio_input_projection
        # self.a_encoder = self.lava_model._audio_encoder

        # self.v_feature_model = self.lava_model._video_feature_model
        # self.v_projection = self.lava_model._video_input_projection
        # self.v_encoder = self.lava_model._video_encoder

        # self.t_feature_model = self.lava_model._text_feature_model
        # self.t_projection = self.lava_model._text_input_projection

        # self.feature_dimension = self.lava_model._feature_dimension
        # self.model_dimension = self.lava_model._model_dimension

        self.a_encoder = self.lava_model.a_encoder
        self.v_encoder = self.lava_model.v_encoder
        self.t_encoder = self.lava_model.t_encoder

        self.feature_dimension = self.lava_model.feature_dimension
        self.model_dimension = self.lava_model.model_dimension

    def get_lava_features(self,
                    mp4_path=None,
                    text_input=None,
                    wav_path=None,
                    save=False,
                    save_dir=None,
                    run_lava=True):

        """
        Args:
            mp4_path: (str) input path to mp4 file to extract video and audio streams (if any)
            text: (str) (optional) text metadata for a video to be encoded using GUSE
            wav_path: (str) (optional) path to a wav file for the audio stream
            save: (bool) whether or not to save the extracted features
            save_dir: (str) path to the file to be saved (include train, val)
        Return:
            If SAVE is false, will return audio, video and text features (if any) produced by pre-trained LAVA.
            Otherwise, will save features to numpy files.
            Note: if a given input only has video, no corresponding text or audio numpy files will be available for that input.
        """

        a, v, t = None, None, None

        # loading data for audio, video, text (if any)
        # if wav_path:
        #     a = get_audio_from_wav(wav_path)
        # else:
        #     a = get_audio_from_mp4(mp4_path)

        if mp4_path:
            v = get_video_from_mp4(mp4_path)
            v = torch.from_numpy(augment_video(v, eval=True))

        if text_input:
            t = self.process_text(text_input)

        # encoding data with lava to get features (if any)
        if run_lava:
            with torch.no_grad():
                if a is not None:
                    a = self.encode_audio(a.unsqueeze(0)).squeeze().detach().cpu().numpy()
                if v is not None:
                    v = self.encode_video(v.to(dtype=torch.float).unsqueeze(0)).squeeze().detach().cpu().numpy()
                if t is not None:
                    t = self.encode_text(t.unsqueeze(0)).squeeze().detach().cpu().numpy()
        if save:
            filename = (mp4_path.split('/')[-1]).split('.')[0]
            a_path = '{}/audio'.format(save_dir)
            v_path = '{}/video'.format(save_dir)
            t_path = '{}/text'.format(save_dir)

            self.save_npy_file(features=a, dir=a_path, filename=filename) if a is not None else None
            self.save_npy_file(features=v, dir=v_path, filename=filename) if v is not None else None
            self.save_npy_file(features=t, dir=t_path, filename=filename) if t is not None else None

        return a, v, t

    def process_text(self, x, guse=False):
        if guse:
            return torch.from_numpy(self.guse_model([x]).numpy())     
        else:
            tokens = list(itertools.chain(
            *[self.start_token] + list(self.sp_id_generator(x)) + [self.end_token]))[:self.context_length]
            text = torch.zeros(self.context_length, dtype=torch.long)
            text[:len(tokens)] = torch.tensor(tokens).flatten()
            return text

    def encode_audio(self, x):
        # with torch.no_grad():
        #     x = self.a_feature_model(x)
        #     x = self.a_projection(x.reshape(-1, self.feature_dimension)).reshape(
        #             x.shape[0], x.shape[1], self.model_dimension)

        #     x = self.a_encoder(src=x).mean(dim=1)
        # x = torch.nn.functional.normalize(x, p=2, dim=-1)
        # return x
        return self.a_encoder(x)


    def encode_video(self, x):
        # with torch.no_grad():
        #     x = self.v_feature_model(x)
        #     x = self.v_projection(x.reshape(-1, self.feature_dimension)).reshape(
        #             x.shape[0], x.shape[1], self.model_dimension)

        #     x = self.v_encoder(src=x).mean(dim=1)
        # x = torch.nn.functional.normalize(x, p=2, dim=-1)
        # return x
        return self.v_encoder(x)

    def encode_text(self, x):
        # with torch.no_grad():
        #     x = self.t_feature_model(x)
        #     x = self.t_projection(x).squeeze()

        # x = torch.nn.functional.normalize(x, p=2, dim=-1)
        # return x
        return self.t_encoder(x)

    # def remove_batchnorm(self, model, layer_idx):
    #     return torch.nn.Sequential(*(list(model.children())[:layer_idx] + list(model.children())[layer_idx+1:]))

    def save_npy_file(self, features, dir, filename):
            if not os.path.exists(dir):
                os.makedirs(dir)
            path = "{}/{}".format(dir, filename)
            np.save(path, features.detach().numpy())

if __name__ == "__main__":

    modes = ["guse", "lava_old", "lava_new", "lava_init_state_dict_bug, lava_avt"]
    mode = 4


    if mode == 1:
        # old lava (need to import LAVA from encoder instead of new_encoder in lightning.py)
        ckpt_path = "/home/sgurram/Desktop/video_lava/lava/1rexfc6g/checkpoints/epoch=18.ckpt"
    elif mode == 2 or mode == 4:
        # new lava
        ckpt_path = "/home/sgurram/Projects/aai/aai/experimental/sgurram/lava/src/wandb/run-20210329_160536-521szt2u/files/lava/521szt2u/checkpoints/epoch=1-step=2863.ckpt"
        ckpt_path = "/home/sgurram/Projects/aai/aai/experimental/sgurram/lava/src/wandb/run-20210330_183407-342xpuhr/files/lava/342xpuhr/checkpoints/epoch=49-step=499.ckpt"
        ckpt_path = lava_weights_path
        ckpt_path = "/home/sgurram/Projects/aai/aai/experimental/sgurram/lava/src/wandb/run-20210401_141821-3chj78e7/files/lava/3chj78e7/checkpoints/epoch=49-step=499.ckpt"
    
    if mode == 1 or mode == 2 or mode == 4:
        lava_ptl = LAVALightning(batch_size=2)
        lava_ptl.load_from_checkpoint( ckpt_path, strict=True)

    f = LAVAFeatures()
    text1 = "a video of dog shaking hand, rottweiler puppy shaking hand, puppy,"
    text2 = "a video of  rottweiler shaking hand, dog shaking hand, rottweiler puppy shaking hand"
    text3 = "a video of  pink, purple, teal, blue, blonde, dying hair, why are you reading this lol, "

    ## Pure GUSE ##
    if mode == 0:
        t1 = torch.from_numpy(f.guse_model([text1]).numpy()).squeeze()
        t2 = torch.from_numpy(f.guse_model([text2]).numpy()).squeeze()
        t3 = torch.from_numpy(f.guse_model([text3]).numpy()).squeeze()


    ## Fully loaded LAVA (old, GUSE-based)
    if mode == 1:
        t1  = f.process_text(text1, guse=True).unsqueeze(0)
        t2 = f.process_text(text2, guse=True).unsqueeze(0)
        t3 = f.process_text(text3, guse=True).unsqueeze(0)
        t = torch.cat((t1, t2, t3), dim=0)
        t1, t2, t3 = lava_ptl.encoder.encode_text(t)

    ## Fully loaded LAVA (new)
    if mode == 2:
        t1 = f.process_text(text1).unsqueeze(0)
        t2 = f.process_text(text2).unsqueeze(0)
        t3 = f.process_text(text3).unsqueeze(0)

        t = torch.cat((t1, t2, t3), dim=0)

        # t1, t2, t3 = lava_ptl.encoder.t_encoder(t)

        t1, t2, t3  = lava_ptl.encoder.encode_text(t)


    ## Not fully loaded LAVA (state_dict bug)
    if mode == 3:
        t1 = f.encode_text(f.process_text(text1).unsqueeze(0))
        t2 = f.encode_text(f.process_text(text2).unsqueeze(0))
        t3 = f.encode_text(f.process_text(text3).unsqueeze(0))

    if mode == 4:
        text_dict = pickle.load(open(f"{raw_text_dir}/kinetics_train.pickle", "rb"))
        # a, v, t = get_npy_paths("train", pretrained_text=False)
        # i, k = 250, 250
        # a1, v1, t1 = a[i], v[i], t[i]
        # a2, v2, t2 = a[k], v[k], t[k]

        path1  = "07jNiB-vSZc"
        path2 = "07jNiB-vSZc"
        a_prefix, v_prefix = '/big/sgurram/kinetics_numpy/train/audio', '/big/sgurram/kinetics_numpy/train/video'
        a1, v1, t1 = f"{a_prefix}/{path1}.npy", f"{v_prefix}/{path1}.npy", path1
        a2, v2, t2 = f"{a_prefix}/{path2}.npy", f"{v_prefix}/{path2}.npy", path2


        a1, v1, t1 = np.load(a1), np.load(v1), text_dict[t1]
        a1, v1, t1 = torch.from_numpy(a1).to(dtype=torch.float), torch.from_numpy(v1).to(dtype=torch.float), f.process_text(t1)
        
        a2, v2, t2 = np.load(a2), np.load(v2), text_dict[t2]
        a2, v2, t2 = torch.from_numpy(a2).to(dtype=torch.float), torch.from_numpy(v2).to(dtype=torch.float), f.process_text(t2)

        a1, v1, t1 = a1.unsqueeze(0), v1.unsqueeze(0), t1.unsqueeze(0)
        a2, v2, t2 = a2.unsqueeze(0), v2.unsqueeze(0), t2.unsqueeze(0)

        a, v, t = torch.cat((a1, a2), dim=0), torch.cat((v1, v2), dim=0), torch.cat((t1, t2), dim=0)
        a, v, t = lava_ptl.encoder.encode_audio(a), lava_ptl.encoder.encode_video(v), lava_ptl.encoder.encode_text(t)
        # a, v, t = lava_ptl.encoder.a_encoder(a), lava_ptl.encoder.v_encoder(v), lava_ptl.encoder.t_encoder(t)
        
        # a1, v1, t1 = lava_ptl.encoder.a_encoder(a1), lava_ptl.encoder.v_encoder(v1), lava_ptl.encoder.t_encoder(t1)
        # a2, v2, t2 = lava_ptl.encoder.a_encoder(a2), lava_ptl.encoder.v_encoder(v2), lava_ptl.encoder.t_encoder(t2)

        # a1, v1, t1 = a1.squeeze(), v1.squeeze(), t1.squeeze()
        # a2, v2, t2 = a2.squeeze(), v2.squeeze(), t2.squeeze()

        a = nn.functional.normalize(a, p=2, dim=-1)
        v = nn.functional.normalize(v, p=2, dim=-1)
        t = nn.functional.normalize(t, p=2, dim=-1)

        a1, a2 = a
        v1, v2 = v
        t1, t2 = t

        print(cosine_similarity(a1, v1))
        print(cosine_similarity(a1, t1))
        print(cosine_similarity(v1, t1))
        print("")

        print(cosine_similarity(a2, v2))
        print(cosine_similarity(a2, t2))
        print(cosine_similarity(v2, t2))
        print("")

        print(cosine_similarity(a1, v2))
        print(cosine_similarity(a1, t2))
        print(cosine_similarity(v1, t2))
        print("")

        print(cosine_similarity(a2, v1))
        print(cosine_similarity(a2, t1))
        print(cosine_similarity(v2, t1))
        print("")

        print(cosine_similarity(a1, a2))
        print(cosine_similarity(v1, v2))
        print(cosine_similarity(t1, t2))

    if mode < 4:

        t1 = nn.functional.normalize(t1, p=2, dim=-1)
        t2 = nn.functional.normalize(t2, p=2, dim=-1)
        t3 = nn.functional.normalize(t3, p=2, dim=-1)

        print(cosine_similarity(t1, t2))
        print(cosine_similarity(t1, t3))
        print(cosine_similarity(t2, t3))

