import torch
import torchvision
import torchaudio
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

from aai.experimental.sgurram.lava.src.encoder import LAVA
from aai.experimental.sgurram.lava.src.utils import get_src_conditional_mask, position_embed, position_embed_3d
from aai.experimental.sgurram.lava.src.metrics import cosine_similarity

class LAVAFeatures():

    def __init__(self,
                 lava_model_path="/home/sgurram/Desktop/video_lava/lava/2hq4bjww/checkpoints/epoch=15.ckpt",
                 guse_model_path="https://tfhub.dev/google/universal-sentence-encoder/4"):

        self.guse_model_path = guse_model_path
        self.lava_model_path = lava_model_path

        self.guse_model = hub.load(self.guse_model_path)
        self.lava_model = LAVA()
        self.lava_model.load_state_dict(torch.load(self.lava_model_path, map_location='cpu'), strict=False)
        self.lava_model
        self.lava_model.eval()

        self.a_feature_model = self.lava_model._audio_feature_model
        self.a_projection = self.lava_model._audio_input_projection
        self.a_encoder = self.lava_model._audio_encoder

        self.v_feature_model = self.lava_model._video_feature_model
        self.v_projection = self.lava_model._video_input_projection
        self.v_encoder = self.lava_model._video_encoder

        self.t_feature_model = self.lava_model._text_feature_model
        self.t_projection = self.lava_model._text_input_projection

        self.feature_dimension = self.lava_model._feature_dimension
        self.model_dimension = self.lava_model._model_dimension

    def get_lava_features(self,
                    mp4_path=None,
                    text_input=None,
                    wav_path=None,
                    save=False,
                    save_dir=None,):

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

        audio, video, text = None, None, None

        # loading data for audio, video, text (if any)
        if wav_path:
            audio = self.get_audio_from_wav(wav_path)
        else:
            audio = self.get_audio_from_mp4(mp4_path)

        if mp4_path:
            video = self.get_video_from_mp4(mp4_path)

        if text_input:
            text = torch.from_numpy(self.guse_model([text_input]).numpy())

        a, v, t = None, None, None

        # encoding data with lava to get features (if any)
        with torch.no_grad():
            if audio is not None:
                a = self.encode_audio(audio.unsqueeze(0)).squeeze().detach().cpu().numpy()
            if video is not None:
                v = self.encode_video(video.to(dtype=torch.float).unsqueeze(0)).squeeze().detach().cpu().numpy()
            if text is not None:
                t = self.encode_text(text).detach().cpu().numpy()
        if save:
            filename = (mp4_path.split('/')[-1]).split('.')[0]
            a_path = '{}/audio'.format(save_dir)
            v_path = '{}/video'.format(save_dir)
            t_path = '{}/text'.format(save_dir)

            self.save_npy_file(features=a, dir=a_path, filename=filename) if a is not None else None
            self.save_npy_file(features=v, dir=v_path, filename=filename) if v is not None else None
            self.save_npy_file(features=t, dir=t_path, filename=filename) if t is not None else None

        return a, v, t

    def encode_audio(self, x):
        with torch.no_grad():
            x = self.a_feature_model(x)
            x = self.a_projection(x.reshape(-1, self.feature_dimension)).reshape(
                    x.shape[0], x.shape[1], self.model_dimension)

            x = self.a_encoder(src=x).mean(dim=1)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x


    def encode_video(self, x):
        with torch.no_grad():
            x = self.v_feature_model(x)
            x = self.v_projection(x.reshape(-1, self.feature_dimension)).reshape(
                    x.shape[0], x.shape[1], self.model_dimension)

            x = self.v_encoder(src=x).mean(dim=1)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x

    def encode_text(self, x):
        with torch.no_grad():
            x = self.t_feature_model(x)
            x = self.t_projection(x).squeeze()

        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x

    def remove_batchnorm(self, model, layer_idx):
        return torch.nn.Sequential(*(list(model.children())[:layer_idx] + list(model.children())[layer_idx+1:]))

    def get_video_from_mp4(self, path):
        input_v = av.open(path, 'r')

        v_stream = input_v.streams.video[0]

        video_channels = 3

        vid = np.empty([v_stream.frames, v_stream.height, v_stream.width, video_channels], dtype=np.uint8)

        for idx, frame in enumerate(input_v.decode(video=0)):
            vid[idx] = frame.to_ndarray(format='rgb24')

        input_v.close()

        vid = torch.from_numpy(vid)
        vid = self.adjust_video_size(vid)

        vid = self.nan_filter(vid)

        #vid shape: [T x H x W x C], where T = 16, H = W = 128, C = 3
        return vid

    def get_audio_from_mp4(self, path):
        input_a = av.open(path, 'r')

        try:
            a_stream = input_a.streams.audio[0]
        except:
            return None


        audio_channels = 2
        aud = np.empty([a_stream.frames, audio_channels, a_stream.frame_size])

        for idx, frame in enumerate(input_a.decode(audio=0)):
            aud_frame = frame.to_ndarray(format='sp32')
            pad =  a_stream.frame_size - aud_frame.shape[-1]
            if pad > 0:
                aud[idx] =  np.pad(aud_frame, pad_width=[(0,0),(0, pad)])
            else:
                aud[idx] = aud_frame[:2]

        input_a.close()

        aud = self.get_log_mel_spec(torch.flatten(torch.from_numpy(aud).mean(dim=1).type(dtype=torch.float32)))
        aud = self.pad_spec(aud)
        aud = self.nan_filter(aud)

        #aud shape: [M x T], where M = 128, T = 2048
        return aud

    def get_audio_from_wav(self, path):
        try:
            wave, samp_freq = torchaudio.load(path)
        except:
            return None
        spec = torchaudio.transforms.MelSpectrogram()(wave)
        spec = spec.log2().mean(dim=0).squeeze() #avg across channels dimensions
        spec = self.nan_filter(self.pad_spec(spec))
        return spec

    def nan_filter(self, input):
        """
        Args:
            input: (torch.Tensor) spectrogram with shape [T * M] -> T = timesteps, M = mel bins
        Return:
            input: (torch.Tensor) spectrogram with shape [pad_len * M]
        Filter out inf and NaN values from tensor
        """
        input[torch.isinf(input)] = 0
        input[torch.isnan(input)] = 0
        return input

    def adjust_video_size(self, video_frames, target_size=128, seq_len=16):
        """
        Args:
            video_frames: (np.array) input video with shape [T * H * W * C] ->
                                                    T = timesteps, H = height, W = width, C = channels
            target_size: (int) output dimension for H and W
            seq_len: (int) output dimension for T
        Return:
            video_frames: (np.array) output video with desired output shape
        Subsample videoframes and clip to seq_len frames (pad if T < seq_len). Crop frames from top-right corner to target_size (pad if needed).
        """
        subsample_rate = max(1, int(video_frames.shape[0]) // seq_len)
        video_frames = video_frames[::subsample_rate][:seq_len]
        if video_frames.shape[1] != target_size or video_frames.shape[2] != target_size:
            video_frames = video_frames[:, :target_size, :target_size, :]
            w_pad, h_pad = target_size - int(video_frames.shape[1]), target_size - int(video_frames.shape[2])
            video_frames =  torch.nn.functional.pad(video_frames, pad=((0,0, 0,w_pad, 0,h_pad, 0,0)))
        if video_frames.shape[0] < seq_len:
            padding = torch.zeros(seq_len - video_frames.shape[0], target_size, target_size, 3)
            video_frames =  torch.cat((video_frames, padding))
        return video_frames

    def pad_spec(self, spec, pad_len=2048):
        """
        Args:
            spec: (np.array) spectrogram with shape [T * M] -> T = timesteps, M = mel bins
            pad_len: (int) max. sequence length (timesteps) for spectrogram
        Return:
            spec: (np.array) spectrogram with shape [pad_len * M]
        Zero-pad spectrogram to desired sequence length
        """
        to_pad = pad_len - spec.shape[-1]
        if to_pad > 0:
            return torch.nn.functional.pad(spec, (0,to_pad,0,0))
        return spec[:,:pad_len]

    def get_log_mel_spec(self, wave, samp_freq=16000):
        wave = torch.unsqueeze(wave, 0)
        spec = torchaudio.transforms.MelSpectrogram()(wave)
        return spec.log2()[0,:,:]

    def save_npy_file(self, features, dir, filename):
            if not os.path.exists(dir):
                os.makedirs(dir)
            path = "{}/{}".format(dir, filename)
            np.save(path, features.detach().numpy())


if __name__ == "__main__":

    dir = "/big/davidchan/kinetics/kinetics_val_clipped"
    file_paths = []
    for path in glob.glob(f'{dir}/*.mp4'):
        file_paths.append(path)

    model_path = "/home/sgurram/Desktop/video_lava/lava/3l5t28v4/checkpoints/epoch=4.ckpt"
    # model_path = "/home/sgurram/Desktop/video_lava/lava/r29koav/checkpoints/epoch=0.ckpt"

    lf = LAVAFeatures(lava_model_path=model_path)
    
    idx1 = 8282
    idx2 = 56

    # wav_path = "/big/kinetics_audio/train/25_riding a bike/3->-merlXJp4m4c.wav"
    # mp4_path = "/big/sgurram/kinetics_downsample/val/CIRUvo_OGv4.mp4"
    # npy_path = "/big/sgurram/kinetics_numpy/val/video/CIRUvo_OGv4.npy"

    # path1 = "/big/sgurram/kinetics_downsample/val/CIRUvo_OGv4.mp4"
    # path2 = "/big/sgurram/kinetics_downsample/train/-JMdI8PKvsc.mp4"	             
    # path1 = "/big/sgurram/kinetics_downsample/train/L3lKNeY5mIs.mp4"	   
    # path2 = "/big/sgurram/kinetics_downsample/val/Zp44Wj7soCE.mp4"	 
    # path2 = "/big/sgurram/kinetics_downsample/val/oTCio7AcabE.mp4"


    # for _ in tqdm(range(25)):
    #     a1, v1, t1 = lf.get_lava_features(mp4_path=mp4_path, text_input="this is a video of a dog")

    a1, v1, t1 = lf.get_lava_features(mp4_path=file_paths[idx1], text_input="this is a video")
    a1, v1, t1 = torch.from_numpy(a1), torch.from_numpy(v1), torch.from_numpy(t1)

    a2, v2, t2 = lf.get_lava_features(mp4_path=file_paths[idx2], text_input="this is a video")
    a2, v2, t2 = torch.from_numpy(a2), torch.from_numpy(v2), torch.from_numpy(t2)

    sample_1 = torch.stack((a1, v1, t1))
    sample_2 = torch.stack((a2, v2, t2))

    # print(a1.shape)
    # print(v1.shape)
    # print(t1.shape)

    print("Sample 1 Pairwise Modal Comparisons")
    print(cosine_similarity(a1, v1))
    # print(cosine_similarity(a1, t1))
    # print(cosine_similarity(v1, t1))
    print("Sample 2 Pairwise Modal Comparisons")
    print(cosine_similarity(a2, v2))
    # print(cosine_similarity(a2, t2))
    # print(cosine_similarity(v2, t2))
    print("Sample 1 and 2 Different Modal Comparisons")
    print(cosine_similarity(a1, v2))
    print(cosine_similarity(a2, v1))
    # print(cosine_similarity(v1, t2))
    # print(cosine_similarity(t1, a2))
    print("Sample 1 and 2 Same Modal Comparisons")
    print(cosine_similarity(a1, a2))
    print(cosine_similarity(v1, v2))
    # print(cosine_similarity(t1, t2))

    # print(sample_1.shape)
    # print(sample_2.shape)
    # print(torch.mm(sample_1, sample_1.t()))
    # print(torch.mm(sample_2, sample_2.t()))
    # print(torch.mm(sample_1, sample_2.t()))

    # print(v1)
    # print(v2)

    # v_features = torch.from_numpy(np.load(npy_path)).to(dtype=torch.float)

    # v = lf._video_feature_model(torch.stack((v_features, v_features)))
    # print(v.shape)

    # x = torch.arange(64).reshape((2, 2, 4, 4))
    # print(x)
    # print(x.reshape((-1, 4, 4)))
    # print(x.reshape((-1, 2, 4, 4)))

    # x = torch.rand((2, 16, 4, 4, 2))
    # print(position_embed_3d(x).shape)

    # y = torch.rand((2, 128, 2048))
    # print(position_embed(y).shape)