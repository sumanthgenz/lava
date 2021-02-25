import torch
import torchvision
import torchaudio
import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import av

import os
import warnings
import glob

from aai.experimental.sgurram.lava.src.utils import adjust_video_size, nan_filter, pad_spec, get_log_mel_spec
from aai.experimental.sgurram.lava.src.metrics import cosine_similarity

torchaudio.set_audio_backend("sox_io")
os.environ["IMAGEIO_FFMPEG_EXE"] = "/home/sgurram/anaconda3/bin/ffmpeg"
warnings.filterwarnings("ignore")

def get_lava_features(save_dir=None,
                    mp4_path=None,
                    text="default",
                    wav_path=None,
                    save=False,
                    guse_model=None):

    v = get_video_from_mp4(mp4_path)

    if wav_path:
        a = get_audio_from_wav(wav_path)
    else:
        a = get_audio_from_mp4(mp4_path)

    if text:
        t = guse_model([text])

    filename = (mp4_path.split('/')[-1]).split('.')[0]
    print(filename)

    if save:
        filename = (mp4_path.split('/')[-1]).split('.')[0]
        a_path = '{}/audio/{}'.format(save_dir, filename)
        v_path = '{}/video/{}'.format(save_dir, filename)
        t_path = '{}/text/{}'.format(save_dir, filename)

        np.save(v_path, torch.from_numpy(v))
        if a:
            np.save(a_path, torch.from_numpy(a))
        if t:
            np.save(t_path, t.numpy())

    return a, v, t

def get_video_from_mp4(path):
    input_v = av.open(path, 'r')

    v_stream = input_v.streams.video[0]

    video_channels = 3

    vid = np.empty([v_stream.frames, v_stream.height, v_stream.width, video_channels], dtype=np.uint8)

    for idx, frame in enumerate(input_v.decode(video=0)):
        vid[idx] = frame.to_ndarray(format='rgb24')

    input_v.close()

    vid = torch.from_numpy(vid)
    vid = adjust_video_size(vid)

    vid = nan_filter(vid)

    #vid shape: [T x H x W x C], where T = 16, H = W = 128, C = 3
    return vid

def get_audio_from_mp4(path):
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

    aud = get_log_mel_spec(torch.flatten(torch.from_numpy(aud).mean(dim=1).type(dtype=torch.float32)))
    aud = pad_spec(aud)
    aud = nan_filter(aud)

    #aud shape: [M x T], where M = 128, T = 2048
    return aud

def get_audio_from_wav(path):
    try:
        wave, samp_freq = torchaudio.load(path)
    except:
        return None
    spec = torchaudio.transforms.MelSpectrogram()(wave)
    spec = spec.log2().mean(dim=0).squeeze() #avg across channels dimensions
    spec = nan_filter(pad_spec(spec))
    return spec

def get_audiovisual(path):
    input_a = av.open(path, 'r')
    input_v = av.open(path, 'r')

    a_stream = input_a.streams.audio[0]
    v_stream = input_v.streams.video[0]

    audio_channels = 2
    video_channels = 3

    vid = np.empty([v_stream.frames, v_stream.height, v_stream.width, video_channels], dtype=np.uint8)
    aud = np.empty([a_stream.frames, audio_channels, a_stream.frame_size])

    for idx, frame in enumerate(input_a.decode(audio=0)):
        aud_frame = frame.to_ndarray(format='sp32')
        pad =  a_stream.frame_size - aud_frame.shape[-1]
        if pad > 0:
            aud[idx] =  np.pad(aud_frame, pad_width=[(0,0),(0, pad)])
        else:
            aud[idx] = aud_frame[:2]

    for idx, frame in enumerate(input_v.decode(video=0)):
        vid[idx] = frame.to_ndarray(format='rgb24')

    input_a.close()
    input_v.close()

    aud = get_log_mel_spec(torch.flatten(torch.from_numpy(aud).mean(dim=1).type(dtype=torch.float32)))
    aud = pad_spec(aud)

    vid = torch.from_numpy(vid)
    vid = adjust_video_size(vid)

    aud = nan_filter(aud)
    vid = nan_filter(vid)

    #aud shape: [M x T], where M = 128, T = 2048
    #vid shape: [T x H x W x C], where T = 16, H = W = 128, C = 3
    return aud, vid

def numpy_processing(path):
    try:
        a, v = get_audiovisual(path)

    except:
        print("Skipped: ", path)
        return

    datatype = path.split('/')[-2]
    filename = path.split('{}/'.format(datatype))[-1][:-4]
    a_filename = '/big/sgurram/kinetics_numpy/{}/audio/{}.npy'.format(datatype, filename)
    v_filename = '/big/sgurram/kinetics_numpy/{}/video/{}.npy'.format(datatype, filename)

    np.save(a_filename, a)
    np.save(v_filename, v)
    return a_filename, v_filename


def save_npy_files(prefix):
    """
    Args:
        prefix: (str) specifies type of data [train, val, test]
        view_progres: (bool) if True, use tqdm for progress (default False)
    Return:
        a_paths: (list -> str) paths to audio feature npy files
        v_paths: (list -> str) paths to video feature npy files
        t_paths: (list -> str) paths to text feature npy files

    Fetch paths to audio, video, text feature files for a given sample,
    if all 3 feature paths exist.
    """

    files = []
    prefix = 'val' if prefix=='val' else 'train'
    root = '/big/sgurram/kinetics_downsample/{}/*.mp4'.format(prefix)

    for path in glob.glob(root):
        files.append(path)

    cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(cores) as p:
        list(tqdm(p.imap_unordered(numpy_process, files), total=len(files)))

def get_npy_paths(prefix, view_progress=False):
    """
    Args:
        prefix: (str) specifies type of data [train, val, test]
        view_progres: (bool) if True, use tqdm for progress (default False)
    Return:
        a_paths: (list -> str) paths to audio feature npy files
        v_paths: (list -> str) paths to video feature npy files
        t_paths: (list -> str) paths to text feature npy files

    Fetch paths to audio, video, text feature files for a given sample,
    if all 3 feature paths exist.
    """
    a_paths, v_paths, t_paths = [], [], []
    count = 0


    root = '/big/davidchan/kinetics/kinetics_{}_clipped/*.mp4'.format(prefix)

    a_root = '/big/sgurram/kinetics_numpy/{}/audio/'.format(prefix)
    v_root = '/big/sgurram/kinetics_numpy/{}/video/'.format(prefix)
    if prefix == 'val':
        prefix = 'e' + prefix
    t_root = '/big/afang/kinetics_{}_numpy/'.format(prefix)

    paths = tqdm(glob.glob(root)) if view_progress else glob.glob(root)

    for path in paths:
        filename = path.split('clipped/')[-1][:-4] + '.npy'
        a_path = a_root + filename
        v_path = v_root + filename
        t_path = t_root + filename
        exists = [os.path.isfile(a_path), os.path.isfile(v_path), os.path.isfile(t_path)]
        if all(exists):
            a_paths.append(a_path)
            v_paths.append(v_path)
            t_paths.append(t_path)

    return a_paths, v_paths, t_paths


if __name__ == '__main__':

    # Code below is for experimenting with the methods above

    # Load filepaths for Kinetics Validation set
    dir = "/big/davidchan/kinetics/kinetics_val_clipped"
    file_paths = []
    for path in glob.glob(f'{dir}/*.mp4'):
        file_paths.append(path)

    path = "/big/kinetics_audio/train/25_riding a bike/0->--JMdI8PKvsc.wav"
    # print(get_audio_from_wav(path))

    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    a, v, t = get_lava_features(mp4_path=file_paths[4],
                            wav_path=path,
                            guse_model=embed)
    print(a.shape)
    print(v.shape)
    print(t.shape)

    positive = ["this is my pet, a friendly little dog", "this is my pet, a friendly little puppy"]
    negative = ["this is my pet, a friendly little dog", "launch the rocket to reach low-earth orbit and begin re-entry into the atmosphere"]

    pos = embed(positive)
    neg = embed(negative)

    p1, p2 = pos
    n1, n2 = neg
    print(cosine_similarity(p1, p2))
    print(cosine_similarity(n1, n2))

    '''Evaluate Audiovisual Feature Extraction'''
    # for path in tqdm(file_paths[:5]):
    #     a, v = get_audiovisual(path)
    #     print("Audio Features Shape: ", a.shape)
    #     print("Video Features Shape: ", v.shape)

    '''Verify Audiovisual Numpy Feature Files'''
    # for path in tqdm(file_paths[:5]):
    #     a_file, v_file = numpy_processing(path)
    #     print("Audio Features Shape: ", np.load(a_file).shape)
    #     print("Video Features Shape: ", np.load(v_file).shape)

    '''Visualize Downsampled Video Frame'''
    # path = file_paths[0]
    # _, v = get_audiovisual(filepath)
    # img = Image.fromarray(v[0], 'RGB')
    # img.save('visualize_video_frame.png')

    '''Visualize Spectrogram at Given Sampling Rate'''
    # path = file_paths[0]
    # a, _ = get_audiovisual(filepath)
    # plt.savefig("visualize_spec.png")
