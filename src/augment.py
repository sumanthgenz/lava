import torch
import torchvision
import torchaudio
import torchtext
import tensorflow as tf
import tensorflow_hub as hub
from torchvision import transforms
import av
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, Process
import os
import warnings
import glob
import pickle
import time
import pytorch_lightning as pl
from PIL import Image

from aai.experimental.sgurram.lava.src.utils import adjust_video_size, nan_filter, pad_spec, get_log_mel_spec
from aai.experimental.sgurram.lava.src.references import mp4_root_dir, downsample_root_dir, npy_root_dir, text_root_dir, visual_vocab_dir
from aai.experimental.ilianherzi.augmented_video_learning.video_transforms import Resize, CenterCrop, RandomCrop, ColorJitter, Flip


def get_lava_features(save_dir=None,
                    mp4_path=None,
                    text="this is a video",
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

def get_video_from_mp4(path, 
                        indir = '/big/davidchan/kinetics/kinetics600/',
                        outdir = '/big/sgurram/kinetics600/',
                        size=256, 
                        frames=16, 
                        fps=10, 
                        center_frames=20,
                        save=True):
    try:

        input_v = av.open(path, 'r')

        v_stream = input_v.streams.video[0]
        input_fps = v_stream.framerate

        video_channels = 3

        vid = np.empty([v_stream.frames, v_stream.height, v_stream.width, video_channels], dtype=np.uint8)

        for idx, frame in enumerate(input_v.decode(video=0)):
            vid[idx] = frame.to_ndarray(format='rgb24')

        input_v.close()

        sampling_rate = int(input_fps/fps)
        vid = vid[::sampling_rate]

        if vid.shape[0] > center_frames:
            m = vid.shape[0]//2
            offset = center_frames//2
            vid = vid[m-offset:m+offset]

        vid = Resize(size)(vid)
        # vid = nan_filter(vid)

        # print(vid.shape)
        if save:
            save_path = path.replace(indir, outdir).replace(".mp4", "").replace(".avi", "")
            dirs = "/".join(save_path.split("/")[:-1])
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            np.save(save_path, vid)
            return {path : save_path}
        return vid
    except:
        if save:
            return {path : ""}
        return None

def get_audio_from_mp4(path, 
                        indir = '/big/sgurram/kinetics/train',
                        outdir = '/big/sgurram/kinetics_features/train/audio',
                        save=False):

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
    # aud = pad_spec(aud)
    aud = nan_filter(aud)

    #aud shape: [M x T], where M = 128, T = 2048
    if save:
        save_path = path.replace(indir, outdir).replace("mp4", "").replace("avi", "")
        np.save(save_path, aud)
        return {path : save_path}
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
    a_filename = f'{npy_root_dir}/{datatype}/audio/{filename}.npy'
    v_filename = f'{npy_root_dir}/{datatype}/video/{filename}.npy'

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
    root = '{}/{}/*.mp4'.format(downsample_root_dir, prefix)

    for path in glob.glob(root):
        files.append(path)

    cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(cores) as p:
        list(tqdm(p.imap_unordered(numpy_process, files), total=len(files)))

def get_npy_paths(prefix, view_progress=True, pretrained_text=False):
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


    root = f"{mp4_root_dir}/kinetics_{prefix}_clipped/*.mp4"

    a_root = f"{npy_root_dir}/{prefix}/audio/"
    v_root = f"{npy_root_dir}/{prefix}/video/"
    t_root = f"{text_root_dir}/kinetics_{prefix}_numpy/"

    paths = tqdm(glob.glob(root)) if view_progress else glob.glob(root)

    if pretrained_text:
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
    else:
        for path in paths:
            name = path.split('clipped/')[-1][:-4]
            filename = name + '.npy'
            a_path = a_root + filename
            v_path = v_root + filename
            t_path = t_root + filename
            exists = [os.path.isfile(a_path), os.path.isfile(v_path), os.path.isfile(t_path)]
            if all(exists):
                a_paths.append(a_path)
                v_paths.append(v_path)
                t_paths.append(name)

    return a_paths, v_paths, t_paths

def augment_video(vid, 
                    size=224, 
                    frames=16,
                    eval=False, 
                    time_crop=True, 
                    space_crop=True, 
                    hflip=True, 
                    color_jitter=False):

    t, h, w, c = vid.shape

    if eval:
        s, e = (t//2 - frames//2), (t//2 + frames//2)
        vid = torch.from_numpy(vid[s:e]).permute(0, 3, 1, 2)
        vid = transforms.RandomCrop(size)(vid)

        if vid.shape != (16, 3, size, size):
            empty = torch.zeros((16, 3, size, size), dtype=vid.dtype)
            empty[:vid.shape[0], :, :, :] = vid
            vid = empty

        return vid.permute(0, 2, 3, 1)

    if time_crop:
        try:
            t_idx = np.random.randint(0, t - frames)
        except:
            t_idx = 0
    else:
        t_idx = 0

    vid = torch.from_numpy(vid[t_idx: t_idx+frames]).permute(0, 3, 1, 2)

    augment = transforms.Compose([
                    transforms.RandomCrop(size),
                    transforms.RandomHorizontalFlip(),
                    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                    # transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
                    ])

    vid = augment(vid)

    if vid.shape != (16, 3, size, size):
            empty = torch.zeros((16, 3, size, size), dtype=vid.dtype)
            empty[:vid.shape[0], :, :, :] = vid
            vid = empty

    # if vid.shape[0] < 16:
    #     empty = np.zeros((16, 128, 128, 3), dtype=vid.dtype)
    #     empty[:vid.shape[0], :, :, :] = vid
    #     vid = empty
    # elif vid.shape[0] < 16:
    #     vid = vid[:16]
    # v = vid.permute(0,2,3,1)
    # f, arr = plt.subplots(1,2)
    # arr[0].imshow(v[5])
    # arr[1].imshow(v[10])
    # plt.savefig('sanity_checks/aug')

    vid = nan_filter(vid)
    return vid.permute(0, 2, 3, 1)

def augment_visual_vocab(
                    path, 
                    size=128,):

    img = torch.from_numpy(np.array(Image.open(path)))
    if len(img.shape) < 3:
        img = torch.stack([img,img,img])
        return transforms.CenterCrop(size)(img).permute(1, 2, 0)
    return transforms.CenterCrop(size)(img.permute(2, 0, 1)).permute(1, 2, 0)

def augment_video_for_visual_vocab(
                    vid, 
                    frames=4,
                    size=128,):


    t, h, w, c = vid.shape
    s, e = (t//2 - frames//2), (t//2 + frames//2)
    vid = vid[s:e].permute(0, 3, 1, 2)
    vid = transforms.RandomCrop(size)(vid)
    return vid.permute(0, 2, 3, 1)

def augment_audio(aud, gaussian_noise=True, target_len=512):
    # d, t = aud.shape
    # start_idx = int(t*start)
    # end_idx = start_idx + time

    # aud = aud[:, start_idx:end_idx]

    if gaussian_noise:
        aud = aud + np.random.normal(loc=0, scale=0.01, size=aud.shape)
    
    if aud.shape[1] > 512:
        aud = aud[:, :target_len]
    aud = torch.from_numpy(aud)
    aud = nan_filter(aud)
    return aud

def augment_cvrl(vid, 
                    size=224, 
                    frames=16,
                    eval=False, 
                    space_crop=True, 
                    hflip=True, 
                    color_jitter=True):
    t, h, w, c = vid.shape
    interval = np.random.randint(0, 1 + t//2)
    try:
        t1 = np.random.randint(0, t - interval - frames)
    except:
        t1 = 0
    t2 = t1 + interval

    v1 = torch.from_numpy(vid[t1: t1 + frames]).permute(0, 3, 1, 2)
    v2 = torch.from_numpy(vid[t2: t2 + frames]).permute(0, 3, 1, 2)


    augment = transforms.Compose(
                        [transforms.RandomCrop(size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                        # transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
    ])

    if not eval:
        v1, v2 = augment(v1), augment(v2) 
        # f, arr = plt.subplots(2,4)
        # arr[0,0].imshow(v1[0].permute(1,2,0))
        # arr[0,1].imshow(v1[3].permute(1,2,0))
        # arr[0,2].imshow(v1[6].permute(1,2,0))
        # arr[0,3].imshow(v1[9].permute(1,2,0))
        # arr[1,0].imshow(v2[0].permute(1,2,0))
        # arr[1,1].imshow(v2[3].permute(1,2,0))
        # arr[1,2].imshow(v2[6].permute(1,2,0))
        # arr[1,3].imshow(v2[9].permute(1,2,0))
        # plt.savefig('sanity_checks/cvrl_augmentation')
        # return 1/0
    else:
        center_crop = transforms.CenterCrop(size)
        v1, v2 = center_crop(v1), center_crop(v2)
    if v1.shape != (16, 3, size, size):
            empty = torch.zeros((16, 3, size, size), dtype=v1.dtype)
            empty[:v1.shape[0], :, :, :] = v1
            v1 = empty
    if v2.shape != (16, 3, size, size):
            empty = torch.zeros((16, 3, size, size), dtype=v2.dtype)
            empty[:v2.shape[0], :, :, :] = v2
            v2 = empty

    v1 = nan_filter(v1)
    v2 = nan_filter(v2)
    return v1, v2


def augment_views(vid, 
                    size=128, 
                    frames=16,
                    eval=False, 
                    space_crop=True, 
                    hflip=True, 
                    color_jitter=True):
    t, h, w, c = vid.shape

    try:
        t1, t2 = np.random.randint(0, t-frames), np.random.randint(0, t-frames)
        t1, t2 = min(t1, t2), max(t1, t2)
    except:
        t1, t2 = 0, 0

    v1 = torch.from_numpy(vid[t1: t1 + frames]).permute(0, 3, 1, 2)
    v2 = torch.from_numpy(vid[t2: t2 + frames]).permute(0, 3, 1, 2)

    augment = transforms.Compose([transforms.RandomCrop(128),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.126, contrast=0.4, saturation=0.4, hue=0.4),
                        transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
            ])

    if not eval:
        v1, v2 = augment(v1), augment(v2) 
    else:
        center_crop = transforms.CenterCrop(128)
        v1, v2 = center_crop(v1), center_crop(v2)
    if v1.shape != (16, 3, 128, 128):
            empty = torch.zeros((16, 3, 128, 128), dtype=v1.dtype)
            empty[:v1.shape[0], :, :, :] = v1
            v1 = empty
    if v2.shape != (16, 3, 128, 128):
            empty = torch.zeros((16, 3, 128, 128), dtype=v2.dtype)
            empty[:v2.shape[0], :, :, :] = v2
            v2 = empty
    v1, v2 = v1.permute(0, 2, 3, 1), v2.permute(0, 2, 3, 1)
    return v1, v2

def parallel_feature_extractor(fn, 
                                root = '/big/sgurram/kinetics600/*/*/*.mp4',
                                prefix="val", 
                                num_workers=4):
        # with open(f'src/{prefix}_samples.pickle', 'rb') as handle:
        #         master_dict = pickle.load(handle)
        
        all_links = glob.glob(root)[:]
        # links = [a for a in all_links if a not in master_dict]

        links = all_links
        n = len(links)

        if n == 0:
            print("waiting for ffmpeg to gimme some more vids")
            time.sleep(100)
            return len(list(master_dict.keys()))

        with Pool(num_workers) as p:
            result = list(tqdm(p.imap_unordered(fn, links), total=n))


def poll(root):
    target = len(glob.glob(root)[:])
    print(target)
    count = 0
    while count < target:
        count = parallel_feature_extractor(get_video_from_mp4, root=root) 
        break

if __name__=="__main__":
    # path = "/big/sgurram/kinetics_res_features/train/video/08xOMOFezjI.npy"
    # vid = np.load(path)
    # print(vid.shape)
    # plt.imshow(vid[5])
    # plt.savefig('sanity_checks/ffmpeg_sanity')
    # poll(root='/big/davidchan/kinetics/kinetics600/*/*/*.mp4')

    # augment_visual_vocab(root=visual_vocab_dir, outdir="")

    words = [subdir.split("/")[-1] for subdir in list(glob.glob(f'{visual_vocab_dir}/*'))]
    print(words)
    print(len(words))
