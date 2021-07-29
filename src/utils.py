import cv2
import av
import torch
import torchaudio
import torchvision
from torchvision import transforms
import torchtext
import numpy as np
from typing import Tuple, Optional
import warnings
import pickle
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
import pandas as pd
import tqdm
from tqdm import tqdm
import os
from collections import Counter

from torchtext.data.functional import generate_sp_model, load_sp_model, sentencepiece_numericalizer
from aai.experimental.sgurram.lava.src.references import mp4_root_dir, pickle_root_dir, raw_text_dir, npy_root_dir, text_root_dir, k600_path
from aai.experimental.ilianherzi.augmented_video_learning.video_transforms import Resize, CenterCrop, RandomCrop, ColorJitter, Flip
from aai.alexandria.metrics.caption.spice.spice import *
from aai.alexandria.metrics.caption.cider.cider import *
from aai.alexandria.metrics.caption.meteor.meteor import *
from aai.alexandria.metrics.caption.rouge.rouge import *

torchaudio.set_audio_backend("sox_io") 
warnings.filterwarnings("ignore")

def attention(q, k, v, d):
    qk = torch.matmul(q, k.permute(0, 2, 1))
    qk *= d ** -0.5
    qk = qk.softmax(dim=-1)
    qkv = torch.matmul(qk, v)
    return qkv

def nan_filter(input, warning=True):
    """
    Args:
        input: (torch.Tensor) spectrogram with shape [T * M] -> T = timesteps, M = mel bins 
    Return:
        input: (torch.Tensor) spectrogram with shape [pad_len * M]
    Filter out inf and NaN values from tensor
    """

    # if warning and (any(torch.isnan(input)) or any(torch.isinf(input))):
    #     print("Warning, the tensor has NaN/inf ")

    input[torch.isinf(input)] = 0
    input[torch.isnan(input)] = 0
    return input

def is_nan(input):
    return (any(torch.isnan(input.flatten())) or any(torch.isinf(input.flatten())))

def adjust_video_size(video_frames, target_size=128, seq_len=16):
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
    r = Resize(128)
    cc = RandomCrop()
    video_frames = cc(r(video_frames))
    video_frames = torch.from_numpy(video_frames)

    if video_frames.shape[1] != target_size or video_frames.shape[2] != target_size:
        video_frames = video_frames[:, :target_size, :target_size, :]
        w_pad, h_pad = target_size - int(video_frames.shape[1]), target_size - int(video_frames.shape[2])
        video_frames =  torch.nn.functional.pad(video_frames, pad=((0,0, 0,w_pad, 0,h_pad, 0,0)))
    if video_frames.shape[0] < seq_len:
        padding = torch.zeros(seq_len - video_frames.shape[0], target_size, target_size, 3)
        video_frames =  torch.cat((video_frames, padding))
    # plt.imshow(video_frames[15])
    # plt.savefig('src/img_sanity')
    # print(video_frames.shape)
    return video_frames

def pad_spec(spec, pad_len=2048):
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

def pad_batch(x, batch_dim):
    to_pad = batch_dim - x.shape[0]
    padding = torch.zeros(to_pad, *x.shape[1:]).to(x.dtype)
    x = torch.cat((x, padding.to(device=x.device)))
    return x

def pad_video(x, frame_dim):
    to_pad = frame_dim - x.shape[0]
    padding = torch.zeros(to_pad, *x.shape[1:]).to(x.dtype)
    x = torch.cat((x, padding.to(device=x.device)))
    return x

def get_log_mel_spec(wave, samp_freq=48000, n_bins=80, target_len=512):
    # pad = target_len - 
    wave = torch.unsqueeze(wave, 0)
    # spec = torchaudio.transforms.MelSpectrogram(n_mels=128)(wave)
    spec = torchaudio.transforms.MelSpectrogram(n_fft=1730, win_length=1730, hop_length=1730//2, n_mels=n_bins, pad=10)(wave)
    spec = spec.log2()[0,:,:]
    n = spec.shape[1]
    if target_len > n:
        start = (target_len - spec.shape[1])//2
        final_spec = torch.zeros(n_bins, target_len)
        final_spec[:, start:start+n] = spec
        return final_spec
    return spec

def get_src_conditional_mask(max_sequence_length):
    """
    Args:
        max_sequence_length: (int) the length of the longest sequence in a batch
    Return:
        mask: (torch.Tensor) '-inf' and '0' conditional mask 
    """
    mask = (torch.triu(torch.ones(max_sequence_length, max_sequence_length)) == 1).transpose(0, 1)
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

def get_mp4_paths(root=mp4_root_dir):
    """
    Args:
        root: path to root of the Kinetics mp4 files
    Return:
        None: 
    Save the filepaths for the dataset to a txt file
    """
    file_paths = []
    for path in tqdm(glob.glob('{}/kinetics_train_clipped/*.mp4'.format(root))):
        file_paths.append(path.split('kinetics/')[-1]+'\n')

    for path in tqdm(glob.glob('{}/kinetics_val_clipped/*.mp4'.format(root))):
        file_paths.append(path.split('kinetics/')[-1]+'\n')

    f = open("Desktop/kinetics.txt", "a")
    f.writelines(file_paths)
    f.close()

def create_kinetics_labels():
    classes = []

    train = pickle.load(open("{}/kinetics_train.pickle".format(picke_root_dir), "rb"))
    val = pickle.load(open("{}/kinetics_validate.pickle".format(picke_root_dir), "rb"))
    # test = pickle.load(open("{}/kinetics_test.pickle".format(picke_root_dir), "rb"))

    for key in train:
        if train[key][1] not in classes:
            classes.append(train[key][1])

    for key in train:
        class_name = train[key][1]
        train[key] = [class_name, classes.index(class_name)]

    for key in val:
        class_name = val[key][1]
        val[key] = [class_name, classes.index(class_name)]

    pickle.dump(train, open("{}/kinetics_train.pickle".format(picke_root_dir), "wb"))
    pickle.dump(val, open("{}/kinetics_val.pickle".format(picke_root_dir), "wb"))
    pickle.dump(classes, open("{}/kinetics_700_classes.pickle".format(picke_root_dir), "wb"))

    return train, val

    
# def get_kinetics_labels(datatype, path, text=False):

def get_kinetics_labels(prefix="train"):

    data = pickle.load(open("{}/{}.pickle".format(pickle_root_dir, prefix), "rb"))
    root = '{}/kinetics_{}_clipped/*.mp4'.format(mp4_root_dir, prefix)

    present = 0
    total = 0

    for path in glob.glob(root):
        filename = path.split('clipped/')[-1][:-4]
        if filename in data:
            present += 1
        total += 1
    
    print(prefix, present, total)

def generate_guse_embeddings(path):
    """
    Args:
        path: (str) path to root of the jsons containing text captions for each sample
    Return:
        None: 
    Save all the GUSE embeddings to a file (maybe numpy file).
    Would be best to have a way to access embeddings from the sample's URL.
    """
    pass


def _scaled_axis(axis_length, min_scale=1, max_scale=1e4, n_scales=8):
    positions = torch.arange(axis_length).float()
    log_increment = np.log(max_scale / min_scale) / (n_scales - 1)
    inv_scale_increment = min_scale * torch.exp(torch.arange(n_scales).float() * -log_increment)
    scaled_axis = positions.unsqueeze(1) * inv_scale_increment.unsqueeze(0)
    return torch.cat([torch.sin(scaled_axis), torch.cos(scaled_axis)], dim=1)

def position_embed(inputs: torch.Tensor, start: int = 1, concat: bool = False, base: int = 10000) -> torch.Tensor:
    hidden_size = inputs.shape[-1]
    if concat and hidden_size % 2 != 0:
        raise AssertionError('Model hidden size must be even for sinusoidal embedding')
    if hidden_size % 2 != 0:
        hidden_size += 1

    if inputs.is_cuda:
        power = torch.arange(0, hidden_size, 2).cuda() / hidden_size
        seqpos = torch.arange(start, inputs.shape[1] + 1).unsqueeze(0).float().cuda()
    else:
        power = torch.arange(0, hidden_size, 2).float() / hidden_size
        seqpos = torch.arange(start, inputs.shape[1] + 1).unsqueeze(0).float()

    divisor = base**power

    # Compute the sequence positions
    index = seqpos.unsqueeze(-1) / divisor

    sin_embedding = torch.sin(index)
    cos_embedding = torch.cos(index)

    position_embedding = torch.stack([sin_embedding, cos_embedding], dim=-1).reshape(1, inputs.shape[1], hidden_size)

    if concat:
        output = torch.cat([inputs, position_embedding.expand(inputs.shape[0], -1, -1)], dim=-1)
        return output
    return inputs + position_embedding.type_as(inputs)

def position_embed_3d(inputs):
    assert len(inputs.shape) == 5, 'Inputs to 3D position embedding must br a rank 5 tensor'

    # Get the hidden dimension
    batch_size, x_len, y_len, z_len, _ = inputs.shape

    # Get a position embedding for each dimension
    x_embeddings = _scaled_axis(x_len).unsqueeze(1).expand(-1, y_len, -1).unsqueeze(2).expand(-1, -1, z_len, -1)
    y_embeddings = _scaled_axis(y_len).unsqueeze(0).expand(x_len, -1, -1).unsqueeze(2).expand(-1, -1, z_len, -1)
    z_embeddings = _scaled_axis(z_len).unsqueeze(0).expand(x_len, -1, -1).unsqueeze(1).expand(-1, y_len, -1, -1)

    position_embedding = torch.cat([x_embeddings, y_embeddings, z_embeddings], dim=-1)

    # Concatenate the position embeddings to the inputs
    return torch.cat(
        [inputs, position_embedding.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).type_as(inputs)], dim=-1)

def generate_tokenizer(prefix, vocab_path, vocab_size=48000):
    text_dict = pickle.load(open(f"{raw_text_dir}/kinetics_{prefix}.pickle", "rb"))
    vocab_file = open(f"{vocab_path}.txt", 'w+')
    for k in text_dict:
        vocab_file.write(text_dict[k] + '\n')
    vocab_file.close()
    generate_sp_model(filename=f"{vocab_path}.txt", 
                vocab_size=vocab_size, 
                model_type='bpe', 
                model_prefix='lava_sp')

def tokenize_text(sp_model_path, text):
    sp_model = load_sp_model(sp_model_path)
    sp_id_generator = sentencepiece_numericalizer(sp_model)
    return list(sp_id_generator(text))

def process_tags(text, num_tags=7):
    prefix = "A video of"
    t = text.split(prefix)[-1]
    t = t.split(",")[:num_tags]
    random.shuffle(t)
    return f"{prefix} {','.join(t)}"

def visualize_batch(urls, similarity, prefix, qualifier='c', mode='vt'):
    # Based on https://github.com/openai/CLIP/blob/beba48f35392a73c6c47ae67ddffced81ad1916d/notebooks/Interacting_with_CLIP.ipynb
    n = len(urls)

    v = f'{npy_root_dir}/{prefix}/video'
    t = pickle.load(open(f"{raw_text_dir}/kinetics_{prefix}.pickle", "rb"))

    images = [torch.from_numpy(np.load(f'{v}/{u}.npy'))[5, :, :, :] for u in urls]
    texts = [t[u] for u in urls]

    # images[5] = torch.zeros(128, 128, 3)

    plt.figure(figsize=(40, 28))  
    plt.imshow(similarity.cpu(), vmin=-0.1, vmax=1.0)

    plt.yticks(range(n), texts, fontsize=18)
    plt.xticks([])
    for i, image in enumerate(images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=8)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, n - 0.5])
    plt.ylim([n + 0.5, -2])

    plt.title("Cosine similarity between text and video features", size=20)
    plt.savefig(f'{qualifier}_{prefix}_similarity_matrix.png')


def visualize_batch_downstream(similarity, prefix, dataset='ucf', mode='av'):
    # Based on https://github.com/openai/CLIP/blob/beba48f35392a73c6c47ae67ddffced81ad1916d/notebooks/Interacting_with_CLIP.ipynb
    # images[5] = torch.zeros(128, 128, 3)

    n = similarity.shape[0]
    
    plt.figure(figsize=(40, 28))  
    plt.imshow(similarity.cpu(), vmin=-0.1, vmax=1.0)

    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=8)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, n - 0.5])
    plt.ylim([n + 0.5, -2])

    plt.title("Cosine similarity between audio (row) and video features (col)", size=20)
    plt.savefig(f'{dataset}_{prefix}_similarity_matrix.png')

def visualize_batch(urls, similarity, prefix, qualifier='c', mode='vt'):
    # Based on https://github.com/openai/CLIP/blob/beba48f35392a73c6c47ae67ddffced81ad1916d/notebooks/Interacting_with_CLIP.ipynb
    n = len(urls)

    v = f'{npy_root_dir}/{prefix}/video'
    t = pickle.load(open(f"{raw_text_dir}/kinetics_{prefix}.pickle", "rb"))

    images = [torch.from_numpy(np.load(f'{v}/{u}.npy'))[5, :, :, :] for u in urls]
    texts = [t[u] for u in urls]

    # images[5] = torch.zeros(128, 128, 3)

    plt.figure(figsize=(40, 28))  
    plt.imshow(similarity.cpu(), vmin=-0.1, vmax=1.0)

    plt.yticks(range(n), texts, fontsize=18)
    plt.xticks([])
    for i, image in enumerate(images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=8)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, n - 0.5])
    plt.ylim([n + 0.5, -2])

    plt.title("Cosine similarity between text and video features", size=20)
    plt.savefig(f'{qualifier}_{prefix}_similarity_matrix.png')


# https://github.com/doronharitan/human_activity_recognition_LRCN/blob/master/utils_action_recognition.py
def get_confusion_matrix(x, y, classes, path):
    class_nums = list(torch.arange(len(classes)))
    cm = confusion_matrix(x, y, labels=class_nums, normalize='true')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='none', aspect='auto', cmap=plt.cm.Blues)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    plt.xticks(class_nums, classes, rotation=90, fontsize=4)
    plt.yticks(class_nums, classes, fontsize=4)
    plt.ylim(len(class_nums), -0.5)
    plt.title('Normalized confusion matrix')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def get_confusion_dictionary(x, y, classes, path):
    conf_dict = {}
    for i in range(len(x)):
        if y[i] not in conf_dict:
            conf_dict[y[i]] = []
        conf_dict[y[i]].append(x[i])
    
    conf_by_class = {}
    for k in conf_dict:
        class_by_freq = sorted([(c, freq) for c,freq in Counter(conf_dict[k]).items()], key=lambda x: -x[1])
        topk = [classes[x[0]] for x in class_by_freq][:5]
        conf_by_class[classes[k]] = topk
    with open(f'{path}.pickle', 'wb') as handle:
        pickle.dump(conf_by_class, handle, protocol=pickle.HIGHEST_PROTOCOL)

# https://discuss.pytorch.org/t/two-models-with-same-weights-different-results/8918/5
def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

def get_missing_videos(prefix='train'):
    text_dict = pickle.load(open(f"{raw_text_dir}/kinetics_{prefix}.pickle", "rb"))
    count = 0
    missing = []
    for k in text_dict:
        if "this video contains" not in text_dict[k]:
            count += 1
            missing.append(k)
    print(count)
    np.save(f'{prefix}_missing', np.array(missing))


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
    # t_root = f"{text_root_dir}/kinetics_{prefix}_numpy/"
    t_root = f"{text_root_dir}/{prefix}/"

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

def get_k600_paths(prefix, view_progress=True, pretrained_text=False):
    k600 = np.array(pd.read_csv(f'{k600_path}/{prefix}.csv', usecols=["youtube_id"]).values.tolist()).flatten()

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
            if all(exists) and name in k600:
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
            if all(exists) and name in k600:
                a_paths.append(a_path)
                v_paths.append(v_path)
                t_paths.append(name)

    return a_paths, v_paths, t_paths

def text_similarity(i, j):
    method = Rouge()
    return method.calc_score(i,j)

def visualize_dataset_slice(root='/big/sgurram/HMDB-raw-npy'):
    for subdir in glob.glob(f'{root}/*'):
        dataset = root.split('/')[-1]
        savedir = subdir.replace(dataset, f'{dataset}_image_slice')
        if not os.path.exists(savedir):
            os.makedirs(savedir)

    for path in tqdm(glob.glob(f'{root}/*/*')):
        dataset = root.split('/')[-1]
        name = path.split(dataset)[-1].replace('.npy', '')[1:]
        save_path = f'/big/sgurram/{dataset}_image_slice/{name}'
        v = np.load(path)
        t, h, w, c = v.shape
        s, e = (t//2 - 16//2), (t//2 + 16//2)
        v = torch.from_numpy(v[s:e]).permute(0, 3, 1, 2)
        v = transforms.CenterCrop(224)(v)
        try:
            f, arr = plt.subplots(1,4)
            arr[0].imshow(v[0].permute(1,2,0))
            arr[1].imshow(v[1].permute(1,2,0))
            arr[2].imshow(v[2].permute(1,2,0))
            arr[3].imshow(v[3].permute(1,2,0))
            plt.savefig(save_path)
        except:
            continue
            # print(f'bad path {name}')

def get_dataset_len(root='/big/sgurram/UCF-101-raw-npy_image_slice'):
    return len(list(glob.glob(f'{root}/*/*')))


if __name__ == "__main__":
#     num = 0
#     text1 = ["Tommy throws a snowball at the car"]
#     text2 = ["throwing snowball at car, January winter time"]
#     print(text_similarity(text1,text2))

    # visualize_dataset_slice()
    # find -type f | wc -l
    print(get_dataset_len(root='/big/sgurram/HMDB-raw-npy_image_slice'))