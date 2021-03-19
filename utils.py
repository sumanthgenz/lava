import cv2
import av
import torch
import torchaudio
import torchvision
import torchtext
import numpy as np
from typing import Tuple, Optional
import warnings
import pickle
import glob
from torchtext.data.functional import generate_sp_model, load_sp_model, sentencepiece_numericalizer
from aai.experimental.sgurram.lava.src.references import mp4_root_dir, pickle_root_dir

torchaudio.set_audio_backend("sox_io") 
warnings.filterwarnings("ignore")


def nan_filter(input):
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
    if video_frames.shape[1] != target_size or video_frames.shape[2] != target_size:
        video_frames = video_frames[:, :target_size, :target_size, :]
        w_pad, h_pad = target_size - int(video_frames.shape[1]), target_size - int(video_frames.shape[2])
        video_frames =  torch.nn.functional.pad(video_frames, pad=((0,0, 0,w_pad, 0,h_pad, 0,0)))
    if video_frames.shape[0] < seq_len:
        padding = torch.zeros(seq_len - video_frames.shape[0], target_size, target_size, 3)
        video_frames =  torch.cat((video_frames, padding))
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

def get_log_mel_spec(wave, samp_freq=16000):
    wave = torch.unsqueeze(wave, 0)
    spec = torchaudio.transforms.MelSpectrogram()(wave)
    return spec.log2()[0,:,:]

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
        power = torch.arange(0, hidden_size, 2).float().cuda() / hidden_size
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

def generate_tokenizer(filename, vocab_size):
    generate_sp_model(filename, vocab_size, model_type='bpe', model_prefix='lava_sp')

def tokenize_text(sp_model_path, text):
    sp_model = load_sp_model(sp_model_path)
    sp_id_generator = sentencepiece_numericalizer(sp_model)
    return list(sp_id_generator(text))


if __name__ == '__main__':
    # generate_tokenizer('sample.txt', int(10000))
    text = ["Vladmir Putin is the current leader of Russia", "Google Universal Sentence Enocder was merely a phase!"]
    print(tokenize_text(sp_model_path='lava_sp.model', text=text))
