from absl import app, flags, logging
import numpy as np
import multiprocessing as mp
import os
from tqdm import tqdm
from aai.experimental.sgurram.lava.src.features import LAVAFeatures
from aai.experimental.sgurram.lava.src.metrics import cosine_similarity
from aai.alexandria.util import mkdir_p
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

flags.DEFINE_string('save_dir', default="/big/sgurram/msrvtt", help='Path to savedir dataset')
FLAGS = flags.FLAGS
'''
$ export TFHUB_CACHE_DIR=/tmp/; python scripts/preprocess_ucf101_data.py '/big/UCF-101-processed/' '/big/UCF101-LAVA/'
'''
def main(*args):
    global save_dir
    save_dir = FLAGS.save_dir

    csv = '/home/sgurram/Desktop/msrvtt.csv'
    df = pd.read_csv(csv, usecols=['video_id', 'sentence'])

    paths = list(df['video_id'])
    vids, text = load_data(paths)

    sim = vids @ text.T

    print(get_recall_score(sim, k=10))

def load_data(paths):
    vids, text = [], []
    for p in tqdm(paths):
        v = np.load(f'{save_dir}/{p}_v.npy')
        t = np.load(f'{save_dir}/{p}_t.npy')
        vids.append(v)
        text.append(t)
    
    vids = torch.tensor(vids)
    text = torch.tensor(text)
    return vids, text


def get_recall_score(sim, k=10):
    count, total = 0, sim.shape[0]
    for i, row in tqdm(enumerate(sim)):
        topk_indices = torch.topk(row, k=k)[1]
        if i in topk_indices:
            count += 1
    return count / total

if __name__ == "__main__":
    flags.mark_flags_as_required(['save_dir'])
    app.run(main)