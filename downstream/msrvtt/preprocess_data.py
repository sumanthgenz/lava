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

flags.DEFINE_string('root_dir', default="/big/davidchan/msr_vtt/train_val_videos", help='Path to msrvtt dataset')
flags.DEFINE_string('save_dir', default="/big/sgurram/msrvtt", help='Path to savedir dataset')
FLAGS = flags.FLAGS
'''
$ export TFHUB_CACHE_DIR=/tmp/; python scripts/preprocess_ucf101_data.py '/big/UCF-101-processed/' '/big/UCF101-LAVA/'
'''
def main(*args):
    global root_dir, save_dir
    root_dir = FLAGS.root_dir
    save_dir = FLAGS.save_dir

    csv = '/home/sgurram/Desktop/msrvtt.csv'
    df = pd.read_csv(csv, usecols=['video_id', 'sentence'])

    vids = list(df['video_id'])
    text = list(df['sentence'])
    test_set = dict(zip(vids, text))

    # find names of files
    data = get_data(root_dir, save_dir, test_set)
   

    #with mp.Pool(12) as pool:
    #     for _ in tqdm(pool.imap_unordered(extract_features_and_save, enumerate(data)), total=len(data)):
    #         continue
    print(root_dir, save_dir)
    for datum in tqdm(enumerate(data), total=len(data)):
       extract_features_and_save(datum)

def get_data(root_dir, save_dir, test_set):
    data = []
    for path, subdirs, files in os.walk(root_dir):
        for subdir in subdirs:
            mkdir_p(os.path.join(save_dir, subdir))
        for f in files:
            name = f.split('.mp4')[0]
            if '.mp4' in f and name in test_set:
                data.append([path, f, test_set[name]])
    logging.info(f"Peeking files {data[:3]}")
    return data


lava_feature_extractor = LAVAFeatures()  
def extract_features_and_save(datum):
    i, (path, f, sentence) = datum
    _, v, t = lava_feature_extractor.get_lava_features(mp4_path=os.path.join(path, f), run_lava=True, text_input=sentence)
    plt.savefig('/home/sgurram/Projects/aai/aai/experimental/sgurram/lava/src/img_sanity')
    name = f.replace('.mp4', '')
    # print(type(v))
    # print(path, root_dir, save_dir, name)
    # print(os.path.join(path.replace(root_dir, save_dir), name))
    np.save(os.path.join(path.replace(root_dir, save_dir), name  + '_v.npy'), v)
    np.save(os.path.join(path.replace(root_dir, save_dir), name  + '_t.npy'), t)

    # try:
    #     np.save(os.path.join(path.replace(root_dir, save_dir), name + '_a.npy'), a)
    # except AttributeError as ae:
    #     logging.info('couldnt find audio path')

    return

if __name__ == "__main__":
    flags.mark_flags_as_required(['root_dir', 'save_dir'])
    app.run(main)
    # csv = '/home/sgurram/Desktop/msrvtt.csv'
    # df = pd.read_csv(csv, usecols=['video_id', 'sentence'])

    # path = '/big/davidchan/msr_vtt/train_val_videos'
    # vids = df['video_id']

    # count = 0
    # for v in vids:
    #     if os.path.isfile(f'{path}/{v}.mp4'):
    #         count += 1
    # print(count)