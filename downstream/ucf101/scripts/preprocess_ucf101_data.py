from absl import app, flags, logging
import numpy as np
import multiprocessing as mp
import os
from tqdm import tqdm
from aai.experimental.sgurram.lava.src.features import LAVAFeatures
from aai.experimental.sgurram.lava.src.metrics import cosine_similarity
from aai.alexandria.util import mkdir_p
import torch

flags.DEFINE_string('root_dir', default="/big/iherzi/UCF-101-Processed/", help='Path to ucf101 dataset')
flags.DEFINE_string('save_dir', default="/big/sgurram/UCF-101-raw-npy/", help='Path to savedir dataset')
FLAGS = flags.FLAGS
'''
$ export TFHUB_CACHE_DIR=/tmp/; python scripts/preprocess_ucf101_data.py '/big/UCF-101-processed/' '/big/UCF101-LAVA/'
'''
def main(*args):
    global root_dir, save_dir
    root_dir = FLAGS.root_dir
    save_dir = FLAGS.save_dir

    # find names of files
    data = get_data(root_dir, save_dir)
   

    #with mp.Pool(12) as pool:
    #     for _ in tqdm(pool.imap_unordered(extract_features_and_save, enumerate(data)), total=len(data)):
    #         continue
    print(root_dir, save_dir)
    for datum in tqdm(enumerate(data), total=len(data)):
       extract_features_and_save(datum)

def get_data(root_dir, save_dir):
    data = []
    for path, subdirs, files in os.walk(root_dir):
        for subdir in subdirs:
            mkdir_p(os.path.join(save_dir, subdir))
        for f in files:
            if '.mp4' in f:
                data.append([path, f])
    logging.info(f"Peeking files {data[:3]}")
    return data


lava_feature_extractor = LAVAFeatures()  
def extract_features_and_save(datum):
    i, (path, f) = datum
    a, v, _ = lava_feature_extractor.get_lava_features(mp4_path=os.path.join(path, f), run_lava=False)
    name = f.replace('.mp4', '')
    # print(type(v))
    # print(path, root_dir, save_dir, name)
    # print(os.path.join(path.replace(root_dir, save_dir), name))
    np.save(os.path.join(path.replace(root_dir, save_dir), name  + '_v.npy'), v)
    try:
        np.save(os.path.join(path.replace(root_dir, save_dir), name + '_a.npy'), a)
    except AttributeError as ae:
        logging.info('couldnt find audio path')

    return

if __name__ == "__main__":
    flags.mark_flags_as_required(['root_dir', 'save_dir'])
    app.run(main)
