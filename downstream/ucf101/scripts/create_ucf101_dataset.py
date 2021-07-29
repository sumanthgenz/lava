from absl import app, flags, logging
import numpy as np
import multiprocessing as mp
import os
from tqdm import tqdm
#from aai.experimental.sgurram.lava.src.features import LAVAFeatures
from aai.alexandria.util import mkdir_p
from aai.utils.video.file import load_video_to_numpy
from aai.experimental.ilianherzi.augmented_video_learning.video_transforms import Resize, CenterCrop

import multiprocessing as mp


flags.DEFINE_string('root_dir', default="/big/iherzi/UCF-101/", help='Path to ucf101 dataset')
flags.DEFINE_string('save_dir', default="/big/iherzi/UCF-101-Processed/", help='Path to savedir dataset')
FLAGS = flags.FLAGS

def get_data_paths(root_dir, save_dir):
    data = []
    for path, subdirs, files in os.walk(root_dir):
        for subdir in subdirs:
            mkdir_p(os.path.join(save_dir, subdir))
        for f in files:
            if '.avi' in f:
                data.append([os.path.join(path, f), os.path.join(save_dir, subdir, f.replace('.avi', '.npy'))])
    logging.info(f"Peeking files {data[:3]}")
    return data

r = Resize(256)
cc = CenterCrop()
def process_video(video):
    video = cc(r(video))
    return video

def root_func(data):
    i, (source_path, save_path) = data
    video_np = load_video_to_numpy(source_path)
    video_np = process_video(video_np)
    np.save(save_path, video_np)
    #print(save_path)
    return 
    
def main(unused_args):
    root_dir = FLAGS.root_dir
    save_dir = FLAGS.save_dir
    data = get_data_paths(root_dir, save_dir)
    manager = mp.Manager()
    m_list = manager.list(data)
    with mp.Pool(16) as pool:
        for _ in pool.imap_unordered(root_func, tqdm(enumerate(m_list), total=len(m_list))):
            continue           
    
if __name__ == '__main__':
    app.run(main)

# def hotfix():
#     for path, subdirs, files in os.walk(FLAGS.save_dir):
#             for subdir in subdirs:
#                 mkdir_p(os.path.join(FLAGS.save_dir, subdir))
#             for f in files:
#                 if '.avi.npy' in f:
#                     os.remove(os.path.join(FLAGS.save_dir, subdir, f))

#         logging.info(f"Peeking files {data[:3]}")