
import os

from absl import app, logging, flags
import numpy as np
import torch
import tqdm

from aai.utils import iload_xsv, mkdir_p, write_json
from aai.experimental.sgurram.lava.src.features import LAVAFeatures

flags.DEFINE_string('root', default=None, help='The root path of the ESC-50 pull.')
flags.DEFINE_string('eval_fold', default='5', help='The fold of the data to use for evaluation.')
flags.DEFINE_string('lava_checkpoint', default=None, help='The path to the LAVA checkpoint to use.')
FLAGS = flags.FLAGS


def main(*unused_argv):
    metadata = iload_xsv(os.path.join(FLAGS.root, 'meta', 'esc50.csv'))

    # Load the LAVA feature extractor
    lfe = LAVAFeatures(lava_model_path=FLAGS.lava_checkpoint)

    # Create a directory for the output features
    mkdir_p(os.path.join(FLAGS.root, 'lava', 'features'))
    mkdir_p(os.path.join(FLAGS.root, 'lava', 'meta'))

    train_data = []
    val_data = []
    for idx, entry in tqdm.tqdm(enumerate(metadata)):
        input_wav_path = os.path.join(FLAGS.root, 'audio', entry.filename)
        output_feature_path = os.path.join(FLAGS.root, 'lava', 'features', entry.filename.replace('.wav', '.npy'))
        with torch.no_grad():
            a, v, t = lfe.get_lava_features(wav_path=input_wav_path)
        np.save(output_feature_path, a)
        datum = {
            'wav_path': input_wav_path,
            'feature_path': output_feature_path,
            'class': int(entry.target),
            'category': entry.category,
        }
        if entry.fold == FLAGS.eval_fold:
            val_data.append(datum)
        else:
            train_data.append(datum)

    write_json(train_data, os.path.join(FLAGS.root, 'lava', 'meta', 'train.json'))
    write_json(val_data, os.path.join(FLAGS.root, 'lava', 'meta', 'eval.json'))


if __name__ == "__main__":
    flags.mark_flags_as_required(['root', 'lava_checkpoint'])
    app.run(main)
