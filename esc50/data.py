
import os

import numpy as np
import torch

from aai.utils import load_json


class ESC50Dataset(torch.utils.data.Dataset):
    def __init__(self, root, split):

        self._root = root
        self._metadata = load_json(os.path.join(root, 'lava', 'meta', f'{split}.json'))

    def __len__(self,):
        return len(self._metadata)

    def __getitem__(self, idx):
        elem = self._metadata[idx]

        # Load the features
        features = np.load(elem['feature_path'], allow_pickle=True)

        # Return the class elems
        return {
            'features': features,
            'class': elem['class']
        }
