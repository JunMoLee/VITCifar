import torch
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from itertools import product
from pathlib import Path


def make_reproducible(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_rel_pos(width=32, dist=False):
    tmp = np.array(list(product(list(product(range(width), repeat=2)), repeat=2)))
    rel_pos_tmp = (tmp[:, 0, :] - tmp[:, 1, :]).reshape(width**2, width**2, 2)
    if dist:
        rel_pos = (rel_pos_tmp[:, :, 0]**2 + rel_pos_tmp[:, :, 1]**2).flatten()
    else:
        rel_pos_frame = pd.DataFrame(rel_pos_tmp[:, :, 0]).astype(str) + pd.DataFrame(rel_pos_tmp[:, :, 1]).astype(str)
        rel_pos = rel_pos_frame.values.flatten()
    le = LabelEncoder()
    rel_pos = le.fit_transform(rel_pos)
    rel_pos = rel_pos.reshape(width**2, width**2)
    return torch.tensor(rel_pos)


def mkdir(path_to_directory):
    p = Path(path_to_directory)
    p.mkdir(exist_ok=True, parents=True)