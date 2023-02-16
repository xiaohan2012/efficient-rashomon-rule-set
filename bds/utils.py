import numpy as np
import os
import pickle as pkl
import json
import math
import tempfile

from .common import Program


def randints(num, vmin=0, vmax=100000):
    return np.random.randint(vmin, vmax, num)




def int_floor(num: float):
    return int(math.floor(num))


def int_ceil(num: float):
    return int(math.ceil(num))


def flatten(stuff):
    """flatten an array"""
    return np.asarray(stuff).flatten()


def get_tempdir(prefix=None, suffix=None, dir=None):
    if dir is not None:
        makedir(dir, usedir=False)
    return tempfile.mkdtemp(prefix=prefix, suffix=suffix, dir=dir)


def makedir(d, usedir=True):
    if usedir:
        d = os.path.dirname(d)

    if not os.path.exists(d):
        os.makedirs(d)


def save_pickle(obj, path):
    return pkl.dump(obj, open(path, "wb"))


def save_file(string, path):
    with open(path, "w") as f:
        f.write(string)


def load_pickle(path):
    return pkl.load(open(path, "rb"))


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)
