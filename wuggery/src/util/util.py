import os
import pathlib
import io
import csv
import pickle
import random
import numpy as np
import torch


def config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_csv(filename, results):
    with io.open(filename, 'w', encoding='utf8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(results)


def write_data(filename, embeddings):
    with open(filename, "wb") as f:
        pickle.dump(embeddings, f)


def read_data(filename):
    with open(filename, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def read_data_if_exists(filename):
    try:
        return read_data(filename)
    except FileNotFoundError:
        return {}


def remove_if_exists(fname):
    try:
        os.remove(fname)
    except FileNotFoundError:
        pass


def get_filenames(filepath):
    filenames = [os.path.join(filepath, f)
                 for f in os.listdir(filepath)
                 if os.path.isfile(os.path.join(filepath, f))]
    return sorted(filenames)


def get_dirs(filepath, full=True):
    if full:
        return get_dirs_full(filepath)
    return get_dirs_short(filepath)


def get_dirs_full(filepath):
    filenames = [os.path.join(filepath, f)
                 for f in get_dirs_short(filepath)]
    return filenames

def get_dirs_short(filepath):
    filenames = [f
                 for f in os.listdir(filepath)
                 if os.path.isdir(os.path.join(filepath, f))]
    return sorted(filenames)


def mkdir(folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
