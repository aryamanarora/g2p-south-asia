import torch
from torch.utils.data import DataLoader

from h01_data.unimorph import Unimorph
from util import constants
from util import util
from .types import TypeDataset
from .tag import TagDataset


def generate_batch(batch):
    r"""
    Since the text entries have different lengths, a custom function
    generate_batch() is used to generate data batches and offsets,
    which are compatible with EmbeddingBag. The function is passed
    to 'collate_fn' in torch.utils.data.DataLoader. The input to
    'collate_fn' is a list of tensors with the size of batch_size,
    and the 'collate_fn' function packs them into a mini-batch.[len(entry[0][0]) for entry in batch]
    Pay attention here and make sure that 'collate_fn' is declared
    as a top level def. This ensures that the function is available
    in each worker.
    """

    tensor = batch[0][0]
    batch_size = len(batch)
    max_length = max([len(entry[0]) for entry in batch]) - 1  # Does not need to predict SOS

    x = tensor.new_zeros(batch_size, max_length)
    y = tensor.new_zeros(batch_size, max_length)

    for i, item in enumerate(batch):
        sentence = item[0]
        sent_len = len(sentence) - 1  # Does not need to predict SOS
        x[i, :sent_len] = sentence[:-1]
        y[i, :sent_len] = sentence[1:]

    if len(batch[0]) == 1:
        return x.to(device=constants.device), y.to(device=constants.device)

    tag = torch.cat([x[1] for x in batch], dim=-1) \
        .to(device=constants.device)
    x, y = x.to(device=constants.device), y.to(device=constants.device)
    return (x, tag), y


def get_data_cls(tag):
    if tag:
        return TagDataset
    return TypeDataset

def load_data(fname, language):
    return Unimorph.read_data(fname, language)


def get_alphabet(data):
    _, alphabets = data
    assert alphabets['chars'].char2idx('PAD') == 0, 'Padding idx should be 0'
    assert alphabets['tag'].char2idx('PAD') == 0, 'Padding idx should be 0'
    return alphabets


def get_data_loader(dataset_cls, data, folds, batch_size, shuffle):
    trainset = dataset_cls(data, folds)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle,
                             collate_fn=generate_batch)
    return trainloader


def get_data_loaders(language, data_path, folds, batch_size, tag=False):
    dataset_cls = get_data_cls(tag)
    data = load_data(data_path, language)
    alphabets = get_alphabet(data)
    trainloader = get_data_loader(
        dataset_cls, data, folds[0], batch_size=batch_size, shuffle=True)
    devloader = get_data_loader(
        dataset_cls, data, folds[1], batch_size=batch_size, shuffle=False)
    testloader = get_data_loader(
        dataset_cls, data, folds[2], batch_size=batch_size, shuffle=False)
    return trainloader, devloader, testloader, alphabets
