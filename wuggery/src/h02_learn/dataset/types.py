import torch
from torch.utils.data import Dataset


class TypeDataset(Dataset):
    # pylint: disable=too-many-instance-attributes

    def __init__(self, data, folds):
        self.data = data
        self.folds = folds
        self.process(data)
        self._train = True

    def process(self, data):
        folds_data, alphabets = data
        self.alphabet = alphabets['chars']

        self.words = [
            item['word']
            for fold in self.folds for item in folds_data[fold]
        ]
        self.words_tensor = [torch.LongTensor(self.get_word_idx(word)) for word in self.words]
        self.n_instances = len(self.words_tensor)

    def get_word_idx(self, word):
        return [self.alphabet.char2idx('SOS')] + \
            self.alphabet.word2idx(word) + \
            [self.alphabet.char2idx('EOS')]

    def __len__(self):
        return self.n_instances

    def __getitem__(self, index):
        return (self.words_tensor[index],)
