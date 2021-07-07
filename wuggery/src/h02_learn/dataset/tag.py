import torch

from .types import TypeDataset


class TagDataset(TypeDataset):

    def process(self, data):
        super().process(data)
        folds_data, alphabets = data
        self.alphabet_tag = alphabets['tag']

        self.tag = [
            item['tag']
            for fold in self.folds for item in folds_data[fold]
        ]
        self.tag_tensor = [torch.LongTensor(self.get_tag_idx(tag)) for tag in self.tag]
        assert len(self.tag_tensor) == self.n_instances, 'Should have as many unimorph tags as words'

    def get_tag_idx(self, tag):
        return [self.alphabet_tag.char2idx(tag)]

    def __getitem__(self, index):
        return (self.words_tensor[index], self.tag_tensor[index])
