import torch
import torch.nn as nn

from util import constants
from .lstm import LstmLM


class CondLstmLM(LstmLM):
    name = 'conditioned-lstm'

    def __init__(self, alphabet, tag, embedding_size, hidden_size,
                 nlayers, dropout):
        super().__init__(alphabet, embedding_size, hidden_size,
                         nlayers, dropout)
        self.tag = tag
        self.tag_size = len(tag)

        self.tag_embedding = nn.Embedding(self.tag_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size * 2, hidden_size, nlayers,
                            dropout=(dropout if nlayers > 1 else 0),
                            batch_first=True)

    def get_embeddings(self, x):
        x_emb = self.embedding(x[0])
        tag_emb = self.tag_embedding(x[1]).unsqueeze(1).repeat(1, x_emb.shape[1], 1)
        return torch.cat([x_emb, tag_emb], dim=-1)

    def get_args(self):
        args = super().get_args()
        args['tag'] = self.tag
        return args

    def get_initial_input(self, args):
        x = super().get_initial_input(args)
        tag = torch.LongTensor(
            [self.tag.char2idx(args.sampled_tag) for _ in range(args.batch_size)]) \
            .to(device=constants.device)
        return x, tag

    def extend_input(self, x, y):
        text, tag = x
        text = torch.cat([text, y], dim=-1)
        return text, tag

    def get_finished(self, x, y, eos, pad):
        text, tag = x

        text, ended, samples = super().get_finished(text, y, eos, pad)
        tag = tag[~ended[:, 0]]

        return (text, tag), ended, samples
