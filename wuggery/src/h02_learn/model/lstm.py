import torch
import torch.nn as nn
import torch.nn.functional as F

from util import constants
from .base import BaseLM


class LstmLM(BaseLM):
    # pylint: disable=arguments-differ
    name = 'lstm'

    def __init__(self, alphabet, embedding_size, hidden_size,
                 nlayers, dropout):
        super().__init__(alphabet, embedding_size, hidden_size,
                         nlayers, dropout)

        self.embedding = nn.Embedding(self.alphabet_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, nlayers,
                            dropout=(dropout if nlayers > 1 else 0),
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, embedding_size)
        self.out = nn.Linear(embedding_size, self.alphabet_size)

        # Tie weights
        self.out.weight = self.embedding.weight

    def forward(self, x):
        x_emb = self.dropout(self.get_embeddings(x))

        c_t, _ = self.lstm(x_emb)
        c_t = self.dropout(c_t).contiguous()

        hidden = F.relu(self.linear(c_t))
        logits = self.out(hidden)
        return logits

    def get_embeddings(self, x):
        return self.embedding(x)

    def sample(self, args):
        eos = self.alphabet.EOS_IDX
        pad = self.alphabet.PAD_IDX

        x = self.get_initial_input(args)

        samples = []
        first = True

        while True:
            logits = self(x)
            logits = self.mask_logits(logits, first, eos, pad)
            probs = F.softmax(logits, dim=-1)

            first = False

            y = probs.multinomial(1)
            x = self.extend_input(x, y)

            x, ended, finished_samples = self.get_finished(x, y, eos, pad)
            samples += finished_samples

            if ended.all():
                break

        return [''.join(x) for x in samples]

    def get_initial_input(self, args):
        x = torch.LongTensor(
            [[self.alphabet.SOS_IDX] for _ in range(args.batch_size)]) \
            .to(device=constants.device)
        return x

    @staticmethod
    def extend_input(x, y):
        return torch.cat([x, y], dim=-1)

    def get_finished(self, x, y, eos, pad):
        samples = []
        ended = (y == eos) | (y == pad)
        if ended.any():
            mask = ended.repeat(1, x.shape[1])
            words = x[mask].reshape(-1, x.shape[-1])
            x = x[~mask].reshape(-1, x.shape[-1])
            samples = [self.alphabet.idx2word(item[1:-1].cpu().numpy()) for item in words]
        return x, ended, samples

    @staticmethod
    def mask_logits(logits, first, eos, pad):
        logits = logits[:, -1, :]
        if first:
            logits[:, eos] = -float('inf')
            logits[:, pad] = -float('inf')

        top_logits, _ = logits.topk(k=10, dim=-1)
        topk_mask = (logits < top_logits[:, -1:].repeat(1, logits.shape[-1]))
        logits[topk_mask] = -float('inf')
        return logits
