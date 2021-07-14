import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import math
import csv
from tqdm import trange

class Model(nn.Module):
    def __init__(self, alphabet, embedding_size, hidden_size, num_layers=1, genders=2):
        super(Model, self).__init__()

        self.alphabet_size = len(alphabet) + 3 # include <START> and <END>
        self.hidden_size = hidden_size
        self.num_layer = num_layers
        self.genders = genders
        self.embedding_size = embedding_size
        self.pad_len = 15

        self.embedding = nn.Embedding(self.alphabet_size, self.embedding_size)
        self.encoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size,
            bidirectional=False, batch_first=True)
        self.classifier = nn.Linear(self.hidden_size, self.genders)

        self.char2idx = {'<START>': 0, '<END>': 1, '<PAD>': 2}
        idx = 3
        for char in alphabet:
            self.char2idx[char] = idx
            idx += 1

    # https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-are-going-to-use-word-embedding-and-bilstm/28184/2
    def simple_elementwise_apply(fn, packed_sequence):
        """applies a pointwise function fn to each element in packed_sequence"""
        return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)

    def forward(self, batch):
        batch = [['<START>'] + list(word.split()) + ['<END>'] for word in batch]
        # lens = [len(word) for word in batch]
        batch = torch.LongTensor([[self.char2idx[char] for char in word] for word in batch])
        batch = batch.long() # (1, n) => (1, n)
        # print(batch.size())
        # batch = nn.utils.rnn.pack_padded_sequence(batch, batch_first=True, lengths=lens, enforce_sorted=False)
        embed = self.embedding(batch) # (1, n) => (1, n, e)
        output, (ht, ct) = self.encoder(embed) # (1, n, e) => (1, n, e), (1, 1, e), ?
        output = self.classifier(ht[-1]) # (1, 1, e) => (1, g)
        # output = F.softmax(output, dim=1)
        return output

data = []
alphabet = set()
with open('morph/lemmas_ipa.csv', 'r') as fin:
    reader = csv.reader(fin, delimiter='\t')
    for row in reader:
        for char in row[0].split(): alphabet.add(char)
        data.append((row[0], 0 if 'MASC' in row[2] else 1))

print(data[:5])

lens = [{}, {}]
for i, (entry, label) in enumerate(data):
    l = len(entry.split())
    j = 0 if i < 3000 else 1
    if l not in lens[j]:
        lens[j][l] = []
    lens[j][l].append((entry, label))

model = Model(alphabet=alphabet, embedding_size=32, hidden_size=32)
criterion = nn.CrossEntropyLoss()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.0001)

tq = trange(200, desc="Training", unit="epochs")
minibatch_size = 20
for i in tq:
    model.train()
    total_loss = torch.tensor([0.0], requires_grad=True)
    optimizer.zero_grad()
    for j in lens[0]:
        for k in range(0, len(lens[0][j]), minibatch_size):
            end = min(len(lens[0][j]), k + minibatch_size)
            res = model([x[0] for x in lens[0][j][k:end]])
            goal = torch.Tensor([x[1] for x in lens[0][j][k:end]]).long()
            loss = criterion(res, goal)
            # print(len(lens[0][j]), loss)
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss

    model.eval()
    with torch.no_grad():
        all_goal = []
        all_preds = []
        val_total_loss = torch.tensor([0.0], requires_grad=False)
        for j in lens[1]:
            val_res = model([x[0] for x in lens[1][j]])
            val_goal = torch.Tensor([x[1] for x in lens[1][j]]).long()
            all_goal.extend(val_goal.tolist())
            val_loss = criterion(val_res, val_goal)
            val_preds = torch.argmax(val_res, 1)
            all_preds.extend(val_preds.tolist())
            val_total_loss = val_total_loss + val_loss
    
    tq.set_postfix(loss=loss.item(), val_loss=val_loss.item(), val_acc=accuracy_score(all_goal, all_preds))


