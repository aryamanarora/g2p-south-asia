import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, balanced_accuracy_score
import math
import csv
from tqdm import trange
from collections import Counter
from scipy.stats import entropy
from itertools import groupby
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

random.seed(42)

GENDER = {
    'MASC': 0,
    'FEM': 1,
    'NEUT': 2
}

class Model(nn.Module):
    def __init__(self, alphabet, embedding_size, hidden_size, genders, num_layers=1):
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
        # add start and end tokens and convert chars to ids
        # b = batch size, n = input length, e = embedding size
        batch = [['<START>'] + list(word.split()) + ['<END>'] for word in batch]
        batch = torch.LongTensor([[self.char2idx[char] for char in word] for word in batch])
        batch = batch.long() # (b, n) => (b, n)

        # embeddings
        embed = self.embedding(batch) # (b, n) => (b, n, e)

        # lstm
        output, (ht, ct) = self.encoder(embed) # (b, n, e) => (b, n, e), (1, b, e), (1, b, e)

        # gender probabilities
        output = self.classifier(ht[-1]) # (1, b, e) => (b, g)
        
        return output, ht[-1]

def load_data(filename, ipa, ignore_mwe=False):
    data = []
    alphabet = set()
    with open(filename, 'r') as fin:
        reader = csv.reader(fin, delimiter='\t')
        for row in reader:
            if not ipa:
                row[0] = ' '.join(list(row[0]))
            if ignore_mwe:
                if '   ' in row[0] or ' - ' in row[0]:
                    continue
            row[0] = '<S> ' + row[0] + ' <E>'
            for char in row[0].split(): alphabet.add(char)
            gender = 0
            attrs = row[2].split(';')
            for option in GENDER:
                if option in attrs:
                    gender = GENDER[option]
            data.append((row[0], gender))
    print(data[:5])
    random.shuffle(data)
    return alphabet, data

def conditional_entropy(data, func):
    res = {}
    for i in data:
        key = func(i)
        if key not in res: res[key] = []
        res[key].append(i)
    
    tot = len(data)
    cts = [Counter([z[1] for z in x]).values() for x in res.values()]
    entropies = [(sum([y for y in x]) / tot) * entropy([y for y in x], base=2) for x in cts]
    return sum(entropies)

def conditional_entropy_arr(arr1, arr2):
    res = {}
    for i in range(len(arr1)):
        key = arr2[i]
        if key not in res: res[key] = []
        res[key].append(arr1[i])
    
    tot = len(arr1)
    cts = [Counter([z for z in x]).values() for x in res.values()]
    entropies = [(sum([y for y in x]) / tot) * entropy([y for y in x], base=2) for x in cts]
    return sum(entropies)

def print_stats(data):
    cts = Counter([row[1] for row in data])
    print(f'Total: {len(data)}')
    print(f'Gender counts: {cts}')
    print(f'H(G): {entropy([x for x in cts.values()], base=2)} bits')
    print(f'H(G|S): {conditional_entropy(data, lambda x: tuple(x[0].split()))}')
    print(f'H(G|S[n]): {conditional_entropy(data, lambda x: tuple(x[0].split()[-1]))}')
    print(f'H(G|S[n-1:n]): {conditional_entropy(data, lambda x: tuple(x[0].split()[-2:]))}')
    print(f'H(G|S[n-2:n]): {conditional_entropy(data, lambda x: tuple(x[0].split()[-3:]))}')
    print(f'H(G|S[1]): {conditional_entropy(data, lambda x: tuple(x[0].split()[0]))}')
    print(f'H(G|S[1:2]): {conditional_entropy(data, lambda x: tuple(x[0].split()[0:2]))}')
    print(f'H(G|S[1:3]): {conditional_entropy(data, lambda x: tuple(x[0].split()[0:3]))}')
    # input()
    # print(f'H(G|S_[n-1,n]): {adjusted_mutual_info_score([x[1] for x in data], [x[0][-2:] for x in data])}')

def split_by_lengths(data):
    lens = {}
    for entry in data:
        l = len(entry[0].split())
        if l not in lens:
            lens[l] = []
        lens[l].append(entry)
    return lens

def train(filename, genders, ipa, testfile=None):
    alphabet, data = load_data(filename=filename, ipa=ipa)
    more_test_data = []
    if testfile:
        more_test_data = load_data(filename=testfile, ipa=ipa)
    print_stats(data)

    pca = PCA(n_components=2)

    test_data = data[:int(0.1 * len(data))]
    test_data.extend(more_test_data)
    split = len(data) - 2 * len(test_data)
    data = data[len(test_data):]
    print(len(data), len(test_data))

    lens = [{}, {}, {}]
    lens[0] = split_by_lengths(data[:split])
    lens[1] = split_by_lengths(data[split:])
    lens[2] = split_by_lengths(test_data)

    model = Model(alphabet=alphabet, embedding_size=5, hidden_size=5, genders=genders)
    criterion = nn.CrossEntropyLoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.001)

    tq = trange(20, desc="Training", unit="epochs")
    minibatch_size = 20
    last_acc = 0
    for i in tq:
        model.train()
        total_loss = torch.tensor([0.0], requires_grad=True)
        optimizer.zero_grad()
        for j in lens[0]:
            for k in range(0, len(lens[0][j]), minibatch_size):
                end = min(len(lens[0][j]), k + minibatch_size)
                res, _ = model([x[0] for x in lens[0][j][k:end]])
                goal = torch.Tensor([x[1] for x in lens[0][j][k:end]]).long()
                loss = criterion(res, goal)
                # print(len(lens[0][j]), loss)
                loss.backward()
                optimizer.step()
                total_loss = total_loss + loss / (end - k)

        model.eval()
        with torch.no_grad():
            all_goal = []
            all_preds = []
            val_total_loss = torch.tensor([0.0], requires_grad=False)
            for j in lens[1]:
                val_res, val_hid = model([x[0] for x in lens[1][j]])
                val_goal = torch.Tensor([x[1] for x in lens[1][j]]).long()
                all_goal.extend(val_goal.tolist())
                val_loss = criterion(val_res, val_goal)
                val_preds = torch.argmax(val_res, 1)
                all_preds.extend(val_preds.tolist())
                val_total_loss = val_total_loss + val_loss
        
        acc = accuracy_score(all_goal, all_preds)
        ami = adjusted_mutual_info_score(all_goal, all_preds)
        tq.set_postfix(loss=total_loss.item() / len(data), val_loss=val_total_loss.item() / len(test_data), val_acc=acc, val_ami=ami)
    
    with torch.no_grad():
        all_labels = []
        all_goal = []
        all_preds = []
        all_hid = []
        val_total_loss = torch.tensor([0.0], requires_grad=False)

        # model estimate of entropy conditioned on form, per https://arxiv.org/pdf/2005.00626.pdf
        model_entropy = 0

        for j in lens[2]:
            all_labels.extend([''.join(x[0][4:-4].split()) for x in lens[2][j]])
            val_res, val_hid = model([x[0] for x in lens[2][j]])
            val_goal = torch.Tensor([x[1] for x in lens[2][j]]).long()
            all_goal.extend(val_goal.tolist())
            val_loss = criterion(val_res, val_goal)
            val_preds = torch.argmax(val_res, 1)
            for i in range(len(lens[2][j])):
                pred = F.softmax(val_res[i])
                model_entropy -= torch.log(pred)[val_goal[i]]
                print(f'{lens[2][j][i]}, {F.softmax(val_res[i])}, {val_preds[i]}')
            all_preds.extend(val_preds.tolist())
            val_total_loss = val_total_loss + val_loss
            for i in val_hid:
                all_hid.append(i.tolist())
        
        # print(all_hid[:10])
        pca.fit(all_hid)
        hid_pca = pca.transform(all_hid)
        plt.scatter(hid_pca[:, 0], hid_pca[:, 1], c=all_goal)
        # for i in range(len(all_labels)):
        #     plt.annotate(all_labels[i], hid_pca[i])
        # plt.show()
        plt.savefig(filename.replace('.csv', '') + '-5chart.png')
        plt.clf()

        print(conditional_entropy_arr(all_goal, all_preds))
        print(f'Model entropy: {model_entropy / len(all_goal)}')
        acc = accuracy_score(all_goal, all_preds)
        bacc = balanced_accuracy_score(all_goal, all_preds)
        ami = adjusted_mutual_info_score(all_goal, all_preds)
        print(f'loss={val_total_loss / len(test_data)}, test_acc={acc}, test_bacc={bacc}, test_ami={ami}')

if __name__ == '__main__':
    train(filename='hindi/morph/lemmas.csv', genders=2, ipa=False)
    train(filename='german/morph/lemmas.csv', genders=3, ipa=False)
    train(filename='arabic/morph/lemmas.csv', genders=2, ipa=False)