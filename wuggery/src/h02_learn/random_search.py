import sys
import random
import itertools
import subprocess

sys.path.append('./src/')
from util import argparser


def get_args():
    # Data
    argparser.add_argument('--language', type=str, required=True)
    # Optimization
    argparser.add_argument('--n-runs', type=int, required=True)

    return argparser.parse_args()


def get_hyperparameters():
    nlayers = random.randint(1, 3)
    embedding_size = int(2**(3 + random.random() * 7))
    hidden_size = int(2**(3 + random.random() * 7))
    dropout = random.random()

    hyperparameters = {
        '--nlayers': nlayers,
        '--embedding-size': embedding_size,
        '--hidden-size': hidden_size,
        '--dropout': round(dropout, 4),
    }
    return dict2list(hyperparameters)


def dict2list(data):
    list2d = [[k, str(x)] for k, x in data.items()]
    return list(itertools.chain.from_iterable(list2d))


def main():
    args = get_args()
    for i in range(args.n_runs):
        print()
        hyperparameters = get_hyperparameters()
        cmd = ['python', 'src/h02_learn/train.py', '--language', args.language]
        print('(%02d/%02d) Training on language: %s' % (i + 1, args.n_runs, args.language))
        print(cmd + hyperparameters)
        subprocess.check_call(cmd + hyperparameters)
        print()


if __name__ == '__main__':
    main()
