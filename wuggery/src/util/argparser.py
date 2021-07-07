import argparse
from . import util

parser = argparse.ArgumentParser(description='LanguageModel')
# Data
parser.add_argument('--data-path', type=str, default='data/')

# Model
parser.add_argument('--model', type=str, choices=['lstm', 'cond-lstm'],
                    default='lstm')
parser.add_argument('--checkpoints-path', type=str, default='checkpoints/')

# Results
parser.add_argument('--results-path', type=str, default='results/')

# Others
parser.add_argument('--seed', type=int, default=7,
                    help='Seed for random algorithms repeatability (default: 7)')


def add_argument(*args, **kwargs):
    return parser.add_argument(*args, **kwargs)


def set_defaults(*args, **kwargs):
    return parser.set_defaults(*args, **kwargs)


def get_default(*args, **kwargs):
    return parser.get_default(*args, **kwargs)


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    args.tag = (args.model == 'cond-lstm')

    util.config(args.seed)
    return args
