import sys
import logging
import numpy as np
from tqdm import tqdm

sys.path.append('./src/')
from h01_data.alphabet import Alphabet
from h01_data.unimorph import Unimorph
from util import argparser


def get_args():
    argparser.add_argument(
        "--n-folds", type=int, default=10,
        help="Number of folds to split data")
    argparser.add_argument(
        "--language", type=str, required=True,
        help="Language to analyse")

    return argparser.parse_args()


def get_data(data_path, language):
    return Unimorph.read_data_orig(data_path, language)


def get_fold_splits(concepts, n_folds):
    splits = concepts.unique()
    np.random.shuffle(splits)
    splits = np.array_split(splits, n_folds)
    splits = {x: i for i, fold in enumerate(splits) for x in fold}
    return splits


def process_line(row, alphabets):
    alphabets['chars'].add_word(row.word)
    alphabets['tag'].add_char(row.tag)

    return {
        'count': 1,
        'id': row.id,
        'tag': row.tag,
        'concept': row.concept,
        'word': row.word
    }


def process_data(df, n_folds, splits, alphabets):
    folds = [[] for _ in range(n_folds)]

    for _, row in tqdm(df.iterrows(), desc='Processing wiki data',
                       total=df.shape[0]):
        fold = splits[row.concept]
        item = process_line(row, alphabets)
        folds[fold] += [item]

    return folds


def process(data_path, language, n_folds):
    df = get_data(data_path, language)
    splits = get_fold_splits(df.concept, n_folds)
    alphabets = {
        'chars': Alphabet(),
        'tag': Alphabet()
    }

    folds = process_data(df, n_folds, splits, alphabets)
    Unimorph.write_data(data_path, language, folds, alphabets)

    print('# unique chars:', len(alphabets['chars']))
    print('# unique tag:', len(alphabets['tag']))


def main():
    args = get_args()
    logging.info(args)

    process(args.data_path, args.language, args.n_folds)


if __name__ == '__main__':
    main()
