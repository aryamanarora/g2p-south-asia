import sys
import logging
import numpy as np
from tqdm import tqdm

sys.path.append('./src/')
from h01_data.alphabet import Alphabet
from h01_data.unimorph import Unimorph
from util import argparser
from util import util


def get_args():
    argparser.add_argument(
        "--src-file", type=str, required=True,
        help="Wikipedia tokenized file")
    argparser.add_argument(
        "--dict-file", type=str, required=True,
        help="Wikipedia word types dict file")

    return argparser.parse_args()


def process_data(src_file):
    types = set()

    with open(src_file, "r", encoding="utf-8") as fp:
        for i, line in tqdm(enumerate(fp), desc='Getting wikipedia types'):
            tokens = line.strip().lower().split(' ')
            # token = [x.lower() for x in line]
            types |= set(tokens)

            if (i % 1000000) == 0:
                # tqdm.desc = 'Processed %d word types' % len(types)
                tqdm.write('Processed %d word types' % len(types))
                # tqdm.update()

    return types


def process(src_file, tgt_file):
    types = process_data(src_file)
    util.write_data(tgt_file, types)


def main():
    args = get_args()
    logging.info(args)

    process(args.src_file, args.dict_file)


if __name__ == '__main__':
    main()
