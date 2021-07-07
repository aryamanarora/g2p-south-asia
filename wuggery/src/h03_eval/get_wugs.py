import sys
import os

sys.path.append('./src/')
# from h01_data.northeuralex import Northeuralex
from h01_data.unimorph import Unimorph
from h03_eval.eval import load_model
from util import argparser
from util import util


def get_args():
    # Data
    argparser.add_argument('--language', type=str, required=True)
    argparser.add_argument('--batch-size', type=int, default=200)
    argparser.add_argument('--n-wugs', type=int, default=1000)
    # argparser.add_argument('--sampled-tag', type=str, default='V;NFIN')
    argparser.add_argument('--pos-class', type=str, default='verb')
    argparser.add_argument("--dict-file", type=str, required=True)

    args = argparser.parse_args()
    if args.pos_class == 'verb':
        args.sampled_tag = 'V;NFIN'
    elif args.pos_class == 'noun':
        args.sampled_tag = 'N;ACC;SG'
    else:
        raise ValueError('Invalid pos class %s' % args.pos_class)

    # if args.tag:
    #     args.results_file = '%s/%s/wugs__%s__%s.csv' % (args.results_path, args.language, args.model, args.sampled_tag)
    # else:
    #     args.results_file = '%s/%s/wugs__%s.csv' % (args.results_path, args.language, args.model)
    args.results_file = '%s/%s/wugs__%s__%s.csv' % (args.results_path, args.language, args.model, args.pos_class)

    return args


def get_words(data):
    words_info = data[0]
    words = [' '.join(list(x['word']))
             for fold in words_info for x in fold]
    return set(words)


def get_wugs(lang, model_path, data, args):
    model = load_model(model_path)

    words = get_words(data)
    types = util.read_data(args.dict_file)
    types |= words

    samples = set([])
    n_wugs = args.n_wugs

    while len(samples) < n_wugs:
        new_samples = model.sample(args)
        samples |= (set(new_samples) - types)
    print('Got %d examples for %s' % (len(samples), lang))

    return [list(samples)]


def transpose(results):
    max_len = max([len(x) for x in results])
    inverse = [[''] * len(results) for _ in range(max_len)]
    for i, row in enumerate(results):
        for j, x in enumerate(row):
            inverse[j][i] = x

    return inverse


def main():
    args = get_args()
    model_name = '%s__nl_2-es_128-hs_128-d_0.3300/' % (args.model)

    results = []
    data = Unimorph.read_data(args.data_path, args.language)
    model_path = os.path.join(args.checkpoints_path, args.language, model_name)

    results += get_wugs(args.language, model_path, data, args)

    results = transpose(results)
    util.write_csv(args.results_file, results)


if __name__ == '__main__':
    main()
