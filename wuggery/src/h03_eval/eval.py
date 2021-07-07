import sys
import os

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders
from h02_learn.model import LstmLM, CondLstmLM
from h02_learn.train import evaluate
from util import argparser
from util import util
from util import constants


def get_args():
    # Data
    argparser.add_argument('--language', type=str, required=True)
    argparser.add_argument('--batch-size', type=int, default=512)

    args = argparser.parse_args()
    args.results_file = '%s/%s/losses__%s.csv' % (args.results_path, args.language, args.model)
    return args


def get_model_cls(fpath):
    model_name = fpath.split('/')[-2]
    model_type = model_name.split('_')[0]
    if model_type == 'lstm':
        model_cls = LstmLM
    elif model_type == 'cond-lstm':
        model_cls = CondLstmLM
    else:
        raise ValueError('Invalid model type when loading model: %s' % (model_type))
    return model_cls


def load_model(fpath):
    model_cls = get_model_cls(fpath)
    return model_cls.load(fpath).to(device=constants.device).eval()


def get_losses(model, trainloader, devloader, testloader):
    train_loss = evaluate(trainloader, model)
    dev_loss = evaluate(devloader, model)
    test_loss = evaluate(testloader, model)
    return train_loss, dev_loss, test_loss


def count_chars(trainloader, devloader, testloader):
    n_chars = sum([(y != 0).sum() - y.shape[0] for _, y in trainloader])
    n_chars += sum([(y != 0).sum() - y.shape[0] for _, y in devloader])
    n_chars += sum([(y != 0).sum() - y.shape[0] for _, y in testloader])
    return n_chars.item()


def count_words(trainloader, devloader, testloader):
    n_words = sum([y.shape[0] for _, y in trainloader])
    n_words += sum([y.shape[0] for _, y in devloader])
    n_words += sum([y.shape[0] for _, y in testloader])
    return n_words


def eval_lang(lang, model_path, dataloader):
    trainloader, devloader, testloader, _ = dataloader

    model = load_model(model_path)
    model_name = model_path.split('/')[-2]

    train_loss, dev_loss, test_loss = get_losses(
        model, trainloader, devloader, testloader)
    n_chars = count_chars(
        trainloader, devloader, testloader)
    n_words = count_words(
        trainloader, devloader, testloader)
    avg_len = n_chars / n_words

    print(('Language: %s Train size: %d Dev size: %d Test size: %d' +
           '\tTraining loss: %.4f Dev loss: %.4f Test loss: %.4f Avg len: %.2f') %
          (lang, len(trainloader.dataset),
           len(devloader.dataset), len(testloader.dataset),
           train_loss, dev_loss, test_loss,
           avg_len))

    return [[model_name, lang, avg_len, train_loss, dev_loss, test_loss]]


def main():
    args = get_args()
    folds = [list(range(8)), [8], [9]]
    model_name = '%s__nl_2-es_128-hs_128-d_0.3300/' % (args.model)


    results = [['model', 'language', 'avg_len', 'train_loss', 'dev_loss', 'test_loss']]
    dataloader = get_data_loaders(
        args.language, args.data_path, folds, args.batch_size, tag=args.tag)
    model_path = os.path.join(args.checkpoints_path, args.language, model_name)

    results += eval_lang(args.language, model_path, dataloader)

    util.write_csv(args.results_file, results)


if __name__ == '__main__':
    main()
