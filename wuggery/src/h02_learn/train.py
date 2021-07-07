import sys
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders
from h02_learn.model import LstmLM, CondLstmLM
from h02_learn.train_info import TrainInfo
from util import argparser
from util import util
from util import constants


def get_model_name(args):
    fpath = '%s__nl_%d-es_%d-hs_%d-d_%.4f' % \
        (args.model, args.nlayers, args.embedding_size, args.hidden_size, args.dropout)
    return fpath


def get_args():
    # Data
    argparser.add_argument('--language', type=str, required=True)
    argparser.add_argument('--batch-size', type=int, default=32)
    # Model
    argparser.add_argument('--nlayers', type=int, default=2)
    argparser.add_argument('--embedding-size', type=int, default=128)
    argparser.add_argument('--hidden-size', type=int, default=128)
    argparser.add_argument('--dropout', type=float, default=.33)
    # Optimization
    argparser.add_argument('--eval-batches', type=int, default=20)
    argparser.add_argument('--wait-epochs', type=int, default=10)

    args = argparser.parse_args()
    args.wait_iterations = args.wait_epochs * args.eval_batches
    args.save_path = '%s/%s/%s/' % (args.checkpoints_path, args.language, get_model_name(args))
    return args


def get_model(alphabets, args):
    if args.model == 'lstm':
        return LstmLM(
            alphabets['chars'], args.embedding_size, args.hidden_size,
            nlayers=args.nlayers, dropout=args.dropout) \
            .to(device=constants.device)
    return CondLstmLM(
        alphabets['chars'], alphabets['tag'], args.embedding_size, args.hidden_size,
        nlayers=args.nlayers, dropout=args.dropout) \
        .to(device=constants.device)



def _evaluate(evalloader, model):
    criterion = nn.CrossEntropyLoss(ignore_index=0) \
        .to(device=constants.device)

    dev_loss, n_instances = 0, 0
    for x, y in evalloader:
        y_hat = model(x)
        loss = criterion(y_hat.reshape(-1, y_hat.shape[-1]), y.reshape(-1)).item() / math.log(2)
        batch_size = y.shape[0]
        dev_loss += loss * batch_size
        n_instances += batch_size

    return dev_loss / n_instances


def evaluate(evalloader, model):
    model.eval()
    with torch.no_grad():
        result = _evaluate(evalloader, model)
    model.train()
    return result


def train_batch(x, y, model, optimizer, criterion):
    optimizer.zero_grad()
    y_hat = model(x)
    loss = criterion(y_hat.reshape(-1, y_hat.shape[-1]), y.reshape(-1)) / math.log(2)
    loss.backward()
    optimizer.step()

    return loss.item()


def train_epoch(trainloader, devloader, model, optimizer, criterion, train_info):
    for x, y in trainloader:
        loss = train_batch(x, y, model, optimizer, criterion)
        train_info.new_batch(loss)

        if train_info.eval:
            dev_loss = evaluate(devloader, model)

            if train_info.is_best(dev_loss):
                model.set_best()
            elif train_info.finish:
                train_info.print_progress(dev_loss)
                return

            train_info.print_progress(dev_loss)


def train(trainloader, devloader, model, eval_batches, wait_iterations):
    optimizer = optim.AdamW(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0) \
        .to(device=constants.device)

    with tqdm(total=wait_iterations) as pbar:
        train_info = TrainInfo(pbar, wait_iterations, eval_batches)
        while not train_info.finish:
            train_epoch(trainloader, devloader, model,
                        optimizer, criterion, train_info)

    model.recover_best()


def save_results(model, train_loss, dev_loss, test_loss, results_fname):
    results = [['alphabet_size', 'embedding_size', 'hidden_size', 'nlayers',
                'dropout_p', 'train_loss', 'dev_loss', 'test_loss']]
    results += [[model.alphabet_size, model.embedding_size, model.hidden_size,
                 model.nlayers, model.dropout_p,
                 train_loss, dev_loss, test_loss]]
    util.write_csv(results_fname, results)


def save_checkpoints(model, train_loss, dev_loss, test_loss, save_path):
    util.mkdir(save_path)
    model.save(save_path)
    results_fname = save_path + '/results.csv'
    save_results(model, train_loss, dev_loss, test_loss, results_fname)


def main():
    args = get_args()
    folds = [list(range(8)), [8], [9]]

    trainloader, devloader, testloader, alphabets = \
        get_data_loaders(args.language, args.data_path, folds, args.batch_size, tag=args.tag)
    print('Language: %s Train size: %d Dev size: %d Test size: %d' %
          (args.language, len(trainloader.dataset),
           len(devloader.dataset), len(testloader.dataset)))
    print(args)

    model = get_model(alphabets, args)
    train(trainloader, devloader, model, args.eval_batches, args.wait_iterations)

    train_loss = evaluate(trainloader, model)
    dev_loss = evaluate(devloader, model)
    test_loss = evaluate(testloader, model)

    print('Final Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (train_loss, dev_loss, test_loss))

    save_checkpoints(model, train_loss, dev_loss, test_loss, args.save_path)


if __name__ == '__main__':
    main()
