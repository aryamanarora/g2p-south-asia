import sys
import os
from functools import partial
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

# sys.path.append('./src/')
sys.path.append('./transducer/src/')
from decode_transducer import decode_top_k
import model
import util
from dataloader import SIGMORPHON2017Task1, BOS, EOS, BOS_IDX, EOS_IDX, UNK_IDX
from decoding import get_decode_fn
from model import dummy_mask
from trainer import DEV, TEST
from train import Trainer
from transformer import Transformer

tqdm.monitor_interval = 0

tqdm = partial(tqdm, bar_format="{l_bar}{r_bar}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WUGS = 'wugs'


class WugsDataloader(SIGMORPHON2017Task1):
    def __init__(
        self,
        tags: str,
        train_file: List[str],
        dev_file: List[str],
        wug_file: List[str],
    ):
        super().__init__(
            train_file=train_file, dev_file=dev_file)
        self.tags = tags
        self.wug_file = wug_file
        self.nb_wug = 1185

    def wug_sample(self):
        yield from self._sample(self.wug_file)

    def read_file(self, file, is_wug=False):
        if not is_wug:
            for x in super().read_file(file):
                yield x
        else:
            with open(file, "r", encoding="utf-8") as fp:
                for line in fp.readlines():
                    lemma = line.strip()
                    if not lemma:
                        continue

                    yield list(lemma), [], self.tags.split(";")

    def _iter_helper(self, file):
        for lemma, word, tags in self.read_file(file, is_wug=True):
            src = [self.source_c2i[BOS]]
            for tag in tags:
                src.append(self.attr_c2i.get(tag, UNK_IDX))
            for char in lemma:
                src.append(self.source_c2i.get(char, UNK_IDX))
            src.append(self.source_c2i[EOS])
            trg = [self.target_c2i[BOS]]
            for char in word:
                trg.append(self.target_c2i.get(char, UNK_IDX))
            trg.append(self.target_c2i[EOS])
            yield src, trg


def output2str(output, data):
    output = list(map(list, zip(*output)))
    predictions = []
    for x in output:
        pred = []
        for char in data.decode_target(x)[1:-1]:
            if char == EOS:
                break
            pred += [char]
        predictions += [''.join(pred)]
    return predictions


def decode_entropy_transformer(
        transducer, data, src_sentence, max_len=100, trg_bos=BOS_IDX, trg_eos=EOS_IDX):
    assert isinstance(transducer, Transformer)
    transducer.eval()
    src_mask = dummy_mask(src_sentence)
    src_mask = (src_mask == 0).transpose(0, 1)
    enc_hs = transducer.encode(src_sentence, src_mask)

    batch_size = 500

    output = [[trg_bos] * batch_size]
    enc_hs = enc_hs.repeat(1, batch_size, 1)
    src_mask = src_mask.repeat(batch_size, 1)
    logprobs = torch.zeros(batch_size, device=DEVICE)
    finished = (torch.zeros(batch_size, device=DEVICE) == 1)

    for _ in range(max_len):
        output_tensor = torch.tensor(output, device=DEVICE).view(len(output), batch_size)
        trg_mask = (output_tensor == 0).transpose(0, 1)

        word_logprob = transducer.decode(enc_hs, src_mask, output_tensor, trg_mask)
        word_logprob = word_logprob[-1]

        word = torch.multinomial(torch.exp(word_logprob), 1)
        finished |= (word.squeeze() == trg_eos)

        idx_logprobs = word_logprob.gather(1, word).squeeze()
        idx_logprobs[finished] = 0
        logprobs += idx_logprobs
        output.append(word.squeeze().tolist())

        if finished.all():
            break

    # predictions = output2str(output, data)
    entropy = - logprobs.mean()
    return entropy


class Evaluator(Trainer):

    def set_args(self):
        # fmt: off
        super().set_args()
        parser = self.parser
        parser.add_argument('--wug_file', default=None, type=str)
        parser.add_argument('--tgt_tag', required=True, type=str)
        parser.add_argument('--tgt_file', required=True, type=str)
        # fmt: on

    def load_data(self, dataset, train, dev, wug_file, tgt_tag):
        assert self.data is None
        self.tgt_tag = tgt_tag
        self.data = WugsDataloader(tgt_tag, train, dev, wug_file)

        logger = self.logger
        logger.info("src vocab size %d", self.data.source_vocab_size)
        logger.info("trg vocab size %d", self.data.target_vocab_size)
        logger.info("src vocab %r", self.data.source[:500])
        logger.info("trg vocab %r", self.data.target[:500])

    def iterate_instance(self, mode):
        if mode == WUGS:
            return self.data.wug_sample, self.data.nb_wug
        else:
            super().iterate_instance(mode)

    def get_entropies(self, mode, tgt_file, decode_fn):
        self.model.eval()
        cnt = 0
        n_beams = 5
        sampler, nb_instance = self.iterate_instance(mode)
        decode_fn.reset()
        with open(tgt_file, "w") as fp:
            beam_str = '\t'.join(['beam_%d' % (x + 1) for x in range(0, n_beams)])
            fp.write(f'src\tgreedy\t{beam_str}\tentropy\n')
            for src, trg in tqdm(sampler(), total=nb_instance):

                entropy = decode_entropy_transformer(
                    self.model, self.data, src)

                top_k = decode_top_k(self.model, src, nb_beam=n_beams, k=n_beams)
                top_k = [self.data.decode_target(x) for x in top_k]
                top_k = [''.join(x) for x in top_k]

                pred, _ = decode_fn(self.model, src)
                pred = self.data.decode_target(pred)
                pred = ''.join(pred)

                tags_len = len(self.data.tags.split(';'))
                src = self.data.decode_source(src)[1 + tags_len:-1]
                src = ''.join(src)

                top_k_str = '\t'.join(top_k)
                fp.write(f'{src}\t{pred}\t{top_k_str}\t{entropy}\n')
                cnt += 1

        self.logger.info(f"finished decoding {cnt} {mode} instance")

    def checklist_before_run(self):
        assert self.data is not None, "call load_data before run"
        assert self.model is not None, "call build_model before run"

    def test(self, decode_fn, tgt_file):
        self.logger.info("decoding dev set")
        self.get_entropies(WUGS, tgt_file, decode_fn)

    def run(self, start_epoch, decode_fn=None):
        self.checklist_before_run()
        params = self.params

        with torch.no_grad():
            self.test(decode_fn, params.tgt_file)


def main():
    evaluator = Evaluator()
    params = evaluator.params
    assert params.load and params.load != "0"

    decode_fn = get_decode_fn(
        params.decode, params.max_decode_len, params.decode_beam_size
    )
    evaluator.load_data(params.dataset, params.train, params.dev, params.wug_file, params.tgt_tag)
    evaluator.setup_evalutator()

    if params.load == "smart":
        start_epoch = evaluator.smart_load_model(params.model) + 1
    else:
        start_epoch = evaluator.load_model(params.load) + 1

    evaluator.run(start_epoch, decode_fn=decode_fn)


if __name__ == "__main__":
    main()
