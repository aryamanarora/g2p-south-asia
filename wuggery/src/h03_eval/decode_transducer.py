import sys
import torch

sys.path.append('./transducer/src/')
from decoding import Beam

from dataloader import BOS_IDX, EOS_IDX, STEP_IDX
from transformer import Transformer
from model import dummy_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decode_top_k(
    transducer,
    src_sentence,
    k=10,
    max_len=50,
    nb_beam=10,
    norm=True,
    trg_bos=BOS_IDX,
    trg_eos=EOS_IDX,
):
    assert isinstance(transducer, Transformer)

    def score(beam):
        assert isinstance(beam, Beam)
        if norm:
            return -beam.log_prob / beam.seq_len
        return -beam.log_prob

    transducer.eval()
    src_mask = dummy_mask(src_sentence)
    src_mask = (src_mask == 0).transpose(0, 1)
    enc_hs = transducer.encode(src_sentence, src_mask)

    input_ = torch.tensor([trg_bos], device=DEVICE).view(1, 1)
    start = Beam(1, 0, None, input_, "", None)
    beams = [start]
    finish_beams = []
    for _ in range(max_len):
        next_beams = []
        for beam in sorted(beams, key=score)[:nb_beam]:
            trg_mask =  (beam.input)
            trg_mask = (trg_mask == 0).transpose(0, 1)

            word_logprob = transducer.decode(enc_hs, src_mask, beam.input, trg_mask)
            word_logprob = word_logprob[-1]

            topk_log_prob, topk_word = word_logprob.topk(nb_beam)
            topk_log_prob = topk_log_prob.view(nb_beam, 1)
            topk_word = topk_word.view(nb_beam, 1)
            for log_prob, word in zip(topk_log_prob, topk_word):
                if word == trg_eos:
                    new_beam = Beam(
                        beam.seq_len + 1,
                        beam.log_prob + log_prob.item(),
                        None,
                        None,
                        beam.partial_sent,
                        None,
                    )
                    finish_beams.append(new_beam)
                else:
                    new_beam = Beam(
                        beam.seq_len + 1,
                        beam.log_prob + log_prob.item(),
                        None,
                        torch.cat((beam.input, word.view(1, 1))),
                        " ".join([beam.partial_sent, str(word.item())]),
                        None,
                    )
                    next_beams.append(new_beam)
        beams = next_beams
    finish_beams = finish_beams if finish_beams else next_beams
    top_k = sorted(finish_beams, key=score)[0:k]
    return [list(map(int, x.partial_sent.split())) for x in top_k]
