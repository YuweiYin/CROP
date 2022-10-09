import argparse
# import faiss
import os
import torch
import numpy as np
import sentencepiece as spm

from sklearn.preprocessing import normalize
from scipy.stats import entropy
from collections import OrderedDict
from torch.utils.data.dataloader import DataLoader

from fairseq import checkpoint_utils
from fairseq.data.dictionary import Dictionary
from fairseq.data import data_utils, FairseqDataset, Dictionary
from fairseq.tasks import register_task, FairseqTask
from fairseq.models.transformer import (
    TransformerModel,
    transformer_vaswani_wmt_en_de_big
)


def mapping(languages: str) -> dict:
    return dict(
        tuple(pair.split(":"))
        for pair in languages.strip().replace("\n", "").split(",")
    )


ISO2M100 = mapping(
    """
afr:af,amh:am,ara:ar,asm:as,ast:ast,azj:az,bel:be,ben:bn,bos:bs,bul:bg,
cat:ca,ceb:ceb,ces:cs,ckb:ku,cym:cy,dan:da,deu:de,ell:el,eng:en,est:et,
fas:fa,fin:fi,fra:fr,ful:ff,gle:ga,glg:gl,guj:gu,hau:ha,heb:he,hin:hi,
hrv:hr,hun:hu,hye:hy,ibo:ig,ind:id,isl:is,ita:it,jav:jv,jpn:ja,kam:kam,
kan:kn,kat:ka,kaz:kk,kea:kea,khm:km,kir:ky,kor:ko,lao:lo,lav:lv,lin:ln,
lit:lt,ltz:lb,lug:lg,luo:luo,mal:ml,mar:mr,mkd:mk,mlt:mt,mon:mn,mri:mi,
msa:ms,mya:my,nld:nl,nob:no,npi:ne,nso:ns,nya:ny,oci:oc,orm:om,ory:or,
pan:pa,pol:pl,por:pt,pus:ps,ron:ro,rus:ru,slk:sk,slv:sl,sna:sn,snd:sd,
som:so,spa:es,srp:sr,swe:sv,swh:sw,tam:ta,tel:te,tgk:tg,tgl:tl,tha:th,
tur:tr,ukr:uk,umb:umb,urd:ur,uzb:uz,vie:vi,wol:wo,xho:xh,yor:yo,zho_simp:zh,
zho_trad:zh,zul:zu
"""
)

ISO2OUR = mapping(
    """
afr:af,amh:am,ara:ar,asm:as,ast:ast,azj:az,bel:be,ben:bn,bos:bs,bul:bg,
cat:ca,ceb:ceb,ces:cs,ckb:ku,cym:cy,dan:da,deu:de,ell:el,eng:en,est:et,
fas:fa,fin:fi,fra:fr,ful:ff,gle:ga,glg:gl,guj:gu,hau:ha,heb:he,hin:hi,
hrv:hr,hun:hu,hye:hy,ibo:ig,ind:id,isl:is,ita:it,jav:jv,jpn:ja,kam:kam,
kan:kn,kat:ka,kaz:kk,kea:kea,khm:km,kir:ky,kor:ko,lao:lo,lav:lv,lin:ln,
lit:lt,ltz:lb,lug:lg,luo:luo,mal:ml,mar:mr,mkd:mk,mlt:mt,mon:mn,mri:mi,
msa:ms,mya:my,nld:nl,nob:no,npi:ne,nso:ns,nya:ny,oci:oc,orm:om,ory:or,
pan:pa,pol:pl,por:pt,pus:ps,ron:ro,rus:ru,slk:sk,slv:sl,sna:sn,snd:sd,
som:so,spa:es,srp:sr,swe:sv,swh:sw,tam:ta,tel:te,tgk:tg,tgl:tl,tha:th,
tur:tr,ukr:uk,umb:umb,urd:ur,uzb:uz,vie:vi,wol:wo,xho:xh,yor:yo,zho_simp:zh,
zho_trad:zt,zul:zu
"""
)

MODEL_TYPE = "Our"  # Our and M2M
ISO2SIMP = ISO2OUR if MODEL_TYPE == "Our" else ISO2M100


def _lang_token(lang: str):
    return '__{}__'.format(lang)


def _lang_token_index(dic: Dictionary, lang: str):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, \
        'cannot find language token for lang {}'.format(lang)
    return idx


def collate(
        samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
        input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class LanguagePairLangidDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing
            (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(
            self, src, src_sizes, src_dict, src_langs,
            tgt=None, tgt_sizes=None, tgt_dict=None, tgt_langs=None,
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
            encoder_langtok='tgt', decoder_langtok=False
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_langs = src_langs
        self.tgt_langs = tgt_langs
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.encoder_langtok = encoder_langtok
        self.decoder_langtok = decoder_langtok

    def get_encoder_langtok(self, src_lang, tgt_lang):
        if self.encoder_langtok == 'src' and src_lang is not None:
            return _lang_token_index(self.src_dict, src_lang)
        elif self.encoder_langtok == 'tgt' and tgt_lang is not None:
            return _lang_token_index(self.src_dict, tgt_lang)
        return self.src_dict.eos()

    def get_decoder_langtok(self, tgt_lang):
        if self.decoder_langtok and tgt_lang is not None:
            return _lang_token_index(self.tgt_dict, tgt_lang)
        return self.tgt_dict.eos()

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        tgt_lang = self.tgt_langs[index] if self.tgt_langs is not None else None
        src_lang = self.src_langs[index]

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if src_item[-1] == eos:
                src_item = src_item[:-1]

        # append langid to source end
        if self.encoder_langtok is not None:
            new_eos = self.get_encoder_langtok(src_lang, tgt_lang)
            src_item = torch.cat([src_item, torch.LongTensor([new_eos])])

        # append langid to target start
        if self.decoder_langtok:
            new_eos = self.get_decoder_langtok(tgt_lang)
            tgt_item = torch.cat([torch.LongTensor([new_eos]), tgt_item])

        """
        print("[debug]==========================")
        print("encoder_langtok: {}".format(self.encoder_langtok))
        print("{}-src: {}".format(index, self.src_dict.string(src_item)))
        print("{}-tgt: {}".format(index, self.src_dict.string(tgt_item) if tgt_item is not None else "None"))
        print("[debug]==========================")
        """

        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    @property
    def sizes(self):
        return np.maximum(self.src_sizes, self.tgt_sizes)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
                getattr(self.src, 'supports_prefetch', False)
                and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)


class TestTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        parser.add_argument('--use_proj', default=False, action='store_true')
        parser.add_argument('--use_bn', default=False, action='store_true')
        parser.add_argument('--tokens_per_sample', type=int, default=512)
        parser.add_argument('--cxlm_queue_size', type=int, default=131072)
        parser.add_argument('--roberta_model_path', type=str, default='')
        parser.add_argument('--fp16', default=False, action='store_true')

    @classmethod
    def setup_task(cls, args, dictionary, **kwargs):
        return cls(args, dictionary)

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        lang_pairs = "de-en,cs-en,et-en,fi-en,fr-en,gu-en,hi-en,lv-en,ro-en,tr-en".split(',')
        sorted_langs = sorted(list({x for lang_pair in lang_pairs for x in lang_pair.split('-')}))
        for lang_to_add in sorted_langs:
            _land_idx = self.dictionary.add_symbol(_lang_token(lang_to_add))
            self.dictionary.add_special_symbol(_land_idx)
        # self.mask_idx = self.dictionary.add_symbol('<mask>')

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


def load_test_task(args, vocab):
    task = TestTask.setup_task(args, vocab)
    return task


def load_mt_model(ckpt_path, vocab):
    models, _model_args = checkpoint_utils.load_xlmt_model_ensemble(ckpt_path.split(), src_dict=vocab, tgt_dict=vocab)
    return models


def get_dataset(args, vocab):
    with open(args.src_fn) as fp: src_lines = [l for l in fp]
    src = [torch.tensor([vocab.indices[w] for w in src_line.split()] + [vocab.eos()]) for src_line in src_lines]
    src_sizes = np.array([len(s)] for s in src_lines)
    # src_lang = [args.src_fn.split('.')[-1]] * len(src)
    src_lang = ["en"] * len(src)
    dataset = LanguagePairLangidDataset(src, src_sizes, vocab, src_lang)
    return dataset


def run(args):
    args.tokens_per_sample = 512
    vocab = Dictionary.load(args.vocab_path)
    for lang in ISO2SIMP.values():
        vocab.add_symbol(f"__{lang}__")
    vocab.pad_to_multiple_(padding_factor=8)
    model = load_mt_model(args.ckpt_path, vocab)
    dataset = get_dataset(args, vocab)

    dl = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=dataset.collater)
    model.eval()
    with open("{}.{}".format(args.src_fn, args.suffix), "w") as w:
        with torch.no_grad():
            for sample in dl:
                encoder_out = model.encoder(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'],
                                            return_all_hiddens=True)
                reprs = [[round(w, 3) for w in ws] for ws in encoder_out.encoder_out[0, :, :].tolist()]
                for repr in reprs:
                    repr = [str(w) for w in repr]
                    w.write("{}\n".format(" ".join(repr)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', default="/path/to/XMT/x2e/model.pt", type=str)
    # parser.add_argument('--roberta_model_path', default="", type=str)
    parser.add_argument("--src_fn", type=str, default="/path/to/XMT/data/similarity/parallel.en")
    parser.add_argument('--vocab_path', default="/path/to/xlm-t/dict.txt", type=str)
    parser.add_argument('--wa_layer', default=23, type=int)
    parser.add_argument('--suffix', default="repr", type=str)
    args = parser.parse_args()
    run(args)
