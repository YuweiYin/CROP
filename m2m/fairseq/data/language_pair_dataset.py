# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils, indexed_dataset
import random
from fairseq.data.encoders.utils import get_whole_word_mask
import itertools
import nltk
logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if (
            alignment[:, 0].max().item() >= src_len - 1
            or alignment[:, 1].max().item() >= tgt_len - 1
        ):
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(
            align_tgt, return_inverse=True, return_counts=True
        )
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1.0 / align_weights.float()

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        if pad_to_length is not None and 'prepend_target' in pad_to_length.keys():
            preprend_target = merge(
                "prepend_target",
                left_pad=left_pad_source,
                pad_to_length=pad_to_length["prepend_target"] if pad_to_length is not None else None,
            )
            preprend_target = preprend_target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    else:
        ntokens = src_lengths.sum().item()
        tgt_lengths=None

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
        "tgt_lengths": tgt_lengths
    }
    if pad_to_length is not None and "prepend_target" in pad_to_length.keys():
        batch["net_input"]["prepend_target"] = preprend_target
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )

    # if samples[0].get("alignment", None) is not None:
    #     alignments = [samples[align_idx]["alignment"] for align_idx in sort_order]
    #     batch["alignments"] = alignments

    if samples[0].get("constraints", None) is not None:
        # Collate the packed constraints across the samples, padding to
        # the length of the longest sample.
        lens = [sample.get("constraints").size(0) for sample in samples]
        max_len = max(lens)
        constraints = torch.zeros((len(samples), max(lens))).long()
        for i, sample in enumerate(samples):
            constraints[i, 0 : lens[i]] = samples[i].get("constraints")
        batch["constraints"] = constraints

    return batch


class LanguagePairDataset(FairseqDataset):
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
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """

    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        tgt_lang_id=None,
        pad_to_multiple=1,
        args=None
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(
                tgt
            ), "Source and target must contain the same number of examples {} | {} {}".format(tgt._path, len(src), len(tgt))
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert (
                self.tgt_sizes is not None
            ), "Both source and target needed when alignments are provided"
            assert len(align_dataset) == len(src) and len(align_dataset) == len(tgt), f"Align Dataset Lines: {tgt._path if args.langtoks['main'][1] is None else tgt.dataset._path} | {len(align_dataset)}"
            self.src_lang = self.get_lang_from_dataset(self.src)
            self.tgt_lang = self.get_lang_from_dataset(self.tgt)
            self.args = args
            self.slot_method = args.slot_method
            self.max_slot_num = args.max_slot_num
            self.max_span_length = args.max_span_length
            self.slot_prob = args.slot_prob
            self.subword_prob = args.subword_prob
            self.mask_whole_word = get_whole_word_mask(self.args, self.src_dict)
            self.pad_index = self.src_dict.pad_index
            self.slot_idxs = []
            for w in self.src_dict.symbols[-100:]:
                if "SLOT" in w:
                    self.slot_idxs.append(self.src_dict.index(w))
                self.slot_idxs = self.slot_idxs[:self.max_slot_num]
            if self.mask_whole_word is not None:
                self.mask_whole_word = self.mask_whole_word.bool()
            self.construct_phrase_level_align_dataset = args.construct_phrase_level_align_dataset
            if self.subword_prob < 1.0 and self.construct_phrase_level_align_dataset:
                self.pattern_src = []
                self.pattern_tgt = []
                logger.info("Start Constructing Phrase-level Patterns...")
                for index in range(len(self.align_dataset)):
                    src_item, tgt_item = self.prepare_phrase_level_pattern_based_sentence(index, self.align_dataset[index][0].tolist(), self.align_dataset[index][1].tolist(), self.src[index], self.tgt[index])
                    self.pattern_src.append(src_item)
                    self.pattern_tgt.append(tgt_item)
                    if index % 10000:
                        logger.info(f"Processing {index} examples | All {len(self.align_dataset)} examples")
                logger.info("Complete Constructing Phrase-level Patterns...")
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset

            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info("bucketing source lengths: {}".format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info(
                    "bucketing target lengths: {}".format(list(self.tgt.buckets))
                )

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple


    def get_lang_from_dataset(self, dataset):
        index_dataset = dataset
        while not isinstance(index_dataset, indexed_dataset.MMapIndexedDataset):  # Recover to the IndexDataset
            index_dataset = index_dataset.dataset
        return index_dataset._path.split(".")[-1]


    def get_batch_shapes(self):
        return self.buckets


    def word_starts(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        return is_word_start


    def get_align_dict(self, aligns):
        align_dict = {}
        for ali in aligns:
            sid, tid = ali
            if sid not in align_dict:
                align_dict[sid] = [tid]
            else:
                align_dict[sid].append(tid)
        return align_dict

    def convert_tensor2str(self, x):
        return [str(w) for w in x.tolist()]


    def convert_str2tensor(self, x):
        x = " ".join(x).split()
        x = [int(w) for w in x]
        return torch.LongTensor(x)


    def insert_values(self, x, index, insert_values):
        #Calculate the new index
        sorted_index, sort_order = torch.sort(index[torch.arange(0, len(index) - 1, 2, dtype=index.dtype)])
        add_index = torch.zeros_like(index)
        add_index[sort_order * 2] = torch.arange(0, len(index)//2, dtype=index.dtype) * 2
        add_index[sort_order * 2 + 1] = torch.arange(0, len(index)//2, dtype=index.dtype) * 2 + 1
        #
        # sorted_index, sort_order = torch.sort(index)
        # add_index = torch.zeros_like(index)
        # add_index[sort_order] = torch.range(0, len(index) - 1, dtype=index.dtype)
        new_index = index + add_index
        y = torch.zeros(x.size(0) + new_index.size(0), dtype=x.dtype)
        mask = torch.zeros(x.size(0) + new_index.size(0), dtype=torch.bool)
        mask.scatter_(dim=0, index=new_index, src=torch.ones(size=new_index.size(), dtype=torch.bool))
        y.scatter_(dim=0, index=new_index, src=insert_values)
        y.masked_scatter_(mask=~mask, source=x)
        return y


    def insert_sentence_with_aligns(self, src_item, tgt_item, mask_aligns):
        if len(mask_aligns) > 0:
            if isinstance(mask_aligns[0][0], int):
                src_mask_aligns = list(itertools.chain(*[[align[0], align[0] + 1] for align in mask_aligns]))
                tgt_mask_aligns = list(itertools.chain(*[[align[1], align[1] + 1] for align in mask_aligns]))
            else:
                src_mask_aligns = list(itertools.chain(*[[align[0][0], align[0][1]] for align in mask_aligns]))
                tgt_mask_aligns = list(itertools.chain(*[[align[1][0], align[1][1]] for align in mask_aligns]))
            try:
                insert_values = list(itertools.chain(*[[self.slot_idxs[i], self.slot_idxs[i]] for i in range(len(mask_aligns))]))
                src_item = self.insert_values(src_item, torch.LongTensor(src_mask_aligns), torch.LongTensor(insert_values))
                tgt_item = self.insert_values(tgt_item, torch.LongTensor(tgt_mask_aligns), torch.LongTensor(insert_values))
            except:
                logger.info(f"Error mask_aligns: {mask_aligns}")
        return src_item, tgt_item


    def remove_overlapped_aligns(self, phrases, src_item, tgt_item):
        if len(phrases) <= 1:
            return phrases
        span_mask_aligns = []  # remove intersected phrases
        if isinstance(phrases[0][0], int):
            span_mask_aligns = list({phrase[0]: phrase[1] for phrase in phrases}.items())
            span_mask_aligns = list({span_mask_align[1]: span_mask_align[0] for span_mask_align in span_mask_aligns}.items())
            span_mask_aligns = [[span_mask_align[1], span_mask_align[0]] for span_mask_align in span_mask_aligns]
            span_mask_aligns = list(span_mask_aligns)
        else:
            # src mask
            src_masks = torch.zeros(len(src_item), dtype=torch.bool)
            # tgt mask
            tgt_masks = torch.zeros(len(tgt_item), dtype=torch.bool)
            for span_mask_align in phrases:
                src_span_mask_align = span_mask_align[0]
                tgt_span_mask_align = span_mask_align[1]
                if not src_masks[src_span_mask_align[0]: src_span_mask_align[1]].any() and not tgt_masks[tgt_span_mask_align[0]: tgt_span_mask_align[1]].any():
                    span_mask_aligns.append(span_mask_align)
                    src_masks[src_span_mask_align[0]: src_span_mask_align[1]] = True
                    tgt_masks[tgt_span_mask_align[0]: tgt_span_mask_align[1]] = True
        return span_mask_aligns


    def choose_aligns(self, aligns):
        mask_num = min(len(aligns), random.randint(0, self.max_slot_num))
        align_idxs = np.random.choice(np.arange(len(aligns)), mask_num, replace=False)
        mask_aligns = [aligns[align_idx] for align_idx in align_idxs]
        return mask_aligns


    def save_word_based_phrase(self, phrases, src_item, tgt_item):
        return list(filter(lambda phrase: self.mask_whole_word[src_item[phrase[0][0]]] and self.mask_whole_word[src_item[phrase[0][1]]] and self.mask_whole_word[tgt_item[phrase[1][0]]] and self.mask_whole_word[tgt_item[phrase[1][1]]], phrases))
        # word_based_phrase = []
        # for phrase in range(len(phrases)):
        #     source_phrase, target_phrase = phrase[0], phrase[1]
        #     if self.mask_whole_word[source_phrase[0]] and self.mask_whole_word[source_phrase[1]] and self.mask_whole_word[source_phrase[0]] and self.mask_whole_word[source_phrase[1]]:
        #         word_based_phrase.append(phrase)
        # return word_based_phrase


    def prepare_phrase_level_pattern_based_sentence(self, index, aligns, phrases, src_item, tgt_item):
        if random.random() < self.slot_prob:  # ensure 1 alignment at least, slot_prob
            if random.random() <= self.subword_prob:
                if len(aligns) == 0 or len(aligns[0]) != 2:
                    return src_item, tgt_item
                # random.shuffle(aligns)
                mask_aligns = self.choose_aligns(aligns)
                mask_aligns = self.remove_overlapped_aligns(mask_aligns, src_item, tgt_item)
                if self.slot_method == "replace":
                    slot_idxs = np.random.choice(self.slot_idxs, len(self.slot_idxs), replace=True)
                    slot_idxs = torch.LongTensor([slot_idxs[i] for i in range(len(mask_aligns))])
                    src_mask_aligns = torch.LongTensor([mask_align[0] for mask_align in mask_aligns])
                    tgt_mask_aligns = torch.LongTensor([mask_align[1] for mask_align in mask_aligns])
                    src_item[src_mask_aligns] = slot_idxs
                    tgt_item[tgt_mask_aligns] = slot_idxs
                elif self.slot_method == "insert":
                    src_item, tgt_item = self.insert_sentence_with_aligns(src_item, tgt_item, mask_aligns)
                    # src_words = self.convert_tensor2str(src_item)
                    # tgt_words = self.convert_tensor2str(tgt_item)
                    # for i in range(len(mask_aligns)):
                    #     src_mask_align = mask_aligns[i][0]
                    #     tgt_mask_align = mask_aligns[i][1]
                    #     src_words[src_mask_align] = f"{self.slot_idxs[i]} {src_words[src_mask_align]} {self.slot_idxs[i]}"
                    #     tgt_words[tgt_mask_align] = f"{self.slot_idxs[i]} {tgt_words[tgt_mask_align]} {self.slot_idxs[i]}"
                    # src_item = self.convert_str2tensor(src_words)
                    # tgt_item = self.convert_str2tensor(tgt_words)
                else:
                    logger.info("SLOT METHOD: {}".format(self.slot_method))
                if self.args.debug:
                    # if "▁ander __SLOT0__ __SLOT1__ ▁styl __SLOT0__" in self.src_dict.string(src_item):
                    #     logger.info(src_item)
                    logger.info("Token Level | src: {}".format(self.src_dict.string(src_item)))
                    logger.info("Token Level | tgt: {}".format(self.src_dict.string(tgt_item)))
            else:
                if len(phrases) == 0 or len(phrases[0]) != 2:
                    return src_item, tgt_item
                # Phrase level mask
                phrases = list(filter(lambda phrase: phrase[0][1] - phrase[0][0] <= self.max_span_length, phrases))
                # phrases = self.save_word_based_phrase(phrases, src_item, tgt_item)
                phrases.append([[1, len(src_item) - 1], [1, len(tgt_item) - 1]])  # add the SLOT for whole sentence
                # random.shuffle(phrases)
                span_mask_aligns = self.choose_aligns(phrases)
                span_mask_aligns = self.remove_overlapped_aligns(span_mask_aligns, src_item, tgt_item)
                if self.slot_method == "replace":
                    slot_idxs = np.random.choice(self.slot_idxs, len(self.slot_idxs), replace=True)
                    for i, span_mask_align in enumerate(span_mask_aligns):
                        src_span_mask_align = torch.arange(span_mask_align[0][0], span_mask_align[0][1])
                        tgt_span_mask_align = torch.arange(span_mask_align[1][0], span_mask_align[1][1])
                        src_item[src_span_mask_align] = torch.cat([torch.LongTensor([slot_idxs[i]]), torch.LongTensor(len(src_span_mask_align) - 1).fill_(self.pad_index)])
                        tgt_item[tgt_span_mask_align] = torch.cat([torch.LongTensor([slot_idxs[i]]), torch.LongTensor(len(tgt_span_mask_align) - 1).fill_(self.pad_index)])
                    # Remove Pad Index
                    src_item = src_item.masked_select(src_item != self.pad_index)
                    tgt_item = tgt_item.masked_select(tgt_item != self.pad_index)
                elif self.slot_method == "insert":
                    src_item, tgt_item = self.insert_sentence_with_aligns(src_item, tgt_item, span_mask_aligns)
                    # src_words = self.convert_tensor2str(src_item)
                    # tgt_words = self.convert_tensor2str(tgt_item)
                    # for i, span_mask_align in enumerate(span_mask_aligns):
                    #     src_words[span_mask_align[0][0]] = f"{self.slot_idxs[i]} {src_words[span_mask_align[0][0]]}"
                    #     src_words[span_mask_align[0][1] - 1] = f"{src_words[span_mask_align[0][1]]} {self.slot_idxs[i]}"
                    #     tgt_words[span_mask_align[1][0]] = f"{self.slot_idxs[i]} {tgt_words[span_mask_align[1][0]]}"
                    #     tgt_words[span_mask_align[1][1] - 1] = f"{tgt_words[span_mask_align[1][1]]} {self.slot_idxs[i]}"
                    # src_item = self.convert_str2tensor(src_words)
                    # tgt_item = self.convert_str2tensor(tgt_words)
                else:
                    logger.info("SLOT METHOD: {}".format(self.slot_method))
                if self.args.debug:
                    logger.info(f"Index {index} | Phrase Level | src: {self.src_dict.string(src_item)}")
                    logger.info(f"Index {index} | Phrase Level | tgt: {self.src_dict.string(tgt_item)}")
        return src_item, tgt_item


    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        if self.align_dataset is not None and self.slot_method != "None":
            if self.construct_phrase_level_align_dataset:
                src_item = self.pattern_src[index]
                tgt_item = self.pattern_tgt[index]
            else:
                aligns, phrases = self.align_dataset[index][0], self.align_dataset[index][1]
                if aligns is not None:
                    aligns = aligns.tolist()
                if phrases is not None:
                    phrases = phrases.tolist()
                src_item, tgt_item = self.prepare_phrase_level_pattern_based_sentence(index, aligns, phrases, src_item, tgt_item)

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
        }
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

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
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
                #res["net_input"]["tgt_lang_id"] = res["tgt_lang_id"]
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )
