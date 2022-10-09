import os
import argparse
import numpy as np
import logging
import collections
import string
import itertools
import linecache
import re
import copy
import langid
import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(  # must set the level
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dataset', '-source-dataset', type=str,
                        default=r'/path/to/NER/xtreme_MulDA/panx_processed_maxlen128/en/orig_data/train.xlmr',
                        help='input stream')
    parser.add_argument('--target-dataset', '-target-dataset', type=str,
                        default=r'/path/to/NER/xtreme_MulDA/MulDA/FINAL/', help='input stream')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/path/to/NER/xtreme_MulDA/panx_processed_maxlen128/en/train.xlmr',
                        help='input stream')
    parser.add_argument('--idx', '-idx', type=str,
                        default=r'/path/to/NER/xtreme_MulDA/panx_processed_maxlen128/en/train.xlmr.idx',
                        help='input stream')
    parser.add_argument('--sampling-method', '-sampling-method', default="down-sampling",
                        choices=["up-sampling", "down-sampling", "None"], help='input stream')
    parser.add_argument('--sampling-ratio', '-sampling-ratio', default=0.1, type=float, help='input stream')
    args = parser.parse_args()
    return args


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, langs=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          words: list. The words of the sequence.
          labels: (Optional) list. The labels for each word of the sequence. This should be
          specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.langs = langs


def read_examples_from_file(file_path, lang="en", lang2id=None):
    if not os.path.exists(file_path):
        logger.info("[Warming] file {} not exists".format(file_path))
        return []
    guid_index = 1
    examples = []
    subword_len_counter = 0
    if lang2id:
        lang_id = lang2id.get(lang, lang2id['en'])
    else:
        lang_id = 0
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        langs = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if word:
                    examples.append(InputExample(guid="{}-{}".format(lang, guid_index),
                                                 words=words,
                                                 labels=labels,
                                                 langs=langs))
                    guid_index += 1
                    words = []
                    labels = []
                    langs = []
                    subword_len_counter = 0
                else:
                    print(f'guid_index', guid_index, words, langs, labels, subword_len_counter)
            else:
                splits = line.split("\t")
                word = splits[0]

                words.append(splits[0])
                langs.append(lang_id)
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid="%s-%d".format(lang, guid_index),
                                         words=words,
                                         labels=labels,
                                         langs=langs))

    return examples


if __name__ == "__main__":
    args = parse_args()
    source_examples = read_examples_from_file(args.source_dataset)
    files = list(filter(lambda file: "idx" not in file, os.listdir(args.target_dataset)))
    files.sort()
    target_examples = []
    for file in files:
        logger.info(f"Reading {file}")
        target_examples.append(read_examples_from_file(f"{args.target_dataset}/{file}"))
    target_examples = list(itertools.chain(*target_examples))
    if args.sampling_method != "None" and len(target_examples) > len(source_examples):
        if args.sampling_method == "up-sampling":
            sampling_examples = np.random.choice(source_examples, len(target_examples) - len(source_examples),
                                                 replace=True)
            source_examples.extend(sampling_examples)
            assert len(source_examples) == len(target_examples)
            source_examples = source_examples * args.sampling_ratio
        elif args.sampling_method == "down-sampling":
            target_examples = list(
                np.random.choice(target_examples, int(len(source_examples) * args.sampling_ratio), replace=False))
            # target_examples = list(np.random.choice(target_examples, len(source_examples), replace=False))
            # assert len(source_examples) == len(target_examples)
            # source_examples = source_examples * args.sampling_ratio

    total_examples = source_examples + target_examples
    with open(args.output, "w", encoding="utf-8") as w:
        with open(args.idx, "w", encoding="utf-8") as idx_w:
            for i in range(len(total_examples)):
                for j in range(len(total_examples[i].words)):
                    w.write(f"{total_examples[i].words[j]}\t{total_examples[i].labels[j]}\n")
                    idx_w.write(f"{i}\n")
                w.write("\n")
                idx_w.write("\n")
    logger.info(f"Successfully Saving {len(total_examples)} examples into {args.output} and {args.idx}")
