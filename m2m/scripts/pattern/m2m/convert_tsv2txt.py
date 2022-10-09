import os
import argparse
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE
import logging
import linecache
from argparse import Namespace

logger = logging.getLogger(__name__)
logging.basicConfig(  # must set the level
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
CHARACTER_SPLIT_LANGS = ["zh", "ja", "th"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-input', type=str,
                        default=r'/path/to/NER/xtreme_v1/translation/LABELED_X/en0000.2zh', help='input stream')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/path/to/NER/xtreme_v1/analysis/xlmr/translation/train.zh', help='input stream')
    parser.add_argument('--sentencepiece-model', '-sentencepiece-model', type=str,
                        default="/path/to/NER/PretrainedModels/xlm-roberta-base/sentencepiece.bpe.model",
                        help='input stream')
    parser.add_argument('--lang', '-lang', type=str,
                        default=r'zh', help='input stream')
    args = parser.parse_args()
    return args


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
            line = line.replace("\u200c", "")
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
                if word == "None" or word == "\u200c" or word == "":
                    continue
                words.append(word)
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
        self.raw_sentence = " ".join(words)
        self.tokenized_sentence = None
        self.tokenized_words = None


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    logger.info("Loading data from {}".format(args.input))
    tokenized_sentences = []
    spm = SentencepieceBPE(Namespace(sentencepiece_model=args.sentencepiece_model))
    if args.input.split('.')[-1] == "tsv":
        examples = read_examples_from_file(args.input, "en")
        for example in examples:
            if args.lang in CHARACTER_SPLIT_LANGS:
                tokenized_sentences.append(spm.encode(example.raw_sentence.replace(" ", "")))
            else:
                tokenized_sentences.append(spm.encode(example.raw_sentence))
    else:
        MAX_SLOT_NUM = 20
        examples = linecache.getlines(args.input)
        for example in examples:
            for i in range(MAX_SLOT_NUM):
                example = example.replace(f"__SLOT{i}__", " ")
            example = example.replace("  ", " ").strip()
            example = spm.decode(example)
            if args.lang in CHARACTER_SPLIT_LANGS:
                tokenized_sentences.append(spm.encode(example.replace(" ", "")))
            else:
                tokenized_sentences.append(spm.encode(example))
    with open(args.output, "w", encoding="utf-8") as w:
        for tokenized_sentence in tokenized_sentences:
            w.write(f"{tokenized_sentence}\n")
