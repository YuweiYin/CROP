import os
import argparse
import logging
from word2word import Word2word
import collections
import string
import itertools
import linecache
import re

logger = logging.getLogger(__name__)
logging.basicConfig(  # must set the level
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# create file handler which logs even debug messages
log_file = "/path/to/xTune/m2m/shells/m2m/UniTrans/ner/generate_pseudo.log"
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
logger.addHandler(fh)
CHARACTER_SPLIT_LANGS = ["zh", "ja", "th"]


class word2word_translation(object):
    def __init__(self, src, tgt):
        if tgt == "zh":
            tgt = "zh_cn"
        try:
            self.word_translation = Word2word(src, tgt)
        except:
            self.word_translation = None
            logger.info(f"Can not find the language {tgt} word translator")

    def word2word_translation(self, input):
        if self.word_translation is None:
            return []
        if input in string.punctuation:
            return input
        try:
            output = self.word_translation(input)
        except:
            output = []
        return output


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
        self.all_labels = labels
        self.langs = langs
        self.raw_sentence = " ".join(words)
        self.words_idxs = None
        self.tokenized_sentence = None
        self.tokenized_words = None
        self.tokenized_masked_words = None
        self.tokenized_masked_sentence = None
        self.entities = []
        self.entity_number = None

    def get_words_segment(self, spm, lang):
        if lang in CHARACTER_SPLIT_LANGS:
            self.words_idxs = [[i, i + 1] for i in range(len(self.words))]
        else:
            self.words_idxs = []
            for i in range(len(self.tokenized_words)):
                if spm.is_beginning_of_word(self.tokenized_words[i]):
                    if len(self.words_idxs) > 0:
                        self.words_idxs[-1].append(i)
                    self.words_idxs.append([i])
            if len(self.words_idxs[-1]) == 1:
                self.words_idxs[-1].append(self.words_idxs[-1][0] + 1)

    def is_beginning_of_entity(self, i):
        return self.labels[i].startswith("B-")

    def is_entity(self, i):
        return not self.labels[i].startswith("O")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ner', '-ner', type=str,
                        default=r'/path/to/NER/xtreme_UniTrans/UniTrans/NER/en/test.xlmr.tsv', help='input stream')
    parser.add_argument('--translated-ner', '-translated-ner', type=str,
                        default=r'/path/to/NER/xtreme_UniTrans/UniTrans/X/en0000.2zh', help='input stream')
    parser.add_argument('--lang', '-lang', type=str,
                        default=r'zh', help='language')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/path/to/NER/xtreme_UniTrans/UniTrans/FINAL/train.zh.tsv', help='input stream')
    parser.add_argument('--idx', '-idx', type=str,
                        default=r'/path/to/NER/xtreme_UniTrans/UniTrans/FINAL/train.zh.tsv.idx', help='input stream')
    parser.add_argument('--max-length', '-max-length', type=int,
                        default=16, help='input stream')
    parser.add_argument('--max-sentences', '-max-sentences', type=int,
                        default=50, help='input stream')
    args = parser.parse_args()
    return args


def read_examples_from_file(file_path, lang, lang2id=None):
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
    logger.info("lang_id={}, lang={}, lang2id={}".format(lang_id, lang, lang2id))
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


def change_labels(examples):
    for example in examples:
        for i in range(len(example.labels)):
            if i == 0 and example.is_entity(i):
                example.labels[i] = example.labels[i].replace("I-", "B-")
            elif i > 0 and not example.is_entity(i - 1) and example.is_entity(i):
                example.labels[i] = example.labels[i].replace("I-", "B-")
            elif i > 0 and example.is_entity(i - 1) and example.is_entity(i) and example.labels[i - 1][2:] != \
                    example.labels[i][2:]:  # B-PER I_ORG I-ORG
                example.labels[i] = example.labels[i].replace("I-", "B-")
    logger.info("Changing I- -> B-")


def ner_filter(examples):
    chosen_examples = []
    for i in range(len(examples)):
        if len(examples[i].words) > args.max_length:
            continue
        words_count = collections.Counter(examples[i].words)
        repeated_limit = 2
        if sorted(words_count.values())[-1] > repeated_limit:
            continue
        words_length = list(filter(lambda w: len(w) != 1 or "" in w, [w.split() for w in examples[i].words]))
        if len(words_length) > 0:
            continue
        chars_length = list(filter(lambda w: len(w) > 25, examples[i].words))
        if len(chars_length) > 0:
            continue
        EXCLUDE_ZH_SENT = ["公司简介", "项目简介", "简体中文", "繁体中文", "关于我们", "联系我们", "中文(简体)", "产品介绍", "企业简介", "介绍介绍", "企业简介",
                           "简介", "联系人", "新闻动态", "网站首页", "联系方式", "关于我们", "中国", "公司介绍", "关于", "立即预订", "关于我们", "立即预订",
                           "了解更多", "公司"]
        if args.lang == "zh" and len(list(filter(lambda w: w in EXCLUDE_ZH_SENT, examples[i].words))) > 0:
            continue
        chosen_examples.append(examples[i])
    return chosen_examples


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    logger.info("Loading data from {}".format(args.ner))
    examples = read_examples_from_file(args.ner, lang="en")
    change_labels(examples)
    word_translation = word2word_translation("en", args.lang)
    lines = linecache.getlines(args.translated_ner)
    translated_examples = []
    index = 0
    for i in range(len(examples)):
        words = []
        for j in range(len(examples[i].words)):
            words.append(lines[index].strip())
            index += 1
        translated_examples.append(InputExample(guid="{}-{}".format("en", i),
                                                words=words,
                                                labels=examples[i].labels,
                                                langs="en"))
    assert len(examples) == len(translated_examples)
    logger.info("Start Word Translation")
    for i in range(len(translated_examples)):
        for j in range(len(translated_examples[i].words)):
            translated_word = word_translation.word2word_translation(examples[i].words[j])
            if len(translated_word) > 0:
                translated_examples[i].words[j] = translated_word[0]
    logger.info("Complete Word Translation")
    translated_examples = ner_filter(translated_examples)
    translated_examples = translated_examples[: args.max_sentences]
    # EXCLUDE_LANGS=["th", "zh", "ja", "fa"]
    EXCLUDE_LANGS = []
    if args.lang in EXCLUDE_LANGS:
        translated_examples = []
    with open(args.output, "w", encoding="utf-8") as w:
        with open(args.idx, "w", encoding="utf-8") as idx_w:
            for i in range(len(translated_examples)):
                for j in range(len(translated_examples[i].words)):
                    w.write(f"{translated_examples[i].words[j]}\t{translated_examples[i].labels[j]}\n")
                    idx_w.write(f"{i}\n")
                w.write("\n")
                idx_w.write("\n")
    logger.info(f"Successfully Saving to {args.output} | Examples: {len(examples)} -> {len(translated_examples)}")
