import os
import argparse
import logging
from word2word import Word2word
import collections
import string
import itertools
import linecache
import re
import copy
import langid
import tqdm

LANGID_LANGS = "af, am, an, ar, as, az, be, bg, bn, br, bs, ca, cs, cy, da, de, dz, el, en, eo, es, et, eu, fa, fi, fo, fr, ga, gl, gu, he, hi, hr, ht, hu, hy, id, is, it, ja, jv, ka, kk, km, kn, ko, ku, ky, la, lb, lo, lt, lv, mg, mk, ml, mn, mr, ms, mt, nb, ne, nl, nn, no, oc, or, pa, pl, ps, pt, qu, ro, ru, rw, se, si, sk, sl, sq, sr, sv, sw, ta, te, th, tl, tr, ug, uk, ur, vi, vo, wa, xh, zh, zu".split(
    ", ")
logger = logging.getLogger(__name__)
logging.basicConfig(  # must set the level
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# create file handler which logs even debug messages
log_file = "/path/to/xTune/m2m/shells/m2m/MulDA/ner/generate_pseudo.log"
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
        self.langs = langs
        self.translated_sentence = ""
        self.translated_pattern = []
        self.translated_entities = []
        self.translated_labels = []
        self.translated_words = []
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
                        default=r'/path/to/NER/xtreme_MulDA/panx_processed_maxlen128/en/orig_data/train.xlmr',
                        help='input stream')
    parser.add_argument('--source-pattern', '-source-pattern', type=str,
                        default=r'/path/to/NER/xtreme_MulDA/MulDA/En/en.txt0000', help='input stream')
    parser.add_argument('--translated-pattern', '-translated-pattern', type=str,
                        default=r'/path/to/NER/xtreme_MulDA/MulDA/X/en0000.2yo', help='input stream')
    parser.add_argument('--translated-entities', '-translated-entities', type=str,
                        default=r'/path/to/NER/xtreme_MulDA/MulDA/X/en0001.2yo', help='input stream')
    parser.add_argument('--lang', '-lang', type=str,
                        default=r'yo', help='language')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/path/to/NER/xtreme_MulDA/MulDA/FINAL/train.yo.tsv', help='input stream')
    parser.add_argument('--idx', '-idx', type=str,
                        default=r'/path/to/NER/xtreme_MulDA/MulDA/FINAL/train.yo.tsv.idx', help='input stream')
    parser.add_argument('--max-length', '-max-length', type=int,
                        default=12, help='input stream')
    parser.add_argument('--min-length', '-min-length', type=int,
                        default=4, help='input stream')
    parser.add_argument('--max-sentences', '-max-sentences', type=int,
                        default=5000, help='input stream')
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
    for i in tqdm.tqdm(range(len(examples))):
        if len(examples[i].translated_words) == 0 or len(examples[i].translated_words) > args.max_length or len(
                examples[i].translated_words) < args.min_length:
            continue
        if examples[i].translated_sentence == "":
            continue
        words_count = collections.Counter(examples[i].translated_words)
        repeated_limit = 3
        if sorted(words_count.values())[-1] > repeated_limit:
            continue
        words_length = list(filter(lambda w: len(w) != 1 or "" in w, [w.split() for w in examples[i].translated_words]))
        if len(words_length) > 0:
            continue
        chars_length = list(filter(lambda w: len(w) > 25, examples[i].translated_words))
        if len(chars_length) > 0:
            continue
        EXCLUDE_ZH_SENT = ["公司简介", "项目简介", "简体中文", "繁体中文", "关于我们", "联系我们", "中文(简体)", "产品介绍", "企业简介", "介绍介绍", "企业简介",
                           "简介", "联系人", "新闻动态", "网站首页", "联系方式", "关于我们", "中国", "公司介绍", "关于", "立即预订", "关于我们", "立即预订",
                           "了解更多", "公司", "网站地图"]
        if args.lang == "zh" and len(list(filter(lambda w: w in examples[i].translated_sentence, EXCLUDE_ZH_SENT))) > 0:
            continue
        if args.lang in LANGID_LANGS and langid.classify(examples[i].translated_sentence)[0] != args.lang:
            continue
        chosen_examples.append(examples[i])
    return chosen_examples


def decode(x: str) -> str:
    return x.replace(" ", "").replace("\u2581", " ").strip()


def construct_pseudo_data(examples):
    for i in range(len(examples)):
        MAX_SLOT_NUM = 2
        if examples[i].translated_pattern.count("SLOT") > MAX_SLOT_NUM:
            examples[i].translated_words = []
            examples[i].translated_sentence = ""  # Save Empty
            continue
        if args.lang in CHARACTER_SPLIT_LANGS:
            examples[i].translated_pattern_words = examples[i].translated_pattern.split()
            examples[i].translated_words = copy.deepcopy(examples[i].translated_pattern_words)
            for idx, word in enumerate(examples[i].translated_words):
                if re.match(r"__SLOT(\d+)__", word):
                    examples[i].translated_words[idx] = [word]
                else:
                    examples[i].translated_words[idx] = list(word)
            examples[i].translated_entities = [list(entity.replace(" ", "")) for entity in
                                               examples[i].translated_entities]
            examples[i].translated_words = list(itertools.chain(*examples[i].translated_words))
        else:
            examples[i].translated_pattern_words = examples[i].translated_pattern.split()
            examples[i].translated_words = copy.deepcopy(examples[i].translated_pattern_words)
            examples[i].translated_entities = [list(filter(lambda w: w != "", entity.split())) for entity in
                                               examples[i].translated_entities]
        examples[i].translated_labels = ["O"] * len(examples[i].translated_words)
        assert len(examples[i].translated_words) == len(examples[i].translated_labels)
        for j in range(len(examples[i].translated_words)):
            if re.match(r"__SLOT(\d+)__", examples[i].translated_words[j]) is not None:
                slot_id = int(re.findall(r"__SLOT(\d+)__", examples[i].translated_words[j])[0])
                if slot_id < len(examples[i].translated_entities):
                    examples[i].translated_words[j] = examples[i].translated_entities[slot_id]
                    examples[i].translated_labels[j] = [examples[i].labels[slot_id][0]] + [
                        examples[i].labels[slot_id][0].replace("B-", "I-")] * (
                                                                   len(examples[i].translated_entities[slot_id]) - 1)
                else:
                    examples[i].translated_words = []
                    examples[i].translated_labels = []
                    examples[i].translated_sentence = ""  # Save Empty
                    break
        examples[i].translated_words = [w if isinstance(w, list) else [w] for w in examples[i].translated_words]
        examples[i].translated_labels = [l if isinstance(l, list) else [l] for l in examples[i].translated_labels]
        examples[i].translated_labels = list(itertools.chain(*examples[i].translated_labels))
        examples[i].translated_words = list(itertools.chain(*examples[i].translated_words))
        if args.lang in CHARACTER_SPLIT_LANGS:
            examples[i].translated_sentence = "".join(examples[i].translated_words)
        else:
            examples[i].translated_sentence = " ".join(examples[i].translated_words)
        assert len(examples[i].translated_words) == len(examples[i].translated_labels)
    return examples


def constrcut_entity_labels(examples):
    for example in examples:
        labels = []
        for i in range(len(example.labels)):
            if example.is_beginning_of_entity(i):
                labels.append([example.labels[i]])
            elif example.is_entity(i):
                labels[-1].append(example.labels[i])
        example.labels = labels
    return examples


def read_entities(translated_pattern_lines, translated_entities, examples):
    index = 0
    SLOT_NUM = 20
    examples = constrcut_entity_labels(examples)
    for i in range(len(translated_pattern_lines)):
        examples[i].translated_pattern = decode(translated_pattern_lines[i].strip())
        for slot_id in range(SLOT_NUM):
            examples[i].translated_pattern = examples[i].translated_pattern.replace(f"__SLOT{slot_id}__",
                                                                                    f" __SLOT{slot_id}__ ")
        examples[i].translated_pattern_words = list(
            filter(lambda w: w.strip() != "", examples[i].translated_pattern.split()))
        examples[i].translated_pattern = " ".join(examples[i].translated_pattern_words)
        examples[i].entity_number = source_pattern_lines[i].count("SLOT")
        examples[i].translated_entities = [decode(entity.strip()) for entity in
                                           translated_entities[index: index + examples[i].entity_number]]
        index += examples[i].entity_number
    assert index == len(translated_entities)
    return examples


if __name__ == "__main__":
    args = parse_args()
    # if args.lang in LANGID_LANGS:
    #     langid.set_languages(['en', args.lang])
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    logger.info("Loading data from {}".format(args.ner))
    examples = read_examples_from_file(args.ner, lang="en")
    change_labels(examples)
    source_pattern_lines = linecache.getlines(args.source_pattern)
    translated_pattern_lines = linecache.getlines(args.translated_pattern)
    translated_entities = linecache.getlines(args.translated_entities)
    assert len(examples) == len(translated_pattern_lines) and len(source_pattern_lines) == len(translated_pattern_lines)
    examples = read_entities(translated_pattern_lines, translated_entities, examples)
    translated_examples = construct_pseudo_data(copy.deepcopy(examples))
    translated_examples = ner_filter(translated_examples)
    translated_examples = translated_examples[:args.max_sentences]
    with open(args.output, "w", encoding="utf-8") as w:
        with open(args.idx, "w", encoding="utf-8") as idx_w:
            for i in range(len(translated_examples)):
                for j in range(len(translated_examples[i].translated_words)):
                    w.write(
                        f"{translated_examples[i].translated_words[j]}\t{translated_examples[i].translated_labels[j]}\n")
                    idx_w.write(f"{i}\n")
                w.write("\n")
                idx_w.write("\n")
    logger.info(f"Successfully Saving to {args.output} | Examples: {len(examples)} -> {len(translated_examples)}")
