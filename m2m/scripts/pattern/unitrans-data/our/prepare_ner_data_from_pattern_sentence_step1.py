import os
import argparse
import logging
import itertools
import linecache
import re

LANGS = "af,am,ar,ast,az,ba,be,bg,bn,br,bs,ca,ceb,cs,cy,da,de,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kk,km,kn,ko,lb,lg,ln,lo,lt,lv,mg,mk,ml,mn,mr,ms,my,ne,nl,no,ns,oc,or,pa,pl,ps,pt,ro,ru,sd,si,sk,sl,so,sq,sr,ss,su,sv,sw,ta,th,tl,tn,tr,uk,ur,uz,vi,wo,xh,yi,yo,zh,zu,eu,te".split(
    ",")
logger = logging.getLogger(__name__)
logging.basicConfig(  # must set the level
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
CHARACTER_SPLIT_LANGS = ["zh", "ja", "th"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--en-input', '-en-input', type=str,
                        default=r'/path/to/NER/unitrans-data/xlmr_augment_v1/translation/LABELED_EN/de.txt0000',
                        help='input stream')
    parser.add_argument('--input', '-input', type=str,
                        default=r'/path/to/NER/unitrans-data/xlmr_augment_v1/translation/LABELED_X/en0000.2de',
                        help='input stream')
    parser.add_argument('--beam-size', '-beam-size', type=int,
                        default=1, help='beam size')
    parser.add_argument('--ner', '-ner', type=str,
                        default=r'/path/to/NER/unitrans-data/xlmr_augment_v1/translation/NER/de/test.xlmr.tsv',
                        help='input stream')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/path/to/NER/unitrans-data/xlmr_augment_v1/translation/LABELED_X_NER/train.de.tsv',
                        help='input stream')
    parser.add_argument('--idx', '-idx', type=str,
                        default=r'/path/to/NER/unitrans-data/xlmr_augment_v1/translation/LABELED_X_NER/train.de.tsv.idx',
                        help='input stream')
    parser.add_argument('--lang', '-lang', type=str,
                        default="de", help='input stream')
    parser.add_argument('--max-length', '-max-length', type=int,
                        default=128, help='input stream')
    parser.add_argument('--sentencepiece-model', '-sentencepiece-model', type=str,
                        default=r'/path/to/NER/PretrainedModels/flores/flores101_mm100_615M/sentencepiece.bpe.model',
                        help='input stream')
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
    logger.info("Changing Begin I- -> B-")


def match_special_case(en_words, words, example, lang):
    words = list(filter(lambda w: re.match(r"__SLOT(\d+)__", w) is None, words))  # remove SLOT TAG
    if re.match(r"__SLOT0__", en_words[0]) is not None and re.match(r"__SLOT0__", en_words[-1]):
        if lang in CHARACTER_SPLIT_LANGS:
            words = [[c for c in word] for word in words]
            words = list(itertools.chain(*words))
        else:
            words = words
        labels = [example.labels[0][0].replace("I-", "B-")] + [example.labels[0][0].replace("B-", "I-")] * (
                    len(words) - 1)
        return words, labels
    return None, None


def spm_decode(line):
    return line.replace(" ", "").replace("\u2581", " ").strip()


def filter_empty_words(words):
    return list(filter(lambda w: w != "", words))


def multiply_examples(items, multiply):
    items = [[items[i]] * multiply for i in range(len(items))]
    items = list(itertools.chain(*items))
    return items


def get_labeled_sent(en_lines, x_lines, examples):
    ner_words = []
    ner_labels = []
    SLOT_NUM = 14
    # examples = multiply_examples(examples, args.beam_size)
    # en_lines = multiply_examples(en_lines, args.beam_size)
    assert len(x_lines) == len(examples) and len(en_lines) == len(examples)
    for idx, (en_sent, sent, example) in enumerate(zip(en_lines, x_lines, examples)):
        ner_words.append([])
        ner_labels.append([])
        sent = spm_decode(sent)
        for i in range(SLOT_NUM):
            sent = sent.replace(f"__SLOT{i}__", f" __SLOT{i}__ ")
        words = filter_empty_words(sent.split())
        # en_words = filter_empty_words(en_sent.split())
        start = -1
        # special_words, special_labels = match_special_case(en_words, words, example, lang=args.lang)
        # if special_words is not None and special_labels is not None:
        #     ner_words[-1].extend(special_words)
        #     ner_labels[-1].extend(special_labels)
        #     continue
        if args.lang in CHARACTER_SPLIT_LANGS:
            for word in words:
                if start > -1:
                    if re.match(r"__SLOT(\d+)__", word) is not None:
                        start = -1
                    else:
                        length += 1
                        characters = [c for c in word]
                        if length > 1:
                            character_labels = [example.labels[start][0].replace("B-", "I-")] * (len(characters))
                        else:
                            character_labels = [example.labels[start][0]] + [
                                example.labels[start][0].replace("B-", "I-")] * (len(characters) - 1)
                        ner_words[-1].extend(characters)
                        ner_labels[-1].extend(character_labels)
                elif start == -1 and re.match(r"__SLOT(\d+)__", word) is not None:
                    length = 0
                    start = int(re.findall(r"__SLOT(\d+)__", word)[0])
                else:
                    characters = [c for c in word]
                    character_labels = ["O"] * (len(characters))
                    ner_words[-1].extend(characters)
                    ner_labels[-1].extend(character_labels)
        else:
            for word in words:
                if start > -1:
                    if re.match(r"__SLOT(\d+)__", word) is not None:
                        start = -1
                    else:
                        length += 1
                        ner_words[-1].append(word)
                        if length > 1:
                            ner_labels[-1].append(example.labels[start][0].replace("B-", "I-"))
                        else:
                            ner_labels[-1].append(example.labels[start][0])
                elif start == -1 and re.match(r"__SLOT(\d+)__", word) is not None:
                    length = 0
                    start = int(re.findall(r"__SLOT(\d+)__", word)[0])
                else:
                    ner_words[-1].append(word)
                    ner_labels[-1].append("O")
    return ner_words, ner_labels


if __name__ == "__main__":
    args = parse_args()
    TAG_SYMBOL = "O"
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    logger.info("Loading data from {}".format(args.ner))
    examples = read_examples_from_file(args.ner)
    change_labels(examples)
    # Construct Labels
    for example in examples:
        labels = []
        for i in range(len(example.labels)):
            if example.is_beginning_of_entity(i):
                labels.append([example.labels[i]])
            elif example.is_entity(i):
                labels[-1].append(example.labels[i])
        example.labels = labels
    # read English pattern data
    en_lines = linecache.getlines(args.en_input)
    # read translated pattern data
    x_lines = linecache.getlines(args.input)
    assert len(en_lines) * args.beam_size == len(x_lines)
    assert len(examples) * args.beam_size == len(x_lines)
    beam_ner_words = []
    beam_ner_labels = []
    for b in range(args.beam_size):
        beam_x_lines = [x_lines[i] for i in range(b, len(x_lines), args.beam_size)]
        ner_words, ner_labels = get_labeled_sent(en_lines, beam_x_lines, examples)
        beam_ner_words.append(ner_words)
        beam_ner_labels.append(ner_labels)

    with open(args.output, "w", encoding="utf-8") as w:
        with open(args.idx, "w", encoding="utf-8") as idx_w:
            for i in range(len(beam_ner_words[0])):
                for k in range(args.beam_size):
                    for j in range(len(beam_ner_words[k][i])):
                        w.write(f"{beam_ner_words[k][i][j]}\t{beam_ner_labels[k][i][j]}\n")
                        idx_w.write(f"{i * args.beam_size + k}\n")
                    if i != len(beam_ner_words[0]) - 1 or k != args.beam_size - 1:
                        w.write("\n")
                        idx_w.write("\n")
    logger.info(f"Successfully Saving to {args.output} | Examples: {len(examples)} -> {len(beam_ner_words[0])}")
    logger.info(f"Successfully Saving to {args.idx} | Examples: {len(examples)} -> {len(beam_ner_words[0])}")
