import os
import argparse
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE
from argparse import Namespace
import logging

LANGS = "af,am,ar,ast,az,ba,be,bg,bn,br,bs,ca,ceb,cs,cy,da,de,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kk,km,kn,ko,lb,lg,ln,lo,lt,lv,mg,mk,ml,mn,mr,ms,my,ne,nl,no,ns,oc,or,pa,pl,ps,pt,ro,ru,sd,si,sk,sl,so,sq,sr,ss,su,sv,sw,ta,th,tl,tn,tr,uk,ur,uz,vi,wo,xh,yi,yo,zh,zu,eu,te".split(
    ",")
logger = logging.getLogger(__name__)
logging.basicConfig(  # must set the level
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


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
        self.words_idxs = None
        self.tokenized_sentence = None
        self.tokenized_words = None
        self.tokenized_masked_words = None
        self.tokenized_maksed_sentence = None
        self.entities = []
        self.entity_number = None

    def get_words_segment(self, spm):
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

    def insert_with_slot(self, spm):
        self.tokenized_sentence = spm.encode(self.raw_sentence)
        self.tokenized_words = self.tokenized_sentence.split()
        self.tokenized_masked_words = self.tokenized_sentence.split()
        self.get_words_segment(spm)
        self.entity_number = 0
        for i in range(len(self.words)):
            if self.is_beginning_of_entity(i) or (self.is_entity(i) and i == 0):
                self.tokenized_masked_words[self.words_idxs[i][
                    0]] = f"__SLOT{self.entity_number}__ {self.tokenized_masked_words[self.words_idxs[i][0]]}"
                self.entities.append(self.tokenized_words[self.words_idxs[i][0]: self.words_idxs[i][1]])
            elif self.is_entity(i):
                self.entities[-1].extend(self.tokenized_words[self.words_idxs[i][0]: self.words_idxs[i][1]])
            else:
                if i > 0 and self.is_entity(i - 1):
                    self.tokenized_masked_words[self.words_idxs[i - 1][
                                                    1] - 1] = f"{self.tokenized_masked_words[self.words_idxs[i - 1][1] - 1]} __SLOT{self.entity_number}__"
                    self.entity_number += 1
        self.tokenized_maksed_sentence = " ".join(self.tokenized_masked_words)
        return self.tokenized_maksed_sentence, self.entities


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
                    word = None
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-input', type=str,
                        default=r'/path/to/NER/unitrans-data/xlmr/processed/en/train.xlmr', help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logger.info("Loading data from {}".format(args.input))
    examples = read_examples_from_file(args.input, "en")
    with open(f"{args.input}.idx", "w", encoding="utf-8") as idx_w:
        for i in range(len(examples)):
            for j in range(len(examples[i].words)):
                idx_w.write(f"{i}\n")
            if i != len(examples) - 1:
                idx_w.write("\n")
    logger.info(f"Successfully saving to {args.input}.idx")
