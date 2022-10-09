import os
import argparse
import nltk
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
# create file handler which logs even debug messages
fh = logging.FileHandler("/path/to/xTune/m2m/shells/ner/prepare_ner_data/prepare_ner_data.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)

CHARACTER_SPLIT_LANGS = ["zh", "ja", "th"]


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
    parser.add_argument('--input', '-input', type=str,
                        default=r'/path/to/NER/xtreme_v0/translation/BT/zh0000.2en', help='input stream')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/path/to/NER/xtreme_v0/translation/NER/zh/test.xlmr', help='input stream')
    parser.add_argument('--idx', '-idx', type=str,
                        default=r'/path/to/NER/xtreme_v0/translation/NER/zh/test.xlmr.idx', help='input stream')
    parser.add_argument('--max-length', '-max-length', type=int,
                        default=1024, help='input stream')
    args = parser.parse_args()
    return args


def read_examples_from_file(filename):
    tokenized_sentences = []
    with open(filename, "r", encoding="utf-8") as r:
        lines = r.readlines()
        for i, line in enumerate(lines):
            line = line.replace("<unk>", "").strip()
            tokenized_words = list(filter(lambda word: word != "", line.split()))  # remove additional space symbols
            tokenized_words = " ".join(tokenized_words)
            tokenized_words = nltk.word_tokenize(tokenized_words)
            assert len(tokenized_words) > 0, f"len(tokenized_words) > 0, line: {i}"
            tokenized_sentences.append(tokenized_words)
    return tokenized_sentences


if __name__ == "__main__":
    args = parse_args()
    TAG_SYMBOL = "O"
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    logger.info("Loading data from {}".format(args.input))
    examples = read_examples_from_file(args.input)
    length_rm_count = 0
    count = 0
    with open(args.output, "w", encoding="utf-8") as w:
        with open(args.idx, "w", encoding="utf-8") as idx_w:
            for idx, words in enumerate(examples):
                if len(words) > args.max_length:
                    length_rm_count += 1
                    continue
                for word in words:
                    w.write(f"{word}\t{TAG_SYMBOL}\n")
                    idx_w.write(f"{idx}\n")
                if idx != len(examples) - 1:
                    w.write("\n")
                    idx_w.write("\n")
                count += 1
    assert len(examples) == count
    logger.info(
        f"Successfully Saving to {args.output} | Examples: {len(examples)} -> {count} | Exceeding Max Length: {length_rm_count}")
    logger.info(
        f"Successfully Saving to {args.idx} | Examples: {len(examples)} -> {count} | Exceeding Max Length: {length_rm_count}")
