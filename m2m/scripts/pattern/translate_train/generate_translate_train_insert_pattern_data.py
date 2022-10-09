import os
import argparse
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE
from argparse import Namespace
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(  # must set the level
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
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
                self.words_idxs[-1].append(len(self.tokenized_words))

    def is_beginning_of_entity(self, i):
        return self.labels[i].startswith("B-")

    def is_entity(self, i):
        return not self.labels[i].startswith("O")

    def insert_with_slot(self, spm, lang):
        if lang in CHARACTER_SPLIT_LANGS:
            self.raw_sentence = "".join(self.words)
            self.entity_number = 0
            self.tokenized_masked_words = self.words.copy()
            for i in range(len(self.words)):
                if self.is_beginning_of_entity(i) or (self.is_entity(i) and i == 0):
                    if i > 0 and self.is_entity(i - 1):
                        self.tokenized_masked_words[
                            i - 1] = f"{self.tokenized_masked_words[i - 1]}\u2582{self.entity_number}"
                        self.entity_number += 1
                    self.tokenized_masked_words[i] = f"\u2582{self.entity_number}{self.tokenized_masked_words[i]}"
                    self.entities.append([self.words[i]])
                    if i == len(self.words) - 1:
                        self.tokenized_masked_words[i] += f"\u2582{self.entity_number}"
                elif self.is_entity(i):
                    self.entities[-1].append(self.words[i])
                    if i == len(self.words) - 1:
                        self.tokenized_masked_words[-1] = f"{self.tokenized_masked_words[-1]}\u2582{self.entity_number}"
                        self.entity_number += 1
                else:
                    if i > 0 and self.is_entity(i - 1):
                        self.tokenized_masked_words[i] = f"\u2582{self.entity_number}{self.tokenized_masked_words[i]}"
                        self.entity_number += 1
            self.tokenized_masked_sentence = "".join(self.tokenized_masked_words).replace(" ", "")
            self.tokenized_masked_sentence = spm.encode(self.tokenized_masked_sentence)
            self.tokenized_masked_sentence = self.tokenized_masked_sentence
            for i in range(14):
                self.tokenized_masked_sentence = self.tokenized_masked_sentence.replace(f"\u2582 {i}", f" __SLOT{i}__ ")
            self.tokenized_masked_sentence = self.tokenized_masked_sentence.replace("  ", " ").strip()
        else:
            self.tokenized_sentence = spm.encode(self.raw_sentence)
            self.tokenized_words = self.tokenized_sentence.split()
            self.tokenized_masked_words = self.tokenized_sentence.split()
            self.get_words_segment(spm, lang)
            self.entity_number = 0
            for i in range(len(self.words)):
                if self.is_beginning_of_entity(i) or (self.is_entity(i) and i == 0):
                    if i > 0 and self.is_entity(i - 1):
                        self.tokenized_masked_words[self.words_idxs[i - 1][
                                                        1] - 1] = f"{self.tokenized_masked_words[self.words_idxs[i - 1][1] - 1]} __SLOT{self.entity_number}__"
                        self.entity_number += 1
                    self.tokenized_masked_words[self.words_idxs[i][
                        0]] = f"__SLOT{self.entity_number}__ {self.tokenized_masked_words[self.words_idxs[i][0]]}"
                    self.entities.append(self.tokenized_words[self.words_idxs[i][0]: self.words_idxs[i][1]])
                    if i == len(self.words) - 1:
                        self.tokenized_masked_words[self.words_idxs[i][1] - 1] += f" __SLOT{self.entity_number}__ "
                elif self.is_entity(i):
                    self.entities[-1].extend(self.tokenized_words[self.words_idxs[i][0]: self.words_idxs[i][1]])
                    if i == len(self.words) - 1:
                        self.tokenized_masked_words[self.words_idxs[i][
                                                        1] - 1] = f"{self.tokenized_masked_words[self.words_idxs[i][1] - 1]} __SLOT{self.entity_number}__ "
                else:
                    if i > 0 and self.is_entity(i - 1):
                        self.tokenized_masked_words[self.words_idxs[i - 1][
                                                        1] - 1] = f"{self.tokenized_masked_words[self.words_idxs[i - 1][1] - 1]} __SLOT{self.entity_number}__"
                        self.entity_number += 1
            self.tokenized_masked_sentence = " ".join(self.tokenized_masked_words).strip()
        return self.tokenized_masked_sentence, self.entities


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-input', type=str,
                        default=r'/path/to/NER/xtreme_v1/translate-train/NER/en/test.xlmr.tsv', help='input stream')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/path/to/NER/xtreme_v1/translate-train/NER/en/en.txt0000', help='input stream')
    parser.add_argument('--entity', '-entity', type=str,
                        default=r'', help='input stream')
    parser.add_argument('--raw-sentences', '-raw-sentences', type=str,
                        default=r'', help='input stream')
    parser.add_argument('--lang', '-lang', type=str,
                        default=r'en', help='input stream')
    parser.add_argument('--sentencepiece-model', '-sentencepiece-model', type=str,
                        default=r'/path/to/NER/PretrainedModels/flores/flores101_mm100_615M/se1ntencepiece.bpe.model',
                        help='input stream')
    args = parser.parse_args()
    return args


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


if __name__ == "__main__":
    args = parse_args()
    if args.output != "" and not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    if args.raw_sentences != "" and not os.path.exists(os.path.dirname(args.raw_sentences)):
        os.makedirs(os.path.dirname(args.raw_sentences))
    if args.entity != "" and not os.path.exists(os.path.dirname(args.entity)):
        os.makedirs(os.path.dirname(args.entity))
    logger.info("Loading data from {}".format(args.input))
    examples = read_examples_from_file(args.input, "en")
    # Change Labels
    change_labels(examples)
    ##
    spm = SentencepieceBPE(Namespace(sentencepiece_model=args.sentencepiece_model))
    sources = []
    slot_sentences = []
    raw_sentences = []
    entities = []
    max_entities = 0
    entity_limit = 100000  # Do not limit
    exceeding_entity_count = 0
    for index, example in enumerate(examples):
        slot_sentence, entity = example.insert_with_slot(spm, args.lang)
        max_entities = len(entity) if len(entity) > max_entities else max_entities
        if len(entity) > entity_limit:
            exceeding_entity_count += 1
            continue
        slot_sentences.append(slot_sentence)
        raw_sentences.append(example.raw_sentence)
        entities.extend(entity)
        sources.append(example.tokenized_sentence)
        assert len(entity) * 2 == slot_sentence.count(
            "SLOT"), f"Index: {index} | Please ensure the correct number of the slot symbol!"
    logger.info(
        f"Max Entities: {max_entities} | Exceeding Entity Count {exceeding_entity_count} (Enitity Num >= {entity_limit})")
    logger.info(f"Examples: {len(slot_sentences)} | Entities: {len(entities)}")
    assert len(slot_sentences) == len(examples)
    with open(args.output, "w", encoding="utf-8") as w_sent:
        for slot_sentence in slot_sentences:
            w_sent.write(f"{slot_sentence}\n")
    logger.info(f"Successfully saving to {args.output}")
    if args.entity != "":
        with open(args.entity, "w", encoding="utf-8") as w_entity:
            for entity in entities:
                entity = " ".join(entity)
                w_entity.write(f"{entity}\n")
        logger.info(f"Successfully saving to {args.entity}")
    if args.raw_sentences != "":
        with open(args.raw_sentences, "w", encoding="utf-8") as w_raw_sentences:
            for raw_sentence in raw_sentences:
                w_raw_sentences.write(f"{raw_sentence}\n")
        logger.info(f"Successfully saving to {args.raw_sentences}")
