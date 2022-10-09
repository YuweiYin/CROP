import os
import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(  # must set the level
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
LANGS = "af ar bg bn de el en es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo zh".split()
assert len(LANGS) == 40


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-input', type=str,
                        default=r'/path/to/NER/xtreme_MulDA/panx_processed_maxlen128/en/orig_data/train.xlmr',
                        help='input stream')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/path/to/NER/xtreme_MulDA/panx_processed_maxlen128/en/train.xlmr',
                        help='input stream')
    parser.add_argument('--max-sentences', '-max-sentences', type=int,
                        default=10000, help='input stream')
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


def generate_entites(examples):
    for example in examples:
        entities = []
        entity_labels = []
        for i in range(len(example.labels)):
            if example.is_beginning_of_entity(i):
                entity_labels.append([example.labels[i]])
                entities.append([example.words[i]])
            elif example.is_entity(i):
                entity_labels[-1].append(example.labels[i])
                entities[-1].append(example.words[i])
        example.entities = entities
        example.entity_labels = entity_labels


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
    # logger.info("Changing I- -> B-")


if __name__ == "__main__":
    args = parse_args()
    train_langs_examples = []
    valid_langs_examples = []
    test_langs_examples = []
    for lg in LANGS:
        train_examples = read_examples_from_file(f"/path/to/NER/xtreme_v1/train-{lg}.tsv")
        change_labels(train_examples)
        generate_entites(train_examples)
        train_langs_examples.append(train_examples)
        valid_examples = read_examples_from_file(f"/path/to/NER/xtreme_v1/panx_processed_maxlen128/{lg}/dev.xlmr")
        change_labels(valid_examples)
        generate_entites(valid_examples)
        valid_langs_examples.append(valid_examples)
        test_examples = read_examples_from_file(f"/path/to/NER/xtreme_v1/panx_processed_maxlen128/{lg}/test.xlmr")
        change_labels(test_examples)
        generate_entites(test_examples)
        test_langs_examples.append(test_examples)
        print(
            f"{lg} {len(train_examples) // 1000}K & {len(valid_examples) // 1000}K & {len(test_examples) // 1000}K | {len(train_examples)} & {len(valid_examples)} & {len(test_examples)}")
