# encoding=utf-8
import os
import argparse
import logging
import itertools
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score
import time

LANGS = "af,am,ar,ast,az,ba,be,bg,bn,br,bs,ca,ceb,cs,cy,da,de,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kk,km,kn,ko,lb,lg,ln,lo,lt,lv,mg,mk,ml,mn,mr,ms,my,ne,nl,no,ns,oc,or,pa,pl,ps,pt,ro,ru,sd,si,sk,sl,so,sq,sr,ss,su,sv,sw,ta,th,tl,tn,tr,uk,ur,uz,vi,wo,xh,yi,yo,zh,zu,eu,te".split(
    ",")
logger = logging.getLogger(__name__)
logging.basicConfig(  # must set the level
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# create file handler which logs even debug messages
fh = logging.FileHandler(f"/path/to/xTune/m2m/shells/ner/prepare_ner_data/generate_pseudo.2021.11.16.log")
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
    parser.add_argument('--x-ner', '-x-ner', type=str,
                        default=r'/path/to/NER/xtreme_v1/translation/X_NER/ar/test.xlmr.tsv', help='input stream')
    parser.add_argument('--translated-ner', '-translated-ner', type=str,
                        default=r'/path/to/NER/xtreme_v1/translation/LABELED_X_NER/train.ar.tsv', help='input stream')
    parser.add_argument('--groundtruth-ner', '-groundtruth-ner', type=str,
                        default=r'/path/to/NER/xtreme_v1/train-ar.tsv', help='input stream')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/path/to/NER/xtreme_v1/translation/FINAL/train.ar.tsv', help='input stream')
    parser.add_argument('--idx', '-idx', type=str,
                        default=r'/path/to/NER/xtreme_v1/translation/FINAL/train.ar.idx', help='input stream')
    parser.add_argument('--log', '-log', type=str,
                        default=r'/path/to/xTune/data/log/train.ar.tsv.log', help='input stream')
    parser.add_argument('--lang', '-lang', type=str,
                        default=r'ar', help='input stream')
    parser.add_argument('--beam-size', '-beam-size', type=int,
                        default=1, help='beam size')
    parser.add_argument('--max-length', '-max-length', type=int,
                        default=32, help='input stream')
    parser.add_argument('--downsampling-lgs', '-downsampling-lgs', nargs="+",
                        default=["ar", "id", "it", "jv", "th"], help='input stream')
    parser.add_argument('--downsampling-size', '-downsampling-size', type=int,
                        default=250, help='input stream')
    parser.add_argument('--sentencepiece-model', '-sentencepiece-model', type=str,
                        default=r'/path/to/NER/PretrainedModels/m2m-100/spm.128k.model', help='input stream')
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


def get_list(labels):
    if isinstance(labels[0], list):
        return [[l] for l in list(itertools.chain(*labels))]
    else:
        return [[l] for l in labels]


def calculate_precision(groundtruth_labels, pred_labels):
    groundtruth_labels = get_list(groundtruth_labels)
    pred_labels = get_list(pred_labels)
    return precision_score(groundtruth_labels, pred_labels), recall_score(groundtruth_labels, pred_labels), f1_score(
        groundtruth_labels, pred_labels)


def find_entity(words, entity, strict=False):
    entity_idxs = []
    if strict:
        for i in range(len(words) - len(entity) + 1):
            for j in range(len(entity)):
                if words[i + j] != entity[j]:
                    break
                if j == len(entity) - 1:
                    entity_idxs = list(range(i, i + len(entity)))
                    return entity_idxs
        return entity_idxs
    else:
        for word in entity:
            if words.count(word) == 1:
                entity_idxs.append(groundtruth_example.words.index(word))
            # else:
            #     entity_idxs = []
            #     break
        if float(len(entity_idxs)) / len(entity) < 0.8 or len(entity) > 6:  # Find
            entity_idxs = []

        # for word in entity:
        #     if words.count(word) == 1:
        #         entity_idxs.append(groundtruth_example.words.index(word))
        #     else:
        #         entity_idxs = []
        #         break
        # if float(len(entity_idxs)) / len(words) < 0.5:
        #     entity_idxs = []
    return entity_idxs


def set_labels(labels, entity_idxs, entity_labels):
    min_idx = min(entity_idxs)
    max_idx = max(entity_idxs)
    for j in range(min_idx, max_idx + 1):
        if j == min_idx:
            labels[j] = entity_labels[0]
        else:
            labels[j] = entity_labels[0].replace("B-", "I-")
    return labels


def match_special_case(labels):
    if labels.count("O") > 0:
        return False
    if not labels[0].startswith("B-"):
        return False
    for i in range(1, len(labels)):
        if labels[i][2:] != labels[0][2:]:
            return False
    return True


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


if __name__ == "__main__":
    args = parse_args()
    TAG_SYMBOL = "O"
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    logger.info(f"Loading data from {args.translated_ner} + {args.groundtruth_ner}")
    translated_examples = read_examples_from_file(args.translated_ner, lang="en")
    groundtruth_examples = read_examples_from_file(args.groundtruth_ner, lang="en")
    if os.path.exists(args.x_ner):
        x_examples = read_examples_from_file(args.x_ner, lang="en")
    else:
        x_examples = None
    change_labels(translated_examples)
    change_labels(groundtruth_examples)
    if x_examples is not None:
        change_labels(x_examples)
    else:
        x_examples = [[]] * len(groundtruth_examples)
    #
    length_rm_count = 0
    count = 0
    # Saving Log
    log_ner_words = []
    log_ner_labels = []
    #
    ner_words = []
    ner_labels = []
    groundtruth_labels = []
    SLOT_NUM = 14
    MAX_LENGTH = 32
    MAX_ENTITY = 6
    assert len(translated_examples) == len(groundtruth_examples)
    rm_count = 0
    generate_entites(translated_examples)
    generate_entites(x_examples)
    # Considering Beam
    translated_examples = [translated_examples[i: i + args.beam_size] for i in
                           range(0, len(translated_examples), args.beam_size)]
    for idx, (translated_example, groundtruth_example, x_example) in enumerate(
            zip(translated_examples, groundtruth_examples, x_examples)):
        labels = ["O"] * len(groundtruth_example.words)
        remove_flag = False
        for beam, beam_translated_example in enumerate(translated_example):
            for i in range(len(beam_translated_example.entities)):
                entity = beam_translated_example.entities[i]
                # if match_special_case(translated_example.labels) and len(translated_example.entities) == 1: # __SLOT0__ xxxxx __SLOT__ format
                #     entity_idxs = list(range(len(groundtruth_example.words)))
                # else:
                entity_idxs = find_entity(groundtruth_example.words, entity, strict=False)
                if len(entity_idxs) == 0:
                    remove_flag = True
                    break
                else:
                    labels = set_labels(labels, entity_idxs, beam_translated_example.entity_labels[i])
            if not remove_flag:
                break
            elif remove_flag and beam < args.beam_size - 1:
                remove_flag = False
        ####################################
        if remove_flag and isinstance(x_example, InputExample) and len(groundtruth_example.words) < 32 and len(
                translated_example[0].entities) == len(x_example.entities):
            labels = x_example.labels
            remove_flag = False
        ####################################
        if (len(list(set(labels))) == 1 and list(set(labels))[0] == "O") or (len(labels) > MAX_LENGTH):
            remove_flag = True
        elif len(translated_example[0].entities) > MAX_ENTITY:
            remove_flag = True
        elif args.lang == "zh":
            if "".join(translated_example[0].words).replace(" ", "") == "关于我们":  # Wrong Bad Case
                remove_flag = True
        elif len(set(itertools.chain(*translated_example[0].entities))) < len(
                list(itertools.chain(*translated_example[0].entities))):  # remove overlapped
            remove_flag = True

        words = groundtruth_example.words
        labels = labels

        if remove_flag:
            log_labels = [f"{l}\t{gl}\tNO\t{idx}" for l, gl in zip(labels, groundtruth_example.labels)]
            log_ner_words.append(words)
            log_ner_labels.append(log_labels)
            rm_count += 1
            continue
        else:
            if labels[: MAX_LENGTH] == groundtruth_example.labels[:MAX_LENGTH]:
                log_labels = [f"{l}\t{gl}\tTrue\t{idx}" for l, gl in zip(labels, groundtruth_example.labels)]
                log_ner_words.append(words)
                log_ner_labels.append(log_labels)
            else:
                log_labels = [f"{l}\t{gl}\tFalse\t{idx}" for l, gl in zip(labels, groundtruth_example.labels)]
                log_ner_words.append(words)
                log_ner_labels.append(log_labels)

        # if labels[: MAX_LENGTH] != groundtruth_example.labels[:MAX_LENGTH]:
        #     continue
        ner_labels.append(labels)
        ner_words.append(words)
        groundtruth_labels.append(groundtruth_example.labels)
        count += 1

    precision, recall, f1 = calculate_precision(groundtruth_labels, ner_labels)
    logger.info(f"Prediction Precision {precision}")
    logger.info(f"Prediction Recall {recall}")
    logger.info(f"Prediction F1 {f1}")
    #########################
    if args.lang in args.downsampling_lgs:
        args.downsampling_size = min(len(ner_words), args.downsampling_size)
        chosen_idxs = np.random.choice(np.arange(0, len(ner_words)), args.downsampling_size, replace=False)
        ner_words = [ner_words[idx] for idx in chosen_idxs]
        ner_labels = [ner_labels[idx] for idx in chosen_idxs]
    with open(args.output, "w", encoding="utf-8") as w:
        with open(args.idx, "w", encoding="utf-8") as idx_w:
            for i in range(len(ner_words)):
                for j in range(len(ner_words[i])):
                    w.write(f"{ner_words[i][j]}\t{ner_labels[i][j]}\n")
                    idx_w.write(f"{i}\n")
                if i != len(ner_words) - 1:
                    w.write("\n")
                    idx_w.write("\n")

    with open(args.log, "w", encoding="utf-8") as log_w:
        for i in range(len(log_ner_words)):
            for j in range(len(log_ner_words[i])):
                log_w.write(f"{log_ner_words[i][j]}\t{log_ner_labels[i][j]}\n")
            if i != len(ner_words) - 1:
                log_w.write("\n")
    logger.info(
        f"Successfully Saving to {args.log} | Examples: {len(translated_examples)} -> {len(ner_words)} | Exceeding Max Length: {length_rm_count}")
    logger.info(
        f"Successfully Saving to {args.output} | Examples: {len(translated_examples)} -> {len(ner_words)} | Exceeding Max Length: {length_rm_count}")
    logger.info(
        f"Successfully Saving to {args.idx} | Examples: {len(translated_examples)} -> {len(ner_words)} | Exceeding Max Length: {length_rm_count}")
