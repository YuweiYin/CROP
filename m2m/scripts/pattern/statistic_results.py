# encoding=utf-8
import os
import json
import linecache
import argparse
import logging
import itertools
from seqeval.metrics import precision_score, recall_score, f1_score

LANGS = "af ar bg bn de el en es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo zh".split()
assert len(LANGS) == 40
logger = logging.getLogger(__name__)
logging.basicConfig(  # must set the level
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-input', type=str,
                        default=r'/path/to/NER/xtreme_v1/model/augment_v1/iter1/transformer-bsz32-maxlen128-lr1e-5-epoch10/evaluate_logs.txt',
                        help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Loading log from {args.input}")
    input_lines = linecache.getlines(args.input)
    epoch_scores = [json.loads(line.split("\t")[1]) for line in input_lines]
    dev_avg = -1
    test_avg = -1
    best_epoch = 0
    for epoch in range(len(epoch_scores)):
        if float(epoch_scores[epoch]['dev_avg']['f1']) > dev_avg:
            dev_avg = epoch_scores[epoch]['dev_avg']['f1']
            best_epoch = epoch
    all_best_scores = epoch_scores[best_epoch]
    print(f"{all_best_scores['dev_avg']['f1']} {all_best_scores['test_avg']['f1']}")
    best_scores = [round(float(all_best_scores[f"test_{lang}"]['f1']) * 100, 1) for lang in LANGS]
    best_scores = [str(s) for s in best_scores]
    print(" & ".join(best_scores[:20]))
    print(" & ".join(best_scores[20:]))
    print("New Table")
    a = " & ".join(best_scores[:21]) + " \\\\"
    b = " & ".join(best_scores[
                   21:]) + f" & {round(all_best_scores['dev_avg']['f1'] * 100, 1)}" + f" & {round(all_best_scores['test_avg']['f1'] * 100, 1)}" + " \\\\"
    print(a)
    print(b)
