# encoding=utf-8
import os
import json
import linecache
import argparse
import logging
import itertools
from seqeval.metrics import precision_score, recall_score, f1_score

LANGS = "es nl de no".split()
logger = logging.getLogger(__name__)
logging.basicConfig(  # must set the level
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-input', type=str,
                        default=r'/path/to/NER/unitrans-data/xlmr_augment_v1/model/augment_v1/grid_search/',
                        help='input stream')
    args = parser.parse_args()
    return args


# /path/to/NER/xtreme_MulDA/model/xlmr-bsz32-maxlen128-lr1e-5-epoch10/evaluate_logs.txt
if __name__ == "__main__":
    args = parse_args()
    models = os.listdir(args.input)
    best_scores_list = []
    models.sort()
    max_file = 1000
    for model_id, model in enumerate(models):
        model_path = f"{args.input}/{model}/evaluate_logs.txt"
        input_lines = linecache.getlines(model_path)
        if model_id > max_file:
            break
        if len(input_lines) == 0:
            logger.info(f"model_id : {model_id} | Skipping {model}")
            continue
        logger.info(f"model_id : {model_id} | Loading log from {model}")
        epoch_scores = [json.loads(line.split("\t")[1]) for line in input_lines]
        dev_avg = -1
        test_avg = -1
        best_epoch = 0
        for epoch in range(len(epoch_scores)):
            if float(epoch_scores[epoch]['test_avg']['f1']) > test_avg:
                test_avg = epoch_scores[epoch]['test_avg']['f1']
                best_epoch = epoch
        all_best_scores = epoch_scores[best_epoch]
        all_best_scores["model_name"] = model
        best_scores_list.append(all_best_scores)
    best_test_avg = -1
    experiment_id = 0
    LANGS = [f"test_{lg}" for lg in LANGS]
    for i in range(len(best_scores_list)):
        if best_scores_list[i]['test_avg']['f1'] > best_test_avg:
            best_test_avg = best_scores_list[i]['test_avg']['f1']
            experiment_id = i

    logger.info(f"best model: {best_scores_list[experiment_id]['model_name']}")


    def get_results(digit):
        return round(digit * 100, 1)


    logger.info(
        f"{get_results(best_scores_list[experiment_id][LANGS[0]]['f1'])} & {get_results(best_scores_list[experiment_id][LANGS[1]]['f1'])} & {get_results(best_scores_list[experiment_id][LANGS[2]]['f1'])} & {get_results(best_scores_list[experiment_id][LANGS[3]]['f1'])} & {get_results(best_scores_list[experiment_id]['test_avg']['f1'])}")
