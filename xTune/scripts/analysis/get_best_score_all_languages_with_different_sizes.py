import numpy as np
import logging
import linecache
import json

logger = logging.getLogger(__name__)
logging.basicConfig(  # must set the level
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def get_best_scores(min_size=1000, max_size=10000, step_size=1000):
    best_valid_scores = []
    best_test_scores = []
    for size in range(min_size, max_size, step_size):
        file_name = f"/path/to/NER/xtreme_analysis/data_size/{size}/model/xlmr-bsz32-maxlen128-lr1e-5-epoch15/evaluate_logs.txt"
        logger.info(f"Loading log from {file_name}")
        input_lines = linecache.getlines(file_name)
        epoch_scores = [json.loads(line.split("\t")[1]) for line in input_lines]
        dev_avg = -1
        best_epoch = 0
        for epoch in range(len(epoch_scores)):
            if float(epoch_scores[epoch]['dev_avg']['f1']) > dev_avg:
                dev_avg = epoch_scores[epoch]['dev_avg']['f1']
                best_epoch = epoch
        best_valid_scores.append(epoch_scores[best_epoch])
        best_test_scores.append(epoch_scores[best_epoch])
    return best_valid_scores, best_test_scores


_, best_test_scores_1000 = get_best_scores(min_size=10000, max_size=10001, step_size=10000)
_, best_test_scores_10000 = get_best_scores(min_size=50000, max_size=50001, step_size=50000)
_, best_test_scores_100000 = get_best_scores(min_size=100000, max_size=100001, step_size=100000)
_, best_test_scores_400000 = get_best_scores(min_size=400000, max_size=400001, step_size=400000)
print(best_test_scores_1000)
print(best_test_scores_10000)
print(best_test_scores_100000)
print(best_test_scores_400000)
