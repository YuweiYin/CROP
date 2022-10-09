import argparse
import numpy as np
from multiprocessing import Pool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-input', type=str,
                        default=r'/path/to/XTREME-Pattern/filter-v1/data-bin/train.align.en-af', help='baseline')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/path/to/XTREME-Pattern/filter-v1/data-bin/train.align.en-af', help='baseline')
    parser.add_argument("--workers", "-workers", type=int, default=80)
    args = parser.parse_args()
    return args


def get_aligns(raw_line):
    align_dict = np.array([[int(ali.split('-')[0]), int(ali.split('-')[1])] for ali in raw_line.strip().split()],
                          dtype="uint8")
    # align_dict = [[ali.split('-')[0], ali.split('-')[1]] for ali in raw_line.strip().split()]
    return align_dict


if __name__ == "__main__":
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as r:
        lines = r.readlines()
        pool = Pool(args.workers)
        align_dataset = list(pool.imap(get_aligns, lines, args.workers))
    align_dataset = np.array(align_dataset, dtype=object)
    np.save(args.output, align_dataset)
    print("Successfully {} -> {}".format(args.input, "{}.npy".format(args.output)))
