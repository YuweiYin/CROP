import tqdm
import argparse
import os
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--document-path', '-document-path', type=str, default=r'', help='source file')
    parser.add_argument('--output-path', '-output-path', type=str, default=r'', help='source file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    langs = os.listdir(args.document_path)
    N = 5000000
    for lang in langs:
        document_count = 0
        line_count = 0
        input_file = os.path.join(args.document_path, lang, "{}.txt".format(lang))
        if not os.path.exists(os.path.join(args.document_path, "train-{}".format(N))):
            os.makedirs(os.path.join(args.document_path, "train-{}".format(N)))
        output_file = os.path.join(args.output_path, "train-{}".format(N), "train.{}".format(lang))
        if os.path.exists(output_file):
            print("Deleting existing concat file: {}".format(output_file))
            os.remove(output_file)
        with open(output_file, "w", encoding="utf-8") as output_w:
            with open(input_file, "r", encoding="utf-8") as input_r:
                print("Start reading file: {}".format(input_file))
                lines = input_r.readlines()
                print("Successfully reading file: {}".format(input_file))
                select_number = min(len(lines), N)
                select_indices = np.random.choice(len(lines), select_number)
                for select_index in select_indices:
                    if lines[select_index].strip() != "":
                        output_w.write(lines[select_index])
        print("total selected documents of language {} : {}".format(lang, select_number))
