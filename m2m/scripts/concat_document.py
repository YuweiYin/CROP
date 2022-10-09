import tqdm
import argparse
import os
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--document-path', '-document-path', type=str, default=r'', help='source file')
    parser.add_argument('--output-path', '-output-path', type=str, default=r'', help='source file')
    parser.add_argument('--N', '-N', type=int, default=1000, help='source file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    langs = os.listdir(args.document_path)
    lang_list = "fr,de,fi,cs,et,tr,lv,ro,hi,gu,en".split(",")
    N = args.N
    for lang in langs:
        if lang not in lang_list:
            continue
        print("Start preprocessing language: {}".format(lang))
        document_count = 0
        line_count = 0
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        output_file = os.path.join(args.output_path, "train.{}".format(lang))
        if os.path.exists(output_file):
            print("Deleting existing concat file: {}".format(output_file))
            os.remove(output_file)
        input_files = os.listdir(os.path.join(args.document_path, lang))
        with open(output_file, "w", encoding="utf-8") as output_w:
            print("Extracting from {} files".format(min(len(input_files), N)))
            select_indices = np.random.choice(len(input_files), min(len(input_files), N), replace=False)
            select_indices.sort()
            for index in select_indices:
                input_file = input_files[index]
                input_file = os.path.join(args.document_path, lang, input_file)
                with open(input_file, "r", encoding="utf-8") as input_r:
                    lines = input_r.readlines()
                    document = []
                    for line_id, line in enumerate(lines):
                        line = line.strip()
                        if line == "" or line_id == len(lines) - 1:
                            if len(document) > 0:
                                output_w.write(" ".join(document) + "\n")
                                document.clear()
                                document_count += 1
                            continue
                        document.append(line)
                        line_count += 1
                    print("Complete processing data file: {}".format(input_file))
        print("total lines of language {} : {}".format(lang, line_count))
        print("total documents of language {} : {}".format(lang, document_count))
        print("writing to : {}".format(output_file))
