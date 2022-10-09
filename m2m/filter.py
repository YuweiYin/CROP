import tqdm
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-set', '-src-set', type=str, default=r'', help='source file')
    parser.add_argument('--tgt-set', '-tgt-set', type=str, default=r'', help='target file')
    parser.add_argument('--new-src-set', '-new-src-set', type=str,
                        default=r'',
                        help='new target file')
    parser.add_argument('--new-tgt-set', '-new-tgt-set', type=str,
                        default=r'',
                        help='new target file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args.src_set, "r", encoding="utf-8") as src_r:
        with open(args.tgt_set, "r", encoding="utf-8") as tgt_r:
            with open(args.new_src_set, "w", encoding="utf-8") as new_src_w:
                with open(args.new_tgt_set, "w", encoding="utf-8") as new_tgt_w:
                    print("reading source data file: {}".format(args.src_set))
                    src_lines = src_r.readlines()
                    print("reading target data file: {}".format(args.tgt_set))
                    tgt_lines = tgt_r.readlines()
                    count = 0
                    for line_id, (src_line, tgt_line) in tqdm.tqdm(enumerate(zip(src_lines, tgt_lines))):
                        src_line = src_line.strip()
                        tgt_line = tgt_line.strip()
                        if src_line == "" or tgt_line == "":
                            print("{} line: Skipping this empty sentence !".format(line_id + 1))
                            count += 1
                            continue
                        new_src_w.write(src_line + "\n")
                        new_tgt_w.write(tgt_line + "\n")
                print("Skipping all {} empty lines".format(count))
