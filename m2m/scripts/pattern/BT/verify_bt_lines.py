import os
import argparse
import linecache


def mapping(languages: str) -> dict:
    return dict(
        tuple(pair.split(":"))
        for pair in languages.strip().replace("\n", "").split(",")
    )


LANGS = "ar,bn,de,el,es,fi,hi,id,ko,ru,sw,te,th,tr,vi,zh".split(",")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-input-dir', type=str,
                        default=r'/path/to/XTREME-Pattern/data_from_shuming/split_10W/', help='input stream')
    parser.add_argument('--output-dir', '-output-dir', type=str,
                        default=r'/path/to/XTREME-Pattern/data_from_shuming/split_10W_BT/', help='input stream')
    parser.add_argument('--max-index', '-max-index', type=int,
                        default=600, help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    cmds = ""
    input_files = os.listdir(args.input_dir)
    BEAM = 4
    MODEL = "/path/to/SharedTask/thunder/large_track/data/Filter_v1/model/36L-12L/avg26_40.pt"
    BATCH_SIZE = 32
    count = 0
    MAX_NUM = args.max_index
    for file in input_files:
        print(u"Complete processing {} examples".format(file), end="\r")
        src = file.split('.')[0]
        index = file.split('.')[-1][-4:]
        if int(index) >= MAX_NUM:
            continue
        for tgt in LANGS:
            if (src == "en" or tgt == "en") and src != tgt:
                output = "{}{}.2{}".format(src, index, tgt)
                if not os.path.exists(os.path.join(args.output_dir, output)):
                    print("{} don't exist!".format(os.path.join(args.output_dir, output)))
                else:
                    N = 50000
                    output_lines = linecache.getlines(os.path.join(args.output_dir, output))
                    if len(output_lines) != N:
                        input_lines = linecache.getlines(os.path.join(args.input_dir, file))
                        if len(input_lines) == len(output_lines):
                            continue
                        print("{} < {} lines | input: {} lines | output {} lines".format(N,
                                                                                         os.path.join(args.output_dir,
                                                                                                      output),
                                                                                         len(input_lines),
                                                                                         len(output_lines)))
                    else:
                        continue
                cmd = "- name: {}-{}\n  sku: G1\n  sku_count: 1\n  command: \n    - bash ./shells/pattern/translate/translate.sh {} {} {} {} {} {} {}\n".format(
                    file, "{}2{}".format(src, tgt), src, tgt, BEAM, MODEL, file, output, BATCH_SIZE)
                count += 1
                cmds += cmd
    with open("/path/to/XTREME-Pattern/data_from_shuming/xtreme_translate.txt", "w", encoding="utf-8") as w:
        w.write(cmds)
    print(cmds)
