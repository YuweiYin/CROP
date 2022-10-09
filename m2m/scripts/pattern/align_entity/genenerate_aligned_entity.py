import os
import argparse
import linecache
import itertools
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE


def mapping(languages: str) -> dict:
    return dict(
        tuple(pair.split(":"))
        for pair in languages.strip().replace("\n", "").split(",")
    )


LANGS = "zh af ar bg bn de el es et fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo".split()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-input-dir', type=str,
                        default=r'/path/to/XTREME-Pattern/data_from_shuming/shaohanh/panx_alignment_0909/alignment/',
                        help='input stream')
    parser.add_argument('--output-dir', '-output-dir', type=str,
                        default=r'/path/to/XTREME-Pattern/data_from_shuming/shaohanh/panx_alignment_0909/entity/',
                        help='input stream')
    parser.add_argument('--alignment', '-alignment', type=str,
                        default=r'/path/to/XTREME-Pattern/data_from_shuming/shaohanh/panx_alignment_0909/generate_align/',
                        help='input stream')
    parser.add_argument('--sentencepiece-model', '-sentencepiece-model', type=str,
                        default=r'/path/to/SharedTask/thunder/PretrainedModel/deltalm/large-prenorm/spm.model',
                        help='input stream')
    args = parser.parse_args()
    return args


def find_sublist(x, sub_x, spm):
    for start_ids in range(0, len(x) - len(sub_x) + 1):
        if x[i:i + len(sub_x)] == sub_x:
            end_ids = start_ids + len(en_entity)
            return start_ids, end_ids
    # Can not find the pattern under the spm
    detok_x = spm.decode(" ".join(x))
    detok_sub_x = spm.decode(" ".join(sub_x))
    if detok_sub_x in detok_x:
        raw_start_ids = detok_x.index(detok_sub_x)
        raw_end_ids = raw_start_ids + len(detok_sub_x)
        start_ids = len(spm.encode(detok_x[:raw_start_ids]).split())
        end_ids = len(spm.encode(detok_x[:raw_end_ids]).split())
        return start_ids, end_ids
    else:
        return -1, -1


if __name__ == "__main__":
    args = parse_args()
    spm = SentencepieceBPE(args)
    for lg in LANGS:
        input_file = os.path.join(args.input_dir, "{}-en.tsv".format(lg))
        align_file = os.path.join(args.alignment, "train.align.en-{}".format(lg))
        output_file = os.path.join(args.output_dir, "en-{}.entity".format(lg))
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        skip_count = 0
        with open(output_file, "w", encoding="utf-8") as w:
            with open(input_file, "r", encoding="utf-8") as input_r:
                with open(align_file, "r", encoding="utf-8") as align_r:
                    input_lines = input_r.readlines()
                    align_lines = align_r.readlines()
                    print("Successfully loading from {}...".format(input_file))
                    for i, (input_line, align_line) in enumerate(zip(input_lines, align_lines)):
                        detok_x = input_line.strip().split("\t")[0]
                        detok_en = input_line.strip().split("\t")[1]
                        en_entities = input_line.strip().split("\t")[2:]
                        x = spm.encode(detok_x.strip()).split()
                        en = spm.encode(detok_en.strip()).split()
                        aligns = {}
                        for ali in align_line.strip().split():
                            ali0, ali1 = ali.split('-')
                            ali0, ali1 = int(ali0), int(ali1)
                            if ali0 not in aligns:
                                aligns[ali0] = [ali1]
                            else:
                                aligns[ali0].append(ali1)
                        # aligns =  {int(ali.split('-')[0]): int(ali.split('-')[1]) for ali in align_line.strip().split()}
                        x_entities = []
                        for en_entity in en_entities:
                            en_entity = spm.encode(en_entity).split()
                            start_ids, end_ids = find_sublist(en, en_entity, spm)
                            if start_ids == -1 and end_ids == -1:
                                x_entity = "None"
                                skip_count += 1
                                print("Skipping {}: {} | {}".format(i, en, en_entity))
                                continue
                            else:
                                x_index = [aligns[index] for index in
                                           filter(lambda ids: ids in aligns, range(start_ids, end_ids))]
                                x_index = list(itertools.chain.from_iterable(x_index))
                                if len(x_index) > 0:
                                    x_index = list(range(min(x_index), max(x_index) + 1))
                                    x_entity = [x[index] for index in x_index]
                                    x_entity = spm.decode(" ".join(x_entity))
                                else:
                                    print("{}: Can not find the alignment in English sentence: {} ||| {}".format(i, en,
                                                                                                                 en_entity))
                                    x_entity = "None"
                                    skip_count += 1
                                    continue
                            x_entities.append(x_entity)
                        w.write("{} ||| {} ||| {} ||| {}\n".format(detok_en, detok_x, " ".join(en_entities),
                                                                   " ".join(x_entities)))
                    print("{} -> {} | Total: {} | remove: {}".format(input_file, output_file, len(input_lines),
                                                                     skip_count))
