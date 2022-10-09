import os
import argparse
import linecache
import transformers
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE


def mapping(languages: str) -> dict:
    return dict(
        tuple(pair.split(":"))
        for pair in languages.strip().replace("\n", "").split(",")
    )


LANGS = "af ar bg bn de el es et fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo zh".split()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-input-dir', type=str,
                        default=r'/path/to/XTREME-Pattern/data_from_shuming/shaohanh/panx_alignment_0909/alignment/',
                        help='input stream')
    parser.add_argument('--output-dir', '-output-dir', type=str,
                        default=r'/path/to/XTREME-Pattern/data_from_shuming/shaohanh/panx_alignment_0909/bert-cased-entity/',
                        help='input stream')
    parser.add_argument('--alignment', '-alignment', type=str,
                        default=r'/path/to/XTREME-Pattern/data_from_shuming/shaohanh/panx_alignment_0909/bert-cased/',
                        help='input stream')
    parser.add_argument('--sentencepiece-model', '-sentencepiece-model', type=str,
                        default=r'/path/to/SharedTask/thunder/PretrainedModel/deltalm/large-prenorm/spm.model',
                        help='input stream')
    args = parser.parse_args()
    return args


def find_sublist(x, sub_x, tokenizer):
    for start_ids in range(0, len(x) - len(sub_x) + 1):
        if x[start_ids:start_ids + len(sub_x)] == sub_x:
            end_ids = start_ids + len(en_entity)
            return start_ids, end_ids
    # Can not find the pattern under the spm
    # detok_x = tokenizer.detokenize(" ".join(x))
    # detok_sub_x = tokenizer.detokenize(" ".join(sub_x))
    # if detok_sub_x in detok_x:
    #     raw_start_ids = detok_x.index(detok_sub_x)
    #     raw_end_ids = raw_start_ids + len(detok_sub_x)
    #     start_ids = len(tokenizer.tokenize(detok_x[:raw_start_ids]).split())
    #     end_ids = len(tokenizer.tokenize(detok_x[:raw_end_ids]).split())
    #     return start_ids, end_ids
    # else:
    return -1, -1


if __name__ == "__main__":
    args = parse_args()
    # tokenizer = SentencepieceBPE(args)
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')
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
                        x = tokenizer.tokenize(detok_x.strip())
                        en = tokenizer.tokenize(detok_en.strip())
                        aligns = {int(ali.split('-')[0]): int(ali.split('-')[1]) for ali in align_line.strip().split()}
                        x_entities = []
                        for en_entity in en_entities:
                            en_entity = tokenizer.tokenize(en_entity)
                            start_ids, end_ids = find_sublist(en, en_entity, tokenizer)
                            if start_ids == -1 and end_ids == -1:
                                print("Skipping {}: {} | {}".format(i, x, x_entity))
                                x_entity = "None"
                                skip_count += 1
                                continue
                            else:
                                x_index = [aligns[index] for index in
                                           filter(lambda ids: ids in aligns, range(start_ids, end_ids))]
                                if len(x_index) > 0:
                                    x_index = list(range(min(x_index), max(x_index) + 1))
                                    x_entity = " ".join([x[index] for index in x_index])
                                    x_entity = x_entity.replace(" ##", "")
                                    if x_entity not in detok_x:
                                        x_entity = x_entity.split()
                                        if x_entity[0] in detok_x and x_entity[-1] in detok_x:
                                            x_start_ids = detok_x.index(x_entity[0])
                                            x_end_ids = detok_x.index(x_entity[-1]) + len(x_entity[-1])
                                            x_entity = detok_x[x_start_ids: x_end_ids]
                                        else:
                                            print("Skipping {}: {} | {}".format(i, x, x_entity))
                                            continue
                                else:
                                    print("{}: Can not find the alignment in English sentence: {} ||| {}".format(i, en,
                                                                                                                 en_entity))
                                    x_entity = "None"
                                    skip_count += 1
                                    continue
                            x_entities.append(x_entity)
                        w.write("{} ||| {} ||| {} ||| {}\n".format(" ".join(detok_en), " ".join(detok_x),
                                                                   " ".join(en_entities), " ".join(x_entities)))
                    print("{} -> {} | Total: {} | remove: {}".format(input_file, output_file, len(input_lines),
                                                                     skip_count))
