import argparse
import os
import math

LANG_CODE = {'afr': 'af', 'amh': 'am', 'ara': 'ar', 'asm': 'as', 'ast': 'ast', 'azj': 'az', 'bel': 'be', 'ben': 'bn',
             'bos': 'bs', 'bul': 'bg', 'cat': 'ca', 'ceb': 'ceb', 'ces': 'cs', 'ckb': 'ku', 'cym': 'cy', 'dan': 'da',
             'deu': 'de', 'ell': 'el', 'eng': 'en', 'est': 'et', 'fas': 'fa', 'fin': 'fi', 'fra': 'fr', 'ful': 'ff',
             'gle': 'ga', 'glg': 'gl', 'guj': 'gu', 'hau': 'ha', 'heb': 'he', 'hin': 'hi', 'hrv': 'hr', 'hun': 'hu',
             'hye': 'hy', 'ibo': 'ig', 'ind': 'id', 'isl': 'is', 'ita': 'it', 'jav': 'jv', 'jpn': 'ja', 'kam': 'kam',
             'kan': 'kn', 'kat': 'ka', 'kaz': 'kk', 'kea': 'kea', 'khm': 'km', 'kir': 'ky', 'kor': 'ko', 'lao': 'lo',
             'lav': 'lv', 'lin': 'ln', 'lit': 'lt', 'ltz': 'lb', 'lug': 'lg', 'luo': 'luo', 'mal': 'ml', 'mar': 'mr',
             'mkd': 'mk', 'mlt': 'mt', 'mon': 'mn', 'mri': 'mi', 'msa': 'ms', 'mya': 'my', 'nld': 'nl', 'nob': 'no',
             'npi': 'ne', 'nso': 'ns', 'nya': 'ny', 'oci': 'oc', 'orm': 'om', 'ory': 'or', 'pan': 'pa', 'pol': 'pl',
             'por': 'pt', 'pus': 'ps', 'ron': 'ro', 'rus': 'ru', 'slk': 'sk', 'slv': 'sl', 'sna': 'sn', 'snd': 'sd',
             'som': 'so', 'spa': 'es', 'srp': 'sr', 'swe': 'sv', 'swh': 'sw', 'tam': 'ta', 'tel': 'te', 'tgk': 'tg',
             'tgl': 'tl', 'tha': 'th', 'tur': 'tr', 'ukr': 'uk', 'umb': 'umb', 'urd': 'ur', 'uzb': 'uz', 'vie': 'vi',
             'wol': 'wo', 'xho': 'xh', 'yor': 'yo', 'zho_simpl': 'zh', 'zho_trad': 'zt', 'zul': 'zu'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-input', type=str,
                        default=r'/path/to/NER/m2m/train/train.align.en-af.npy', help='input stream')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/path/to/NER/m2m/split4/', help='input stream')
    parser.add_argument('--split-num', '-split-num', type=int,
                        default=4, help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if os.path.isdir(args.input):
        files = os.listdir(args.input)
        for file in files:
            with open(os.path.join(args.input, file), "r", encoding="utf-8") as r:
                lines = r.readlines()
                chunk_num = math.ceil(len(lines) / args.split_num)
                print("Chunk Num: {}".format(chunk_num))
                for i in range(args.split_num):
                    output_dir = os.path.join(args.output, "train{}".format(i))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    with open(os.path.join(output_dir, file), "w", encoding="utf-8") as w:
                        start_index = i * chunk_num
                        end_index = (i + 1) * chunk_num
                        print("Successfully saving to {}: {} lines".format(os.path.join(output_dir, file),
                                                                           len(lines[start_index: end_index])))
                        # assert len(lines[start_index: end_index]) > 0, "Please ensure the number line chunk lines is greater than zero !"
                        chunk_num_limit = 200
                        if chunk_num <= chunk_num_limit:
                            print("#Chunk Num < {} ! Will Save all {} lines...".format(chunk_num_limit, len(lines)))
                            w.write("".join(lines))
                        else:
                            w.write("".join(lines[start_index: end_index]))
                del lines
    else:
        import numpy as np

        if ".npy" in args.input:
            file = os.path.basename(args.input)
            align_dataset = np.load(args.input, allow_pickle=True)
            chunk_num = math.ceil(len(align_dataset) / args.split_num)
            for i in range(args.split_num):
                output_dir = os.path.join(args.output, "train{}".format(i))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                start_index = i * chunk_num
                end_index = (i + 1) * chunk_num
                np.save(os.path.join(output_dir, file), align_dataset[start_index: end_index])
                print("Chunk Num: {}".format(len(align_dataset[start_index: end_index])))
        else:
            with open(os.path.join(args.input), "r", encoding="utf-8") as r:
                file = os.path.basename(args.input)
                lines = r.readlines()
                chunk_num = math.ceil(len(lines) / args.split_num)
                print("Chunk Num: {}".format(chunk_num))
                for i in range(args.split_num):
                    output_dir = os.path.join(args.output, "train{}".format(i))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    with open(os.path.join(output_dir, file), "w", encoding="utf-8") as w:
                        start_index = i * chunk_num
                        end_index = (i + 1) * chunk_num
                        w.write("".join(lines[start_index: end_index]))
                        print("Successfully saving to {}: {} lines".format(os.path.join(output_dir, file),
                                                                           len(lines[start_index: end_index])))
