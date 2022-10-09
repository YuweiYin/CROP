# encoding=utf-8
import os
import argparse
import random
import hashlib
import logging
from multiprocessing import Pool

logger = logging.getLogger(__name__)
logging.basicConfig(  # must set the level
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# create file handler which logs even debug messages
fh = logging.FileHandler("./filter.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', '-src', type=str,
                        default=r'/path/to/NER/flores/m2m_bpe/train.en-zh.en', help='input src')
    parser.add_argument('--tgt', '-tgt', type=str,
                        default=r'/path/to/NER/flores/m2m_bpe/train.en-zh.zh', help='input tgt')
    parser.add_argument('--new-src', '-new-src', type=str,
                        default=r'/path/to/NER/flores/m2m_bpe/20M/train/train.en-zh.en', help='output src')
    parser.add_argument('--new-tgt', '-new-tgt', type=str,
                        default=r'/path/to/NER/flores/m2m_bpe/20M/train/train.en-zh.zh', help='output tgt')
    parser.add_argument('--length-ratio', '-length-ratio', type=float,
                        default=1.5, help='output tgt')
    parser.add_argument('--max-length', '-max-length', type=int,
                        default=250, help='output tgt')
    parser.add_argument('--max-sentences', '-max-sentences', type=int,
                        default=-1, help='output tgt')
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    return args


def decode(x: str) -> str:
    return x.replace(" ", "").replace("\u2581", " ").strip()


def get_hashes_and_lines(raw_line):
    hash = hashlib.md5(raw_line.encode("utf-8")).hexdigest()
    return hash


def deduplicate_pairs(src_lines, tgt_lines):
    pool = Pool(args.workers)
    results = list(pool.imap(get_hashes_and_lines, src_lines, 100))
    rm_duplicate_count = 0
    seen = set()
    unique_src_lines = []
    unique_tgt_lines = []
    for i, hash in enumerate(results):
        if hash not in seen:
            seen.add(hash)
            unique_src_lines.append(src_lines[i])
            unique_tgt_lines.append(tgt_lines[i])
        else:
            rm_duplicate_count += 1
        if i % 5000000 == 0:
            logger.info(f"Processing {i} lines | removing duplicated pairs {rm_duplicate_count}")
    logger.info(f"Removing duplicate pairs: {len(src_lines)} -> {len(unique_src_lines)}")
    return unique_src_lines, unique_tgt_lines


if __name__ == "__main__":
    random.seed(1)
    args = parse_args()
    length_ratio = args.length_ratio
    max_length = args.max_length
    with open(args.src, "r", encoding="utf-8") as src_r:
        with open(args.tgt, "r", encoding="utf-8") as tgt_r:
            if not os.path.exists(os.path.dirname(args.new_src)):
                os.makedirs(os.path.dirname(args.new_src))
            src_lang = args.src.split('.')[-1]
            tgt_lang = args.tgt.split('.')[-1]
            with open(args.new_src, "w", encoding="utf-8") as src_w:
                with open(args.new_tgt, "w", encoding="utf-8") as tgt_w:
                    remaining_count = 0
                    count = 0
                    rm_count = 0
                    unk_rm_count = 0
                    blank_rm_count = 0
                    length_ratio_rm_count = 0
                    max_length_rm_count = 0
                    latin_rm_count = 0
                    special_rm_count = 0

                    src_lines = src_r.readlines()
                    tgt_lines = tgt_r.readlines()
                    logger.info("Complete Reading from {}...".format(args.src))
                    logger.info("Complete Reading from {}...".format(args.tgt))
                    assert len(src_lines) == len(tgt_lines)
                    logger.info("Start Deduplicating pairs...")
                    src_lines, tgt_lines = deduplicate_pairs(src_lines, tgt_lines)
                    logger.info("Complete Deduplicating pairs...")
                    logger.info("Start shuffling pairs...")
                    all_lines = list(zip(src_lines, tgt_lines))
                    random.shuffle(all_lines)
                    src_lines = [line[0] for line in all_lines]
                    tgt_lines = [line[1] for line in all_lines]
                    clean_src_lines = []
                    clean_tgt_lines = []
                    logger.info("Complete shuffling pairs...")
                    for i, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines)):
                        detok_src = src_line.strip()
                        detok_tgt = tgt_line.strip()
                        if count % 100000 == 0:
                            logger.info(
                                "Complete processing {} examples | removing {} examples (unk) | removing {} examples (length > {}) | removing {} examples (length ratio > {}) | removing {} examples (blank) | removing {} examples (latin)".format(
                                    count, unk_rm_count, max_length_rm_count, max_length, length_ratio_rm_count,
                                    length_ratio, blank_rm_count, latin_rm_count))
                        count += 1
                        if "< unk >" in detok_src or "< unk >" in detok_tgt:
                            unk_rm_count += 1
                            rm_count += 1
                            continue
                        if len(detok_tgt.split()) == 0 or len(detok_src.split()) == 0:
                            blank_rm_count += 1
                            rm_count += 1
                            continue
                        if len(detok_src.split()) > max_length or len(detok_tgt.split()) > max_length:
                            max_length_rm_count += 1
                            rm_count += 1
                            continue
                        if len(detok_src.split()) / len(detok_tgt.split()) > length_ratio or len(
                                detok_tgt.split()) / len(detok_src.split()) > length_ratio:
                            length_ratio_rm_count += 1
                            rm_count += 1
                            continue
                        EXCLUDE_ZH_SENT = ["公司简介", "简体中文", "繁体中文", "关于我们", "联系我们", "中文(简体)", "产品介绍", "企业简介", "介绍介绍",
                                           "企业简介", "简介", "联系人", "新闻动态", "网站首页", "联系方式", "关于我们", "中国", "公司介绍", "关于",
                                           "立即预订", "关于我们", "立即预订", "了解更多", "公司"]  # Bad Case
                        if decode(detok_src.replace(" ", "")) in EXCLUDE_ZH_SENT or decode(
                                detok_tgt.replace(" ", "")) in EXCLUDE_ZH_SENT:  # Bad Case for Chinese
                            special_rm_count += 1
                            rm_count += 1
                            continue

                        clean_src_lines.append(src_line)
                        clean_tgt_lines.append(tgt_line)
                        remaining_count += 1

                    if args.max_sentences > 0 and len(clean_src_lines) > args.max_sentences:
                        logger.info(
                            "Clipping {} sentences -> {} sentences".format(len(clean_src_lines), args.max_sentences))
                        clean_src_lines = clean_src_lines[:args.max_sentences]
                        clean_tgt_lines = clean_tgt_lines[:args.max_sentences]
                    for src_line, tgt_line in zip(clean_src_lines, clean_tgt_lines):
                        src_w.write("{}".format(src_line))
                        tgt_w.write("{}".format(tgt_line))
                    logger.info(
                        "Results: {} | removing {} examples (unk) | removing {} examples (length > {}) | removing {} examples (length ratio > {}) | removing {} examples (blank) | removing {} examples (latin)  | special_rm_count {} examples".format(
                            len(clean_src_lines), unk_rm_count, max_length_rm_count, max_length, length_ratio_rm_count,
                            length_ratio, blank_rm_count, latin_rm_count, special_rm_count))
