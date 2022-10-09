import os
import argparse
from html.parser import HTMLParser
from bs4 import BeautifulSoup
import re

LANGS = "af,am,ar,as,ast,ay,az,ba,be,bg,bn,br,bs,ca,ceb,cjk,cs,cy,da,de,dyu,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kac,kam,kea,kg,kk,km,kmb,kmr,kn,ko,ku,ky,lb,lg,ln,lo,lt,luo,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,no,ns,ny,oc,om,or,pa,pl,ps,pt,qu,ro,ru,sd,shn,si,sk,sl,sn,so,sq,sr,ss,su,sv,sw,ta,te,tg,th,ti,tl,tn,tr,uk,umb,ur,uz,vi,wo,xh,yi,yo,zh,zu,eu,te".split(
    ",")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-input-dir', type=str,
                        default=r'/path/to/xTune/data/ccmatrix/tmx/', help='input stream')
    parser.add_argument('--output-dir', '-output-dir', type=str,
                        default=r'/path/to/xTune/data/ccmatrix/raw/', help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    files = os.listdir(args.input_dir)
    parser = HTMLParser()
    for i, file in enumerate(files):
        if os.path.getsize(os.path.join(args.input_dir, file)) > 0:
            src, tgt = file.split('.')[0].split('-')
            with open(os.path.join(args.input_dir, "{}-{}.tmx".format(src, tgt)), "r", encoding="utf-8") as r:
                src_lines = []
                tgt_lines = []
                lines = r.readlines()
                results = []
                context = []
                start = False
                for i, line in enumerate(lines):
                    if i % 1000000 == 0:
                        print(i)
                    if line.strip() == "<tu>":
                        start = True
                        continue
                    if line.strip() == "</tu>":
                        start = False
                        if len(context) == 2:
                            src_line = ""
                            tgt_line = ""
                            for context_line in context:
                                if '<tuv xml:lang="{}"><seg>'.format(src) in context_line:
                                    src_line = re.findall(r"<seg>(.+?)</seg>", context_line)
                                    assert len(src_line) == 1
                                elif '<tuv xml:lang="{}"><seg>'.format(tgt) in context_line:
                                    tgt_line = re.findall(r"<seg>(.+?)</seg>", context_line)
                                    assert len(tgt_line) == 1
                            if src_line != "" and tgt_line != "" and len(src_line[0].split()) / len(
                                    tgt_line[0].split()) > 2.0 or len(src_line[0].split()) / len(
                                    tgt_line[0].split()) < 2.0:
                                src_lines.append(src_line[0])
                                tgt_lines.append(tgt_line[0])
                                assert len(src_lines) == len(tgt_lines)
                        else:
                            print("Skipping wrong line {}".format(i))
                        context = []
                        continue
                    if start:
                        context.append(line)

                assert len(src_lines) == len(tgt_lines)
                if len(src_lines) > 0 and len(tgt_lines) > 0:
                    if not os.path.exists(os.path.join(args.output_dir, "{}{}".format(src, tgt))):
                        os.makedirs(os.path.join(args.output_dir, "{}{}".format(src, tgt)))
                    with open(os.path.join(args.output_dir, "{}{}".format(src, tgt),
                                           "train.{}-{}.{}".format(src, tgt, src)), "w", encoding="utf-8") as w_src:
                        with open(os.path.join(args.output_dir, "{}{}".format(src, tgt),
                                               "train.{}-{}.{}".format(src, tgt, tgt)), "w", encoding="utf-8") as w_tgt:
                            w_src.write("\n".join(src_lines))
                            w_tgt.write("\n".join(tgt_lines))
                            print("Successfully saving to {}".format(
                                os.path.join(args.output_dir, "{}{}".format(src, tgt),
                                             "train.{}-{}.{}".format(src, tgt, src))))
                            print("Successfully saving to {}".format(
                                os.path.join(args.output_dir, "{}{}".format(src, tgt),
                                             "train.{}-{}.{}".format(src, tgt, tgt))))
