import argparse
import xlwt
import xlrd
import os


def mapping(languages: str) -> dict:
    return dict(
        tuple(pair.split(":")) for pair in languages.strip().replace("\n", "").split(",")
    )


LANGS = "af ar bg bn de el es et fa fi fr he hi hu id it ja jv ka kk ko ml mr " \
        "ms my nl no pt ru sw ta te th tl tr ur vi yo zh".split()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', '-log', type=str,
                        default=r'/path/to/NER/flores/evaluation/BLEU/', help='input stream')
    parser.add_argument('--result', '-result', type=str,
                        default=r'/path/to/NER/flores/evaluation/ExcelResults/', help='input stream')
    args = parser.parse_args()
    return args


def create_excel(results, name):
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet("LargeTrack", cell_overwrite_ok=True)
    worksheet.write(1, 0, label="DeltaLM-Postnorm (Large)")
    for i in range(len(LANGS)):
        worksheet.write(0, i + 1, label=LANGS[i])
        worksheet.write(i + 1, 0, label=LANGS[i])
    for i in range(len(LANGS)):
        for j in range(len(LANGS)):
            worksheet.write(i + 1, j + 1, label=results[i][j])
    save_path = '/path/to/SharedTask/ExcelResults/{}.xls'.format(name)
    workbook.save(save_path)
    print("Saving to {}".format(save_path))
    return workbook


def _lang_pair(src, tgt):
    return "{}->{}".format(src, tgt)


def read_excel(filename):
    m2m_x2x = {}
    workbook = xlrd.open_workbook(filename)
    worksheet = workbook.sheets()[0]
    ncols = worksheet.ncols
    nrows = worksheet.nrows
    M2M_LANGS = []
    for i in range(1, ncols):
        M2M_LANGS.append(worksheet[0][i].value)
    for i in range(1, nrows):
        for j in range(1, ncols):
            if i != j:
                m2m_x2x[_lang_pair(M2M_LANGS[i - 1], M2M_LANGS[j - 1])] = float(worksheet[i][j].value)
    return m2m_x2x


def calculate_avg_score(x2x, src=None, tgt=None, model_name="m2m"):
    results = []
    if src == "x" and tgt == "y":
        for key in x2x.keys():
            if "{}".format("en") not in key:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        print("{}: x->y: {:.2f}".format(model_name, avg))
    elif src is not None:
        for key in x2x.keys():
            if "{}->".format(src) in key:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        print("{}: {}->x: {:.2f}".format(model_name, src, avg))
    elif tgt is not None:
        for key in x2x.keys():
            if "->{}".format(tgt) in key:
                results.append(x2x[key])
        avg = sum(results) / len(LANGS)
        print("{}: x->{}: {:.2f}".format(model_name, tgt, avg))
    else:
        avg = sum(x2x.values()) / len(x2x)
        print("{}: all: {:.2f}".format(model_name, avg))
    return avg


if __name__ == "__main__":
    args = parse_args()
    all_m2m_x2x = read_excel("/path/to/xTune/m2m_615M.xls")
    m2m_x2x = {}
    for key in all_m2m_x2x.keys():
        src, tgt = key.split("->")
        if (src == "en" and tgt in (LANGS + ["en"])) or (src in (LANGS + ["en"]) and tgt == "en"):
            m2m_x2x[key] = all_m2m_x2x[key]
    calculate_avg_score(m2m_x2x, src="en")
    calculate_avg_score(m2m_x2x, tgt="en")
    calculate_avg_score(m2m_x2x)
    x2x = {}
    for i, lg in enumerate(LANGS):
        with open(os.path.join(args.log, "{}-{}.BLEU".format(lg, "en")), "r", encoding="utf-8") as r:
            result_lines = r.readlines()
            last_line = result_lines[-1]
            score = float(last_line.split()[2])
            x2x["{}->{}".format(lg, "en")] = score
        with open(os.path.join(args.log, "{}-{}.BLEU".format("en", lg)), "r", encoding="utf-8") as r:
            result_lines = r.readlines()
            last_line = result_lines[-1]
            score = float(last_line.split()[2])
            x2x["{}->{}".format("en", lg)] = score
    assert len(x2x) == len(m2m_x2x)
    x2e_results = calculate_avg_score(x2x, tgt="en", model_name="our")
    e2x_results = calculate_avg_score(x2x, src="en", model_name="our")
    avg_results = calculate_avg_score(x2x, model_name="our")
