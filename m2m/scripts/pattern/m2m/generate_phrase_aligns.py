import argparse
import numpy as np
import linecache
from multiprocessing import Pool


def phrase_extraction(srctext, trgtext, alignment, max_phrase_length=0):
    """
    Phrase extraction algorithm.
    """

    def extract(
            f_start,
            f_end,
            e_start,
            e_end,
            alignment,
            f_aligned,
            srctext,
            trgtext,
            srclen,
            trglen,
            max_phrase_length,
    ):
        """
        This function checks for alignment point consistency and extracts
        phrases using the chunk of consistent phrases.

        A phrase pair (e, f ) is consistent with an alignment A if and only if:

        (i) No English words in the phrase pair are aligned to words outside it.

               ∀e i ∈ e, (e i , f j ) ∈ A ⇒ f j ∈ f

        (ii) No Foreign words in the phrase pair are aligned to words outside it.

                ∀f j ∈ f , (e i , f j ) ∈ A ⇒ e i ∈ e

        (iii) The phrase pair contains at least one alignment point.

                ∃e i ∈ e  ̄ , f j ∈ f  ̄ s.t. (e i , f j ) ∈ A

        :type f_start: int
        :param f_start: Starting index of the possible foreign language phrases
        :type f_end: int
        :param f_end: End index of the possible foreign language phrases
        :type e_start: int
        :param e_start: Starting index of the possible source language phrases
        :type e_end: int
        :param e_end: End index of the possible source language phrases
        :type srctext: list
        :param srctext: The source language tokens, a list of string.
        :type trgtext: list
        :param trgtext: The target language tokens, a list of string.
        :type srclen: int
        :param srclen: The number of tokens in the source language tokens.
        :type trglen: int
        :param trglen: The number of tokens in the target language tokens.
        """

        if f_end < 0:  # 0-based indexing.
            return {}
        # Check if alignment points are consistent.
        for e, f in alignment:
            if (f_start <= f <= f_end) and (e < e_start or e > e_end):
                return {}

        # Add phrase pairs (incl. additional unaligned f)
        phrases = set()
        fs = f_start
        while True:
            fe = min(f_end, f_start + max_phrase_length - 1)
            while True:
                # add phrase pair ([e_start, e_end], [fs, fe]) to set E
                # Need to +1 in range  to include the end-point.
                src_phrase = " ".join(srctext[e_start: e_end + 1])
                trg_phrase = " ".join(trgtext[fs: fe + 1])
                # Include more data for later ordering.
                phrases.add(
                    ((e_start, e_end + 1), (fs, fe + 1), src_phrase, trg_phrase)
                )
                fe += 1
                if fe in f_aligned or fe >= trglen:
                    break
            fs -= 1
            if fs in f_aligned or fs < 0:
                break
        return phrases

    srctext = srctext.split()  # e
    trgtext = trgtext.split()  # f
    srclen = len(srctext)  # len(e)
    trglen = len(trgtext)  # len(f)
    # Keeps an index of which source/target words that are aligned.
    f_aligned = [j for _, j in alignment]
    max_phrase_length = max_phrase_length or max(srclen, trglen)

    # set of phrase pairs BP
    bp = set()

    for e_start in range(srclen):
        max_idx = min(srclen, e_start + max_phrase_length)
        for e_end in range(e_start, max_idx):
            # // find the minimally matching foreign phrase
            # (f start , f end ) = ( length(f), 0 )
            # f_start ∈ [0, len(f) - 1]; f_end ∈ [0, len(f) - 1]
            f_start, f_end = trglen - 1, -1  # 0-based indexing

            for e, f in alignment:
                if e_start <= e <= e_end:
                    f_start = min(f, f_start)
                    f_end = max(f, f_end)
            # add extract (f start , f end , e start , e end ) to set BP
            phrases = extract(
                f_start,
                f_end,
                e_start,
                e_end,
                alignment,
                f_aligned,
                srctext,
                trgtext,
                srclen,
                trglen,
                max_phrase_length,
            )
            if phrases:
                bp.update(phrases)
    return bp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', '-src', type=str,
                        default=r'/path/to/NER/m2m/20M/train-split10/train0/train.en-af.en', help='input stream')
    parser.add_argument('--tgt', '-tgt', type=str,
                        default=r'/path/to/NER/m2m/20M/train-split10/train0/train.en-af.af', help='input stream')
    parser.add_argument('--align', '-align', type=str,
                        default=r'/path/to/NER/m2m/20M/train-split10/train0/train.align.en-af.npy',
                        help='input stream')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/path/to/NER/m2m/20M/train-split10/train0/train.phrase.align.en-af.npy',
                        help='input stream')
    parser.add_argument('--max-phrase-length', '-max-phrase-length', type=int,
                        default=16, help='input stream')
    parser.add_argument('--workers', '-workers', type=int,
                        default=40, help='input stream')
    args = parser.parse_args()
    return args


def add_tag(line):
    return "<s> " + line + " </s>"


def get_phrase_aligns(line):
    phrases = phrase_extraction(line[0].strip(), line[1].strip(), line[2], max_phrase_length=args.max_phrase_length)
    return phrases


if __name__ == "__main__":
    args = parse_args()
    align_dataset = list(np.load(args.align, allow_pickle=True))
    src_lines = linecache.getlines(args.src)
    tgt_lines = linecache.getlines(args.tgt)
    print("Starting Converting {} -> {} | {} examples".format(args.align, args.output, len(align_dataset)))
    assert len(src_lines) == len(tgt_lines) and len(align_dataset) == len(src_lines)
    input_lines = list(zip(src_lines, tgt_lines, align_dataset))
    pool = Pool(args.workers)
    phrases = list(pool.imap(get_phrase_aligns, input_lines, args.workers))
    phrase_align_dataset = []
    print("Successfully Complete Converting {} -> {} | {} examples".format(args.align, args.output, len(align_dataset)))
    for phrase in phrases:
        phrase_align_dataset.append(np.array([(align[0], align[1]) for align in phrase], dtype="uint8"))
    phrase_align_dataset = np.array(phrase_align_dataset, dtype=object)
    np.save(args.output, phrase_align_dataset)
    print("Successfully Saving {} -> {} | {} examples".format(args.align, args.output, len(align_dataset)))
