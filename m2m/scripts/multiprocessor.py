from multiprocessing import Pool
import math
import argparse
import os
import re
import tqdm
import traceback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-set', '-src-set', type=str,
                        default=r'/path/to/RetriveNMT/data/Marcin/shumma/newstest2016-ende.en',
                        help='source file')
    parser.add_argument('--tgt-set', '-tgt-set', type=str,
                        default=r'/path/to/RetriveNMT/data/Marcin/shumma/newstest2016-ende.de',
                        help='dest file')
    parser.add_argument('--new-src-set', '-new-src-set', type=str,
                        default=r'/path/to/temp-multiprocess/test-2016.en',
                        help='source file')
    parser.add_argument('--new-tgt-set', '-new-tgt-set', type=str,
                        default=r'/path/to/temp-multiprocess/test-2016.de',
                        help='dest file')
    parser.add_argument('--temp-dir', '-temp-dir', type=str,
                        default=r'/path/to/temp-multiprocess/',
                        help='save temporary files')
    parser.add_argument('--max-tokens', '-max-tokens', type=int, default=-1, help='')
    parser.add_argument('--max-sentences', '-max-sentences', type=int, default=3, help='')
    parser.add_argument('--separator-token', '-separator-token', type=str, default=" [APPEND] ", help='')
    parser.add_argument('--workers', '-workers', type=int, default=4, help='')
    parser.add_argument('--debug', '-debug', action="store_true")
    args = parser.parse_args()
    return args


class MPLogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def error(msg, *args):
        return MP.get_logger().error(msg, *args)

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            self.error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result


class MultiProcessor(object):

    def __init__(self, temp_dir, max_tokens, max_sentences, separator_token, workers, debug):
        self.temp_dir = temp_dir
        self.workers = workers
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.separator_token = separator_token
        self.workers = workers
        if not os.path.exists(os.path.dirname(args.new_src_set)):
            os.makedirs(os.path.dirname(args.new_src_set))
        if not os.path.exists(os.path.dirname(args.new_tgt_set)):
            os.makedirs(os.path.dirname(args.new_tgt_set))
        if not os.path.exists(args.temp_dir):
            os.makedirs(args.temp_dir)

    def safe_readline(f):
        pos = f.tell()
        while True:
            try:
                return f.readline()
            except UnicodeDecodeError:
                pos -= 1
                f.seek(pos)  # search where this character begins

    def single_reader(
            self, filename, worker_id=0, workers=1
    ):
        with open(filename, "r", encoding="utf-8") as r:
            size = os.fstat(r.fileno()).st_size
            chunk_size = size // workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            r.seek(offset)
            if offset > 0:
                safe_readline(r)  # drop first incomplete line
            line = r.readline()
            lines = []
            total_count = 1
            while line:
                lines.append(line)
                if worker_id == 0:
                    print("Reading {} | lines".format(total_count), end='\r')
                if r.tell() > end:
                    break
                line = r.readline()
                total_count += 1
        return lines

    def multi_reader(self, filename):
        if not self.debug:
            p = Pool(self.workers)
            results = []
            for worker_id in range(self.workers):
                results.append(
                    p.apply_async(MPLogExceptions(self.single_reader), args=(filename, worker_id, self.workers)))
            p.close()
            p.join()
            total_lines = []
            for result in results:
                total_lines.extend(result.get())
            assert len(results) == self.workers, "Please check the number of results from subprocess !"
            print(" | All Reading Processes Finished !")
        else:
            self.workers = 1
            total_lines = self.single_reader(filename, 0, self.workers)
        return total_lines

    def Process_file(filename, new_filename, process_type="context"):
        p = Pool(args.workers)
        for worker_id in range(args.workers):
            p.apply_async(single_worker, args=(filename, worker_id, args.workers, process_type))
        p.close()
        p.join()
        print("{} | All Subprocess Context files Processes Finished !".format(new_filename))
        with open(new_filename, "w", encoding="utf-8") as w:
            for worker_id in range(args.workers):
                file_suffix = str.zfill(str(worker_id), len(str(args.workers)))
                subprocess_file_name = args.temp_dir + os.path.basename(filename) + file_suffix
                with open(subprocess_file_name, "r", encoding="utf-8") as r:
                    for line in r:
                        w.write(line)
                os.remove(subprocess_file_name)  # remove subprocess_file
        print("{} | All Subprocess Context files Merged !".format(new_filename))


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(os.path.dirname(args.new_src_set)):
        os.makedirs(os.path.dirname(args.new_src_set))
    if not os.path.exists(os.path.dirname(args.new_tgt_set)):
        os.makedirs(os.path.dirname(args.new_tgt_set))
    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)


    def safe_readline(f):
        pos = f.tell()
        while True:
            try:
                return f.readline()
            except UnicodeDecodeError:
                pos -= 1
                f.seek(pos)  # search where this character begins


    def proprcess_line(line, w, process_type="context"):
        if "<BEG>" in line and "<END>" in line:
            document = re.findall(r"<BEG> (.*) <SEP> <END>\n", line)[0].split(" <SEP> ")
        elif "<BEG>" in line and "<BRK>" in line:
            document = re.findall(r"<BEG> (.*) <SEP> <BRK>\n", line)[0].split(" <SEP> ")
        elif "<CNT>" in line and "<BRK>" in line:
            document = re.findall(r"<CNT> (.*) <SEP> <BRK>\n", line)[0].split(" <SEP> ")
        elif "<CNT>" in line and "<END>" in line:
            document = re.findall(r"<CNT> (.*) <SEP> <END>\n", line)[0].split(" <SEP> ")
        else:
            print("Error !")
            exit()
        for j in range(0, len(document)):
            if process_type == "sentence":
                w.write(document[j].strip() + "\n")
            else:
                assert (args.max_sentences - 1) % 2 == 0, "max_sentences must be an odd number"
                r = (args.max_sentences - 1) // 2
                upper_bound = j + r if j + r < len(document) else len(document)
                lower_bound = j - r if j - r >= 0 else 0
                context = document[j] + args.separator_token + " ".join(
                    document[lower_bound:j] + document[j + 1:upper_bound + 1]) \
                    .strip().replace("  ", " ").replace("  ", " ")
                assert context.count(args.separator_token) == 1, "sentence must contain only one [APPEND] symbol"
                w.write(context.strip() + "\n")
            w.flush()


    def single_worker(
            filename, worker_id=0, workers=1, process_type="context"
    ):
        file_suffix = str.zfill(str(worker_id), len(str(args.workers)))
        subprocess_file_name = args.temp_dir + os.path.basename(filename) + file_suffix
        with open(filename, "r", encoding="utf-8") as r:
            with open(subprocess_file_name, "w", encoding="utf-8") as w:
                size = os.fstat(r.fileno()).st_size
                chunk_size = size // workers
                offset = worker_id * chunk_size
                end = offset + chunk_size
                r.seek(offset)
                if offset > 0:
                    safe_readline(r)  # drop first incomplete line
                line = r.readline()
                total_count = 1
                while line:
                    proprcess_line(line, w, process_type=process_type)
                    if worker_id == 0:
                        print("{} | lines: {}".format(filename + file_suffix, total_count), end='\r')
                    # print("{} | lines: {}".format(filename + file_suffix, count))
                    if r.tell() > end:
                        break
                    line = r.readline()
                    total_count += 1


    def Process_file(filename, new_filename, process_type="context"):
        p = Pool(args.workers)
        for worker_id in range(args.workers):
            p.apply_async(single_worker, args=(filename, worker_id, args.workers, process_type))
        p.close()
        p.join()
        print("{} | All Subprocess Context files Processes Finished !".format(new_filename))
        with open(new_filename, "w", encoding="utf-8") as w:
            for worker_id in range(args.workers):
                file_suffix = str.zfill(str(worker_id), len(str(args.workers)))
                subprocess_file_name = args.temp_dir + os.path.basename(filename) + file_suffix
                with open(subprocess_file_name, "r", encoding="utf-8") as r:
                    for line in r:
                        w.write(line)
                os.remove(subprocess_file_name)  # remove subprocess_file
        print("{} | All Subprocess Context files Merged !".format(new_filename))


    file_list = (
        (args.src_set, args.new_src_set, "context"),
        (args.tgt_set, args.new_tgt_set, "sentence")
    )
    for filename, new_filename, processtype in file_list:
        Process_file(filename, new_filename, process_type=processtype)
