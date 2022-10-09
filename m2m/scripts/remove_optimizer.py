from fairseq.file_io import PathManager
from fairseq.checkpoint_utils import torch_persistent_save
import torch
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-path', type=str,
                        default=r'/path/to/SharedTask/thunder/small_task2/Filter_v1/model/24L-12L-A100//avg3_18.pt',
                        help='input src')
    parser.add_argument('--new-path', '-new-path', type=str,
                        default=r'/path/to/SharedTask/xlm-t/small_track2_24L_12L.pt', help='input src')
    args = parser.parse_args()
    return args


def remove_optmizer(model_path, new_model_path):
    with open(PathManager.get_local_path(model_path), "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))
        print("Successfully loading from {}".format(model_path))
    state_dict = {
        "cfg": state["cfg"],
        "args": state["args"],
        "model": state["model"],
        "optimizer_history": None,
        "extra_state": None,
    }
    with PathManager.open(new_model_path, "wb") as f:
        torch_persistent_save(state_dict, f)
    print("Successfully saving to {}".format(new_model_path))


if __name__ == "__main__":
    args = parse_args()
    # remove_optmizer("/path/to/SharedTask/PretrainedModel/mm100_175M/flores101_mm100_175M/", "model.pt", "simple_model.pt")
    # remove_optmizer("/path/to/SharedTask/PretrainedModel/mm100_615M/flores101_mm100_615M/", "model.pt", "simple_model.pt")
    remove_optmizer(args.path, args.new_path)
