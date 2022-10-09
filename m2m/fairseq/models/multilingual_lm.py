# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
from torch import Tensor
from fairseq import checkpoint_utils
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    base_architecture as transformer_base_architecture,
)
from fairseq.models.transformer_from_pretrained_infoxlm import (
    TransformerFromPretrainedInfoXLMModel,
    TransformerEncoderFromPretrainedInfoXLM,
    TransformerDecoderFromPretrainedInfoXLM,
    upgrade_state_dict_with_infoxlm_weights
)
from fairseq.modules.transformer_layer import (
    TransformerDecoderLayer
)
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from fairseq import utils
from fairseq.file_io import PathManager
import torch.nn.functional as F
import random


def upgrade_state_dict_for_two_ffn(
        state_dict: Dict[str, Any],
        pretrained_infoxlm_checkpoint: str,
        num_layers: int,
) -> Dict[str, Any]:
    if not os.path.exists(pretrained_infoxlm_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_infoxlm_checkpoint))

    # state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_infoxlm_checkpoint)
    with open(PathManager.get_local_path(pretrained_infoxlm_checkpoint), "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))
    infoxlm_state_dict = state["model"]
    # load weights for sharing bottom layers
    for key in infoxlm_state_dict.keys():
        if 'layers' in key and int(key.split('.')[3]) > 2 * num_layers - 1:
            continue
        if not key.startswith('decoder.'):
            continue
        if 'lm_head' not in key:
            if 'in_proj_weight' in key:
                q, k, v = infoxlm_state_dict[key].chunk(3, dim=0)
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'q_proj.weight').replace(f'.{i_layer}', f'.{i_layer // 2}')] = q
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'k_proj.weight').replace(f'.{i_layer}', f'.{i_layer // 2}')] = k
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'v_proj.weight').replace(f'.{i_layer}', f'.{i_layer // 2}')] = v
                else:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'q_proj.weight').replace(f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = q
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'k_proj.weight').replace(f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = k
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'v_proj.weight').replace(f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = v
            elif 'in_proj_bias' in key:
                q, k, v = infoxlm_state_dict[key].chunk(3, dim=0)
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'q_proj.bias').replace(f'.{i_layer}', f'.{i_layer // 2}')] = q
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'k_proj.bias').replace(f'.{i_layer}', f'.{i_layer // 2}')] = k
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'v_proj.bias').replace(f'.{i_layer}', f'.{i_layer // 2}')] = v
                else:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'q_proj.bias').replace(f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = q
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'k_proj.bias').replace(f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = k
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'v_proj.bias').replace(f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = v
            elif 'fc1' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('fc1', 'fc3').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
            elif 'fc2' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('fc2', 'fc4').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
            elif 'final_layer_norm' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('final_layer_norm', 'ffn_layer_norm').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
            elif 'self_attn.out_proj' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 1:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('self_attn.out_proj', 'encoder_attn.out_proj').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
            elif 'self_attn.k_proj' in key or 'self_attn.v_proj' in key or 'self_attn.q_proj' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 1:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('self_attn', 'encoder_attn').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
            elif 'self_attn_layer_norm' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 1:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('self_attn_layer_norm', 'encoder_attn_layer_norm').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
            elif 'emb_layer_norm' in key:
                state_dict[key.replace('decoder.sentence_encoder.emb_layer_norm', 'layernorm_embedding')] = infoxlm_state_dict[key]
            elif 'embed_positions' in key:
                state_dict[key.replace('decoder.sentence_encoder.', '')] = infoxlm_state_dict[key][:state_dict[key.replace('decoder.sentence_encoder.', '')].size(0)]
            elif 'embed_tokens' in key:
                state_dict[key.replace('decoder.sentence_encoder.', '')][:infoxlm_state_dict[key].size(0)] = infoxlm_state_dict[key]
            else:
                state_dict[key.replace('decoder.sentence_encoder.', '')] = infoxlm_state_dict[key]

    return state_dict


def upgrade_state_dict_for_two_ffn_with_ls(
        state_dict: Dict[str, Any],
        pretrained_infoxlm_checkpoint: str,
        num_layers: int,
        ls_layers: int,
        pretrained_layers: int,
        langs: list
) -> Dict[str, Any]:
    if not os.path.exists(pretrained_infoxlm_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_infoxlm_checkpoint))

    # state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_infoxlm_checkpoint)
    with open(PathManager.get_local_path(pretrained_infoxlm_checkpoint), "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))
    infoxlm_state_dict = state["model"]
    # load weights for sharing bottom layers
    for key in infoxlm_state_dict.keys():
        if 'layers' in key and int(key.split('.')[3]) > 2 * num_layers - 1:
            continue
        if not key.startswith('decoder.'):
            continue
        if 'lm_head' not in key:
            if 'in_proj_weight' in key:
                q, k, v = infoxlm_state_dict[key].chunk(3, dim=0)
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'q_proj.weight').replace(f'.{i_layer}', f'.{i_layer // 2}')] = q
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'k_proj.weight').replace(f'.{i_layer}', f'.{i_layer // 2}')] = k
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'v_proj.weight').replace(f'.{i_layer}', f'.{i_layer // 2}')] = v
                else:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'q_proj.weight').replace(f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = q
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'k_proj.weight').replace(f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = k
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'v_proj.weight').replace(f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = v
            elif 'in_proj_bias' in key:
                q, k, v = infoxlm_state_dict[key].chunk(3, dim=0)
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'q_proj.bias').replace(f'.{i_layer}', f'.{i_layer // 2}')] = q
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'k_proj.bias').replace(f'.{i_layer}', f'.{i_layer // 2}')] = k
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'v_proj.bias').replace(f'.{i_layer}', f'.{i_layer // 2}')] = v
                else:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'q_proj.bias').replace(f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = q
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'k_proj.bias').replace(f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = k
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'v_proj.bias').replace(f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = v
            elif 'fc1' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('fc1', 'fc3').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
            elif 'fc2' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('fc2', 'fc4').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
            elif 'final_layer_norm' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('final_layer_norm', 'ffn_layer_norm').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
            elif 'self_attn.out_proj' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 1:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('self_attn.out_proj', 'encoder_attn.out_proj').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
            elif 'self_attn.k_proj' in key or 'self_attn.v_proj' in key or 'self_attn.q_proj' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 1:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('self_attn', 'encoder_attn').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
            elif 'self_attn_layer_norm' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 1:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('self_attn_layer_norm', 'encoder_attn_layer_norm').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
            elif 'emb_layer_norm' in key:
                state_dict[key.replace('decoder.sentence_encoder.emb_layer_norm', 'layernorm_embedding')] = infoxlm_state_dict[key]
            elif 'embed_positions' in key:
                state_dict[key.replace('decoder.sentence_encoder.', '')] = infoxlm_state_dict[key][:state_dict[key.replace('decoder.sentence_encoder.', '')].size(0)]
            elif 'embed_tokens' in key:
                state_dict[key.replace('decoder.sentence_encoder.', '')][:infoxlm_state_dict[key].size(0)] = infoxlm_state_dict[key]
            else:
                state_dict[key.replace('decoder.sentence_encoder.', '')] = infoxlm_state_dict[key]
    # load weights for language-specific layers
    start_layers = pretrained_layers - 2 * ls_layers
    ls_name = "ls_decoder"

    def layers2lslayers(i_layer):
        return (i_layer - start_layers) // 2

    for ls_id in range(len(langs)):
        for key in infoxlm_state_dict.keys():
            if 'layers' in key and int(key.split('.')[3]) < start_layers:
                continue
            if not key.startswith('decoder.'):
                continue
            if 'lm_head' not in key:
                if 'in_proj_weight' in key:
                    q, k, v = infoxlm_state_dict[key].chunk(3, dim=0)
                    i_layer = int(key.split('.')[3])
                    if i_layer % 2 == 0:
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace('in_proj_weight', 'q_proj.weight').replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}')] = q
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace('in_proj_weight', 'k_proj.weight').replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}')] = k
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace('in_proj_weight', 'v_proj.weight').replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}')] = v
                    else:
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace('in_proj_weight', 'q_proj.weight').replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}').replace('self_attn', 'encoder_attn')] = q
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace('in_proj_weight', 'k_proj.weight').replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}').replace('self_attn', 'encoder_attn')] = k
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace('in_proj_weight', 'v_proj.weight').replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}').replace('self_attn', 'encoder_attn')] = v
                elif 'in_proj_bias' in key:
                    q, k, v = infoxlm_state_dict[key].chunk(3, dim=0)
                    i_layer = int(key.split('.')[3])
                    if i_layer % 2 == 0:
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace('in_proj_bias', 'q_proj.bias').replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}')] = q
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace('in_proj_bias', 'k_proj.bias').replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}')] = k
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace('in_proj_bias', 'v_proj.bias').replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}')] = v
                    else:
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace('in_proj_bias', 'q_proj.bias').replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}').replace('self_attn', 'encoder_attn')] = q
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace('in_proj_bias', 'k_proj.bias').replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}').replace('self_attn', 'encoder_attn')] = k
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace('in_proj_bias', 'v_proj.bias').replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}').replace('self_attn', 'encoder_attn')] = v
                elif 'fc1' in key:
                    i_layer = int(key.split('.')[3])
                    if i_layer % 2 == 0:
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace('fc1', 'fc3').replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}')] = infoxlm_state_dict[key]
                    else:
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}')] = infoxlm_state_dict[key]
                elif 'fc2' in key:
                    i_layer = int(key.split('.')[3])
                    if i_layer % 2 == 0:
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace('fc2', 'fc4').replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}')] = infoxlm_state_dict[key]
                    else:
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}')] = infoxlm_state_dict[key]
                elif 'final_layer_norm' in key:
                    i_layer = int(key.split('.')[3])
                    if i_layer % 2 == 0:
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace('final_layer_norm', 'ffn_layer_norm').replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}')] = infoxlm_state_dict[key]
                    else:
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}')] = infoxlm_state_dict[key]
                elif 'self_attn.out_proj' in key:
                    i_layer = int(key.split('.')[3])
                    if i_layer % 2 == 1:
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace('self_attn.out_proj', 'encoder_attn.out_proj').replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}')] = infoxlm_state_dict[key]
                    else:
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}')] = infoxlm_state_dict[key]
                elif 'self_attn.k_proj' in key or 'self_attn.v_proj' in key or 'self_attn.q_proj' in key:
                    i_layer = int(key.split('.')[3])
                    if i_layer % 2 == 1:
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace('self_attn', 'encoder_attn').replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}')] = infoxlm_state_dict[key]
                    else:
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}')] = infoxlm_state_dict[key]
                elif 'self_attn_layer_norm' in key:
                    i_layer = int(key.split('.')[3])
                    if i_layer % 2 == 1:
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace('self_attn_layer_norm', 'encoder_attn_layer_norm').replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}')] = infoxlm_state_dict[key]
                    else:
                        state_dict[key.replace('decoder.sentence_encoder.layers', ls_name).replace(f'.{i_layer}', f'.{ls_id}.{layers2lslayers(i_layer)}')] = infoxlm_state_dict[key]
    return state_dict


def upgrade_state_dict_for_ca_first_two_ffn(
        state_dict: Dict[str, Any], pretrained_infoxlm_checkpoint: str, num_layers: int
) -> Dict[str, Any]:
    if not os.path.exists(pretrained_infoxlm_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_infoxlm_checkpoint))

    # state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_infoxlm_checkpoint)
    with open(PathManager.get_local_path(pretrained_infoxlm_checkpoint), "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))
    infoxlm_state_dict = state["model"]

    for key in infoxlm_state_dict.keys():
        if 'layers' in key and int(key.split('.')[3]) > 2 * num_layers - 1:
            continue
        if not key.startswith('decoder.'):
            continue
        if 'lm_head' not in key:
            if 'in_proj_weight' in key:
                q, k, v = infoxlm_state_dict[key].chunk(3, dim=0)
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 1:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'q_proj.weight').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = q
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'k_proj.weight').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = k
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'v_proj.weight').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = v
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'q_proj.weight').replace(
                            f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = q
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'k_proj.weight').replace(
                            f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = k
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'v_proj.weight').replace(
                            f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = v
            elif 'in_proj_bias' in key:
                q, k, v = infoxlm_state_dict[key].chunk(3, dim=0)
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 1:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'q_proj.bias').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = q
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'k_proj.bias').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = k
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'v_proj.bias').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = v
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'q_proj.bias').replace(
                            f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = q
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'k_proj.bias').replace(
                            f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = k
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'v_proj.bias').replace(
                            f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = v
            elif 'fc1' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('fc1', 'fc3').replace(f'.{i_layer}',
                                                                                                          f'.{i_layer // 2}')] = \
                        infoxlm_state_dict[key]
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = \
                        infoxlm_state_dict[key]
            elif 'fc2' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('fc2', 'fc4').replace(f'.{i_layer}',
                                                                                                          f'.{i_layer // 2}')] = \
                        infoxlm_state_dict[key]
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = \
                        infoxlm_state_dict[key]
            elif 'final_layer_norm' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('final_layer_norm',
                                                                                    'ffn_layer_norm').replace(
                        f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = \
                        infoxlm_state_dict[key]
            elif 'self_attn.out_proj' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('self_attn.out_proj',
                                                                                    'encoder_attn.out_proj').replace(
                        f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = \
                        infoxlm_state_dict[key]
            elif 'self_attn.k_proj' in key or 'self_attn.v_proj' in key or 'self_attn.q_proj' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('self_attn', 'encoder_attn').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = \
                        infoxlm_state_dict[key]
            elif 'self_attn_layer_norm' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('self_attn_layer_norm',
                                                                                    'encoder_attn_layer_norm').replace(
                        f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = \
                        infoxlm_state_dict[key]
            elif 'emb_layer_norm' in key:
                state_dict[key.replace('decoder.sentence_encoder.emb_layer_norm', 'layernorm_embedding')] = \
                    infoxlm_state_dict[key]
            elif 'embed_positions' in key:
                state_dict[key.replace('decoder.sentence_encoder.', '')] = infoxlm_state_dict[key][:state_dict[
                    key.replace('decoder.sentence_encoder.', '')].size(0)]
            elif 'embed_tokens' in key:
                state_dict[key.replace('decoder.sentence_encoder.', '')][:infoxlm_state_dict[key].size(0)] = \
                    infoxlm_state_dict[key]
            else:
                state_dict[key.replace('decoder.sentence_encoder.', '')] = infoxlm_state_dict[key]

    return state_dict


def upgrade_state_dict_for_deep_model(
        state_dict: Dict[str, Any], pretrained_infoxlm_checkpoint: str, num_layers: int
) -> Dict[str, Any]:
    if not os.path.exists(pretrained_infoxlm_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_infoxlm_checkpoint))

    # state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_infoxlm_checkpoint)
    with open(PathManager.get_local_path(pretrained_infoxlm_checkpoint), "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))
    mt_state_dict = state["model"]

    for key in mt_state_dict.keys():
        if "loss_weight" not in key:
            state_dict[key] = mt_state_dict[key]
    return state_dict


@register_model("multilingual_lm")
class MultilingualLMModel(TransformerFromPretrainedInfoXLMModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerFromPretrainedInfoXLMModel.add_args(parser)
        parser.add_argument(
            "--variant",
            type=str,
            metavar="STR",
        )
        parser.add_argument(
            "--encoder-group-type",
            type=str,
            metavar="STR",
        )
        parser.add_argument(
            "--encoder-group-layers",
            type=int,
            metavar="STR",
        )
        parser.add_argument(
            "--decoder-group-layers",
            type=int,
            metavar="STR",
        )
        parser.add_argument(
            "--encoder-ls-layers",
            type=int,
            metavar="STR",
        )
        parser.add_argument(
            "--decoder-ls-layers",
            type=int,
            metavar="STR",
        )
        parser.add_argument(
            "--keep-lang-id-prob",
            type=float,
            metavar="STR",
        )
        parser.add_argument(
            "--softmax-temperature",
            type=int,
            metavar="STR",
        )
        parser.add_argument(
            "--decoder-ls-depth-strategy",
            type=str,
            metavar="STR",
        )
        parser.add_argument(
            "--decoder-ls-layers-list",
            type=str,
            metavar="STR",
        )
        parser.add_argument(
            "--decoder-group-type",
            type=str,
            metavar="STR",
        )
        parser.add_argument(
            "--freeze-encoder",
            action="store_true",
        )
        parser.add_argument(
            "--freeze-decoder",
            action="store_true",
        )
        parser.add_argument(
            "--freeze-embedding",
            action="store_true",
        )
        parser.add_argument(
            "--language-specific-lr",
            type=str,
            metavar="STR",
        )
        parser.add_argument(
            "--pretrained-mt-model",
            type=str,
            metavar="STR",
        )
        parser.add_argument(
            "--pretrained-gpt-checkpoint",
            type=str,
            metavar="STR",
        )
        parser.add_argument(
            "--use-adapter",
            action="store_true",
        )
        parser.add_argument(
            "--adapter-dim",
            type=int,
        )
        parser.add_argument(
            "--only-encoder",
            action="store_true",
        )
        parser.add_argument(
            "--only-decoder",
            action="store_true",
        )
        parser.add_argument(
            "--drop-adapter",
            action="store_true",
        )
        parser.add_argument(
            "--adapter-dropout",
            type=float,
        )

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        if hasattr(args, "pretrained_mt_model") and os.path.exists(args.pretrained_mt_model):
            mt_loaded_state_dict = upgrade_state_dict_with_infoxlm_weights(
                state_dict=self.state_dict(),
                pretrained_infoxlm_checkpoint=args.pretrained_infoxlm_checkpoint,
                num_layers=args.encoder_layers,
            )
            self.load_state_dict(mt_loaded_state_dict, strict=True)
            print("Loading pretrained mt model from {0}".format(args.pretrained_mt_checkpoint))

    @classmethod
    def build_model(cls, args, task, cls_dictionary=MaskedLMDictionary):
        assert args.enable_lang_ids, "Please 'set enable-lang-ids' True"
        return super().build_model(args, task)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return MultilingualLMDecoder(args, tgt_dict, embed_tokens)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        if args.only_decoder:
            return None
        else:
            return TransformerEncoder(args, src_dict, embed_tokens)

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.decoder.max_positions()

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            src_lang_id=None,
            tgt_lang_id=None,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            **extra

    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out=None
        if self.encoder is not None:
            encoder_out = self.encoder(
                src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
            )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            src_lang_id=src_lang_id,
            tgt_lang_id=tgt_lang_id,
        )
        return decoder_out





def Weight(in_features, value=0.0):
    m = nn.Parameter(torch.Tensor(in_features))
    nn.init.constant_(m, value)
    return m


def Linear(in_features, out_features, dropout=0.0, bias=True):
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def DeepGatedLinear(in_features, out_features, dropout=0.0, bias=True):
    return nn.Sequential(
        Linear(in_features, out_features * 4, dropout, bias),
        nn.GLU(),
        Linear(out_features * 2, out_features * 2, dropout, bias),
        nn.GLU(),
        Linear(out_features, out_features, dropout, bias),
    )


def ShallowGatedLinear(in_features, out_features, dropout=0.0, bias=True):
    return nn.Sequential(
        Linear(in_features, out_features * 2, dropout, bias),
        nn.GLU(),
    )


class AdapterLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.adapter_dropout = getattr(args, "adapter_dropout", args.dropout)
        self.dropout_module = FairseqDropout(
            self.adapter_dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before
        self.adapter_dim=args.adapter_dim
        self.fc1 = self.build_fc1(
            self.embed_dim,
            self.adapter_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            self.adapter_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = x + residual
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class MultilingualLMDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=True):
        if args.variant == "addffn" or args.variant == "two-attn":
            no_encoder_attn = False
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.langs = args.langs
        if hasattr(args, "pretrained_infoxlm_checkpoint") and os.path.exists(args.pretrained_infoxlm_checkpoint):
            if args.variant == 'addffn':
                infoxlm_loaded_state_dict = upgrade_state_dict_for_two_ffn(
                    state_dict=self.state_dict(),
                    pretrained_infoxlm_checkpoint=args.pretrained_infoxlm_checkpoint,
                    num_layers=args.decoder_layers,
                )
            elif args.variant == 'normal':
                infoxlm_loaded_state_dict = upgrade_state_dict_with_infoxlm_weights(
                    state_dict=self.state_dict(),
                    pretrained_infoxlm_checkpoint=args.pretrained_infoxlm_checkpoint,
                    num_layers=args.decoder_layers,
                )
            elif args.variant == 'two-attn':
                infoxlm_loaded_state_dict = upgrade_state_dict_with_infoxlm_weights(
                    state_dict=self.state_dict(),
                    pretrained_infoxlm_checkpoint=args.pretrained_infoxlm_checkpoint,
                    num_layers=args.decoder_layers,
                    shared_cross_attn = True
                )
            self.load_state_dict(infoxlm_loaded_state_dict, strict=True)
            print("Loading decoder from {0}".format(args.pretrained_infoxlm_checkpoint))
        else:
            print("Random Initialization for Decoder")

        self.use_adapter = getattr(args, "use_adapter", False)
        self.drop_adapter = getattr(args, "drop_adapter", False)
        if self.use_adapter:
            self.adapter = nn.ModuleList([])
            self.adapter.extend(
                nn.ModuleList(
                    [
                        AdapterLayer(args)
                        for _ in range(len(self.langs))
                    ])
            )
        if getattr(args, 'freeze_decoder', False):
            for param in self.layers.parameters():
                param.requires_grad = False
        if getattr(args, 'freeze_adapter', False) and getattr(args, "use_adapter", False):
            for param in self.adapter.parameters():
                param.requires_grad = False
        if getattr(args, 'freeze_embedding', False):
            for param in self.embed_tokens.parameters():
                param.requires_grad = False

    def build_decoder_layer(self, args, no_encoder_attn=False):
        if args.variant == 'normal':
            layer = TransformerDecoderLayer(args, no_encoder_attn=True)
        elif args.variant == 'two-attn':
            layer = TransformerSelfAttnDecoderLayer(args, no_encoder_attn=False)
        elif args.variant == 'addffn':
            layer = XLMTAddFFN(args, no_encoder_attn)
        else:
            raise NotImplementedError
        if getattr(args, "checkpoint_activations", False):
            layer = checkpoint_wrapper(layer)
        return layer

    def forward(
            self,
            prev_output_tokens,
            src_lang_id=None,
            tgt_lang_id=None,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
    ):
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lang_id=src_lang_id,
            tgt_lang_id=tgt_lang_id,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def get_lang_id(self, lang_id):
        # assert lang_id is not None, "lang_id in Batch can not be None Type !"
        if isinstance(lang_id, int):
            return lang_id
        else:
            return int(lang_id[0].cpu())



    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            src_lang_id,
            tgt_lang_id,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = []
        tgt_lang_id = self.get_lang_id(tgt_lang_id) - 1
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                        encoder_out is not None
                        and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)


        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.use_adapter:
            if torch.rand((1,)) < 0.5 or not self.training or not self.drop_adapter:
                x = self.adapter[tgt_lang_id](x)


        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

class TransformerSelfAttnDecoderLayer(TransformerDecoderLayer):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_self_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False


    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)


        residual = x
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        if prev_attn_state is not None:
            prev_key, prev_value = prev_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_attn_state[2]
            assert incremental_state is not None
            self.encoder_attn._set_input_buffer(incremental_state, saved_state)

        x, attn = self.encoder_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


class XLMTAddFFN(TransformerDecoderLayer):

    def __init__(
            self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.fc3 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc4 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.ffn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        ###############################################

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        ###############################################

        residual = x
        if self.normalize_before:
            x = self.ffn_layer_norm(x)

        x = self.activation_fn(self.fc3(x))
        x = self.activation_dropout_module(x)
        x = self.fc4(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.ffn_layer_norm(x)

        ###############################################

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        ###############################################

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


@register_model_architecture(
    "multilingual_lm", "multilingual_lm_base"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.init_encoder_only = getattr(args, "init_encoder_only", False)
    args.init_decoder_only = getattr(args, "init_decoder_only", False)
    args.max_positions = getattr(args, "max_positions", 512)

    #
    # args.encoder_ls_layers = getattr(args, "encoder_ls_layers", 3)
    args.encoder_group_layers = getattr(args, "encoder_group_layers", args.encoder_layers)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_ls_layers = getattr(args, "decoder_ls_layers", args.decoder_layers)
    args.decoder_group_layers = getattr(args, "decoder_group_layers", args.decoder_layers + args.decoder_ls_layers)
    args.decoder_ls_depth_strategy = getattr(args, "decoder_ls_depth_strategy", "static-same")
    args.keep_lang_id_prob = getattr(args, "keep_lang_id_prob", 1.0)
    args.encoder_group_type = getattr(args, "encoder_group_type", "sparse")
    args.decoder_group_type = getattr(args, "decoder_group_type", "sparse")
    args.adapter_dim = getattr(args, "adapter_dim", args.decoder_ffn_embed_dim)
    args.only_encoder = getattr(args, "only_encoder", False)
    args.only_decoder = getattr(args, "only_decoder", False)

