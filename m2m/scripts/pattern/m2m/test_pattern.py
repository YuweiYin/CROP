import json
import logging
import time
import os

import torch
from typing import NamedTuple
from dynalab.handler.base_handler import BaseDynaHandler
from dynalab.tasks.flores_small1 import TaskIO
from omegaconf import OmegaConf
from argparse import Namespace

from fairseq.sequence_generator import SequenceGenerator
from fairseq import checkpoint_utils, utils
from fairseq.data import encoders
from fairseq.data import data_utils
from fairseq_cli.generate import get_symbols_to_strip_from_output
from fairseq.data import Dictionary

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Tell Torchserve to let use do the deserialization
os.environ["TS_DECODE_INPUT_REQUEST"] = "false"


def get_pivot_table(path):
    # PIVOT Pairs
    with open(path, "r", encoding="utf-8") as r:
        pivot_lines = r.readlines()
    if len(pivot_lines) > 0 and pivot_lines[0].strip() != "":
        pivot_table = pivot_lines[0].replace("->", "-").strip().split(",")
    else:
        pivot_table = []
    return pivot_table


PIVOT_TABLE = get_pivot_table("small_track2_pivot_pairs.txt")


def mapping(languages: str) -> dict:
    return dict(
        tuple(pair.split(":"))
        for pair in languages.strip().replace("\n", "").split(",")
    )


ISO2M100 = mapping(
    """
afr:af,amh:am,ara:ar,asm:as,ast:ast,azj:az,bel:be,ben:bn,bos:bs,bul:bg,
cat:ca,ceb:ceb,ces:cs,ckb:ku,cym:cy,dan:da,deu:de,ell:el,eng:en,est:et,
fas:fa,fin:fi,fra:fr,ful:ff,gle:ga,glg:gl,guj:gu,hau:ha,heb:he,hin:hi,
hrv:hr,hun:hu,hye:hy,ibo:ig,ind:id,isl:is,ita:it,jav:jv,jpn:ja,kam:kam,
kan:kn,kat:ka,kaz:kk,kea:kea,khm:km,kir:ky,kor:ko,lao:lo,lav:lv,lin:ln,
lit:lt,ltz:lb,lug:lg,luo:luo,mal:ml,mar:mr,mkd:mk,mlt:mt,mon:mn,mri:mi,
msa:ms,mya:my,nld:nl,nob:no,npi:ne,nso:ns,nya:ny,oci:oc,orm:om,ory:or,
pan:pa,pol:pl,por:pt,pus:ps,ron:ro,rus:ru,slk:sk,slv:sl,sna:sn,snd:sd,
som:so,spa:es,srp:sr,swe:sv,swh:sw,tam:ta,tel:te,tgk:tg,tgl:tl,tha:th,
tur:tr,ukr:uk,umb:umb,urd:ur,uzb:uz,vie:vi,wol:wo,xho:xh,yor:yo,zho_simp:zh,
zho_trad:zh,zul:zu
"""
)

ISO2OUR = mapping(
    """
afr:af,amh:am,ara:ar,asm:as,ast:ast,azj:az,bel:be,ben:bn,bos:bs,bul:bg,
cat:ca,ceb:ceb,ces:cs,ckb:ku,cym:cy,dan:da,deu:de,ell:el,eng:en,est:et,
fas:fa,fin:fi,fra:fr,ful:ff,gle:ga,glg:gl,guj:gu,hau:ha,heb:he,hin:hi,
hrv:hr,hun:hu,hye:hy,ibo:ig,ind:id,isl:is,ita:it,jav:jv,jpn:ja,kam:kam,
kan:kn,kat:ka,kaz:kk,kea:kea,khm:km,kir:ky,kor:ko,lao:lo,lav:lv,lin:ln,
lit:lt,ltz:lb,lug:lg,luo:luo,mal:ml,mar:mr,mkd:mk,mlt:mt,mon:mn,mri:mi,
msa:ms,mya:my,nld:nl,nob:no,npi:ne,nso:ns,nya:ny,oci:oc,orm:om,ory:or,
pan:pa,pol:pl,por:pt,pus:ps,ron:ro,rus:ru,slk:sk,slv:sl,sna:sn,snd:sd,
som:so,spa:es,srp:sr,swe:sv,swh:sw,tam:ta,tel:te,tgk:tg,tgl:tl,tha:th,
tur:tr,ukr:uk,umb:umb,urd:ur,uzb:uz,vie:vi,wol:wo,xho:xh,yor:yo,zho_simp:zh,
zho_trad:zt,zul:zu
"""
)

MODEL_TYPE = "M2M"  # Our and M2M
LANGS = "af,am,ar,ast,az,ba,be,bg,bn,br,bs,ca,ceb,cs,cy,da,de,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kk,km,kn,ko,lb,lg,ln,lo,lt,lv,mg,mk,ml,mn,mr,ms,my,ne,nl,no,ns,oc,or,pa,pl,ps,pt,ro,ru,sd,si,sk,sl,so,sq,sr,ss,su,sv,sw,ta,th,tl,tn,tr,uk,ur,uz,vi,wo,xh,yi,yo,zh,zu,eu,te".split(
    ",")
ISO2SIMP = ISO2OUR if MODEL_TYPE == "Our" else ISO2M100


def encode_fn(x, bpe, tokenizer):
    if tokenizer is not None:
        x = tokenizer.encode(x)
    if bpe is not None:
        x = bpe.encode(x)
    return x


def decode_fn(x, bpe, tokenizer):
    if bpe is not None:
        x = bpe.decode(x)
    if tokenizer is not None:
        x = tokenizer.decode(x)
    return x


class Handler(BaseDynaHandler):
    """Use Fairseq model for translation.
    To use this handler, download one of the Flores pretrained model:
    615M parameters:
        https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_615M.tar.gz
    175M parameters:
        https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_175M.tar.gz
    and extract the files next to this one.
    Notably there should be a "dict.txt" and a "sentencepiece.bpe.model".
    """

    def initialize(self, context):
        """
        load model and extra files.
        """
        logger.info(
            f"Will initialize with system_properties: {context.system_properties}"
        )
        model_pt_path, model_file_dir, device = self._handler_initialize(context)
        logger.info("{} | {} | {}".format(model_pt_path, model_file_dir, device))
        self.device = device
        # Initialization
        self.support_pivot = True
        self.cfg = Namespace()
        self.fp16 = False
        self.cfg.generation = OmegaConf.create(
            {'_name': None, 'beam': 4, 'nbest': 1, 'max_len_a': 0.0, 'max_len_b': 196, 'min_len': 1,
             'unnormalized': False, 'lenpen': 1.0, 'unkpen': 0.0})
        self.cfg.tokenizer = None
        self.tokenizer = encoders.build_tokenizer(self.cfg.tokenizer)
        self.cfg.bpe = OmegaConf.create({'_name': 'sentencepiece', 'sentencepiece_model': 'spm.model'})
        self.bpe = encoders.build_bpe(self.cfg.bpe)
        logger.info("SPM: {}".format(self.bpe))
        # Set dictionaries
        self.src_dict = Dictionary.load("dict.txt")
        if MODEL_TYPE == "Our":
            for lang in ISO2SIMP.values():
                self.src_dict.add_symbol(f"__{lang}__")
            self.src_dict.pad_to_multiple_(padding_factor=8)
        self.tgt_dict = self.src_dict

        # Build Model
        if MODEL_TYPE == "M2M":
            self.models, _model_args = checkpoint_utils.load_transformer_model_ensemble(model_pt_path.split(),
                                                                                        src_dict=self.src_dict,
                                                                                        tgt_dict=self.tgt_dict)
        else:
            self.models, _model_args = checkpoint_utils.load_xlmt_model_ensemble(model_pt_path.split(),
                                                                                 src_dict=self.src_dict,
                                                                                 tgt_dict=self.tgt_dict)
        logger.info("Model: {}".format(self.models))

        # Set model
        for model in self.models:
            if model is None:
                continue
            if self.fp16:
                model.half()
            model.eval().to(self.device)
        # Initialize generator
        self.generator = SequenceGenerator(
            self.models,
            self.tgt_dict,
            beam_size=self.cfg.generation.beam,
            max_len_a=self.cfg.generation.max_len_a,
            max_len_b=self.cfg.generation.max_len_b,
            min_len=self.cfg.generation.min_len,
            normalize_scores=not self.cfg.generation.unnormalized,
            len_penalty=self.cfg.generation.lenpen,
            unk_penalty=self.cfg.generation.unkpen,
            symbols_to_strip_from_output=set([self.src_dict.index(f"__{lang}__") for lang in ISO2SIMP.values()])
        )
        self.max_positions = utils.resolve_max_positions(*[model.max_positions() for model in self.models])
        self.truncate_source = min(196, self.max_positions[0] - 3)
        self.taskIO = TaskIO()
        self.initialized = True

    def lang_token(self, lang: str) -> int:
        """Converts the ISO 639-3 language code to MM100 language codes."""
        simple_lang = ISO2SIMP[lang]
        token = self.src_dict.indices[f"__{simple_lang}__"]
        return token

    def tokenize(self, line: str) -> list:
        tokens = self.src_dict.encode_line(encode_fn(line, self.bpe, self.tokenizer), add_if_not_exist=False).long()[
                 :self.truncate_source]
        return tokens.tolist()

    def preprocess_one(self, sample) -> dict:
        """
        preprocess data into a format that the model can do inference on
        """
        # {main: (src, tgt)} | {main: (tgt, None)}
        tokens = self.tokenize(sample["sourceText"])
        src_lang_token = self.lang_token(sample["sourceLanguage"])
        tgt_lang_token = self.lang_token(sample["targetLanguage"])
        if MODEL_TYPE == "M2M":
            return {
                "src_tokens": [src_lang_token] + tokens,
                "src_length": len(tokens) + 1,
                "tgt_token": tgt_lang_token,
            }
        else:
            return {
                "src_tokens": [tgt_lang_token] + tokens,
                "src_length": len(tokens) + 1,
            }

    def preprocess(self, samples) -> dict:
        samples = [self.preprocess_one(s) for s in samples]
        if MODEL_TYPE == "M2M":
            prefix_tokens = torch.tensor([[s["tgt_token"]] for s in samples])
        else:
            prefix_tokens = None
        src_lengths = torch.tensor([s["src_length"] for s in samples])
        src_tokens = data_utils.collate_tokens(
            [torch.tensor(s["src_tokens"]) for s in samples],
            self.src_dict.pad(),
            self.src_dict.eos(),
        )
        return {
            "nsentences": len(samples),
            "ntokens": src_lengths.sum().item(),
            "net_input": {
                "src_tokens": src_tokens.to(self.device),
                "src_lengths": src_lengths.to(self.device),
            },
            "prefix_tokens": prefix_tokens.to(self.device) if prefix_tokens is not None else None,
        }

    @torch.no_grad()
    def inference(self, input_data: dict) -> list:
        results = self.generator.generate(
            self.models, input_data, prefix_tokens=input_data['prefix_tokens']
        )
        inference_outputs = []
        for hypos in results:
            for hypo in hypos[: min(len(hypos), self.cfg.generation.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=None,
                    alignment=None,
                    align_dict=None,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=None,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.generator),
                )
            inference_outputs.append(decode_fn(hypo_str, self.bpe, self.tokenizer))
        return inference_outputs

    def postprocess(self, translations, samples: list) -> list:
        """
        post process inference output into a response.
        response should be a list of json
        the response format will need to pass the validation in
        ```
        dynalab.tasks.flores_small1.TaskIO().verify_response(response)
        ```
        """
        return [
            # Signing required by dynabench, don't remove.
            self.taskIO.sign_response(
                {"id": sample["uid"], "translatedText": translation},
                sample,
            )
            for translation, sample in zip(translations, samples)
        ]


_service = Handler()


def deserialize(torchserve_data: list) -> list:
    samples = []
    for torchserve_sample in torchserve_data:
        data = torchserve_sample["body"]
        # In case torchserve did the deserialization for us.
        if isinstance(data, dict):
            samples.append(data)
        elif isinstance(data, (bytes, bytearray)):
            lines = data.decode("utf-8").splitlines()
            for i, l in enumerate(lines):
                try:
                    samples.append(json.loads(l))
                except Exception as e:
                    logging.error(f"Couldn't deserialize line {i}: {l}")
                    logging.exception(e)
        else:
            logging.error(f"Unexpected payload: {data}")

    return samples


def handle_mini_batch(service, samples):
    n = len(samples)
    start_time = time.time()
    samples = [
        {
            'uid': samples[i]['uid'],
            'sourceLanguage': samples[i]['sourceLanguage'],
            'targetLanguage': samples[i]['targetLanguage'],
            'sourceText': samples[i]['sourceText']
        } for i in range(len(samples))
    ]
    if _lang_pair(samples[0]['sourceLanguage'], samples[0]['targetLanguage']) in PIVOT_TABLE:
        src2pivot_samples = [
            {
                'uid': samples[i]['uid'],
                'sourceLanguage': samples[i]['sourceLanguage'],
                'targetLanguage': 'eng',
                'sourceText': samples[i]['sourceText']
            } for i in range(len(samples))
        ]
        src_input_data = service.preprocess(src2pivot_samples)
        logger.info(
            f"Preprocessed a batch of size {n} ({n / (time.time() - start_time):.2f} samples / s)"
        )
        start_time = time.time()
        output = service.inference(src_input_data)
        pivot2tgt_samples = [
            {
                'uid': samples[i]['uid'],
                'sourceLanguage': 'eng',
                'targetLanguage': samples[i]['targetLanguage'],
                'sourceText': output[i]
            } for i in range(len(samples))
        ]
        pivot_input_data = service.preprocess(pivot2tgt_samples)
        logger.info(
            f"Preprocessed a batch of size {n} ({n / (time.time() - start_time):.2f} samples / s)"
        )
        start_time = time.time()
        output = service.inference(pivot_input_data)
        logger.info(
            f"Infered a batch of size {n} ({n / (time.time() - start_time):.2f} samples / s)"
        )
    else:
        input_data = service.preprocess(samples)
        logger.info(
            f"Preprocessed a batch of size {n} ({n / (time.time() - start_time):.2f} samples / s)"
        )

        start_time = time.time()
        output = service.inference(input_data)
        logger.info(
            f"Infered a batch of size {n} ({n / (time.time() - start_time):.2f} samples / s)"
        )

    start_time = time.time()
    json_results = service.postprocess(output, samples)
    logger.info(
        f"Postprocessed a batch of size {n} ({n / (time.time() - start_time):.2f} samples / s)"
    )
    return json_results


def _lang_pair(src, tgt):
    return "{}-{}".format(ISO2SIMP[src], ISO2SIMP[tgt])


def handle(torchserve_data, context):
    if not _service.initialized:
        _service.initialize(context)
    if torchserve_data is None:
        return None

    start_time = time.time()
    all_samples = deserialize(torchserve_data)
    for i in range(len(all_samples)):
        all_samples[i]['inner_id'] = i

    n = len(all_samples)
    direct_all_samples = list(
        filter(lambda x: _lang_pair(x['sourceLanguage'], x['targetLanguage']) not in PIVOT_TABLE, all_samples))
    pivot_all_samples = list(
        filter(lambda x: _lang_pair(x['sourceLanguage'], x['targetLanguage']) in PIVOT_TABLE, all_samples))
    direct_n = len(direct_all_samples)
    pivot_n = len(pivot_all_samples)
    logger.info(
        f"Deserialized a batch of size {n} ({n / (time.time() - start_time):.2f} samples / s)"
    )
    # Adapt this to your model. The GPU has 16Gb of RAM.
    batch_size = 64

    results = [None for _ in range(len(all_samples))]
    # direct samples
    direct_samples = []
    for i, direct_sample in enumerate(direct_all_samples):
        direct_samples.append(direct_sample)
        if len(direct_samples) < batch_size and i + 1 < direct_n:
            continue
        translations = handle_mini_batch(_service, direct_samples)
        for k in range(len(translations)):
            results[direct_samples[k]['inner_id']] = translations[k]
        direct_samples = []

    # pivot samples
    pivot_samples = []
    for i, pivot_sample in enumerate(pivot_all_samples):
        pivot_samples.append(pivot_sample)
        if len(pivot_samples) < batch_size and i + 1 < pivot_n:
            continue
        translations = handle_mini_batch(_service, pivot_samples)
        for k in range(len(translations)):
            results[pivot_samples[k]['inner_id']] = translations[k]
        pivot_samples = []

    assert len(results)
    start_time = time.time()
    response = "\n".join(json.dumps(r, indent=None, ensure_ascii=False) for r in results)
    logger.info(
        f"Serialized a batch of size {n} ({n / (time.time() - start_time):.2f} samples / s)"
    )
    return [response]


def local_test():
    from dynalab.tasks import flores_small1, flores_small2

    bin_data = b"\n".join(json.dumps(d).encode("utf-8") for d in flores_small2.data)
    torchserve_data = [{"body": bin_data}]

    manifest = {"model": {"serializedFile": "small_track2.pt"}}
    system_properties = {"model_dir": ".", "gpu_id": 0}

    class Context(NamedTuple):
        system_properties: dict
        manifest: dict

    ctx = Context(system_properties, manifest)
    batch_responses = handle(torchserve_data, ctx)
    print(batch_responses)

    single_responses = [
        handle([{"body": json.dumps(d).encode("utf-8")}], ctx)[0]
        for d in flores_small2.data
    ]
    assert batch_responses == ["\n".join(single_responses)]


def read_from_file(input_file, sourceLanguage, targetLanguage):
    MAX_LINES = 16
    with open(input_file, "r", encoding="utf-8") as r:
        lines = r.readlines()[:MAX_LINES]
        data = [{'uid': str(i), 'sourceLanguage': sourceLanguage, 'targetLanguage': targetLanguage,
                 'sourceText': lines[i].strip()} for i in range(len(lines))]
    return data


def local_test_from_file_from_small_track1():
    # Read from the Files
    input_files = ["/path/to/SharedTask/data/thunder/flores101_dataset/devtest-code/valid.hr",
                   "/path/to/SharedTask/data/thunder/flores101_dataset/devtest-code/valid.sr"]
    sourceLanguages = ['hrv', 'srp']
    targetLanguages = ['srp', 'hrv']
    output_file = "/path/to/SharedTask/data/thunder/flores101_dataset/dynalab.translation"
    data = []
    for input_file, sourceLanguage, targetLanguage in zip(input_files, sourceLanguages, targetLanguages):
        data.extend(read_from_file(input_file, sourceLanguage, targetLanguage))
    bin_data = b"\n".join(json.dumps(d).encode("utf-8") for d in data)
    torchserve_data = [{"body": bin_data}]

    manifest = {"model": {"serializedFile": "small_track2.pt"}}
    system_properties = {"model_dir": ".", "gpu_id": 0}

    class Context(NamedTuple):
        system_properties: dict
        manifest: dict

    ctx = Context(system_properties, manifest)
    batch_responses = handle(torchserve_data, ctx)
    print(batch_responses)
    with open(output_file, "w", encoding="utf-8") as w:
        for response in batch_responses[0].split('\n'):
            w.write("{}\n".format(json.loads(response)['translatedText']))
    single_responses = [
        handle([{"body": json.dumps(d).encode("utf-8")}], ctx)[0]
        for d in data
    ]
    assert batch_responses == ["\n".join(single_responses)]


def local_test_from_file_small_track2():
    # Read from the Files
    input_files = ["/path/to/SharedTask/data/thunder/flores101_dataset/devtest-code/valid.id",
                   "/path/to/SharedTask/data/thunder/flores101_dataset/devtest-code/valid.en",
                   "/path/to/SharedTask/data/thunder/flores101_dataset/devtest-code/valid.id"]
    sourceLanguages = ['ind', 'eng', 'ind', 'tam']
    targetLanguages = ['tam', 'ind', 'eng', 'ind']
    output_file = "/path/to/SharedTask/data/thunder/flores101_dataset/dynalab.translation"
    data = []
    for input_file, sourceLanguage, targetLanguage in zip(input_files, sourceLanguages, targetLanguages):
        data.extend(read_from_file(input_file, sourceLanguage, targetLanguage))
    bin_data = b"\n".join(json.dumps(d).encode("utf-8") for d in data)
    torchserve_data = [{"body": bin_data}]

    manifest = {"model": {"serializedFile": "small_track2.pt"}}
    system_properties = {"model_dir": ".", "gpu_id": 0}

    class Context(NamedTuple):
        system_properties: dict
        manifest: dict

    ctx = Context(system_properties, manifest)
    batch_responses = handle(torchserve_data, ctx)
    print(batch_responses)
    with open(output_file, "w", encoding="utf-8") as w:
        for response in batch_responses[0].split('\n'):
            w.write("{}\n".format(json.loads(response)['translatedText']))
    single_responses = [
        handle([{"body": json.dumps(d).encode("utf-8")}], ctx)[0]
        for d in data
    ]
    print(single_responses)
    assert batch_responses == ["\n".join(single_responses)]


if __name__ == "__main__":
    # local_test_from_file()
    # local_test_from_file_small_track2()
    local_test()
