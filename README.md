# CROP: Zero-shot Cross-lingual Named Entity Recognition with Multilingual Labeled Sequence Translation

![picture](https://yuweiyin.github.io/files/publications/2022-12-09-EMNLP-CROP.png)

## Abstract

Named entity recognition (NER) suffers from
the scarcity of annotated training data, especially
for low-resource languages without
labeled data. Cross-lingual NER has been
proposed to alleviate this issue by transferring
knowledge from high-resource languages
to low-resource languages via aligned crosslingual
representations or machine translation
results. However, the performance of crosslingual
NER methods is severely affected by
the unsatisfactory quality of translation or label
projection. To address these problems,
we propose a **Cro**ss-lingual Entity **P**rojection
framework (**CROP**) to enable zero-shot crosslingual
NER with the help of a multilingual labeled
sequence translation model. Specifically,
the target sequence is first translated into the
source language and then tagged by a source
NER model. We further adopt a labeled sequence
translation model to project the tagged
sequence back to the target language and label
the target raw sentence. Ultimately, the whole
pipeline is integrated into an end-to-end model
by the way of self-training. Experimental results
on two benchmarks demonstrate that our
method substantially outperforms the previous
strong baseline by a large margin of +3 ~ 7
F1 scores and achieves state-of-the-art performance.


## Data

We use **CCaligned**, **CoNLL-5**, and **XTREME-40** datasets.
For more details, please refer to the **4.1 Dataset** Section in our paper.

<!-- ### Preprocessing -->

<!-- ### Post-processing -->


## Environment

* Python: >= 3.6
* [PyTorch](http://pytorch.org/): >= 1.5.0
* NVIDIA GPUs and [NCCL](https://github.com/NVIDIA/nccl)
* [Fairseq](https://github.com/pytorch/fairseq): 1.0.0

```bash
cd m2m/fairseq
pip install --editable ./
```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

## CROP Training

### I. The Pipeline for Cross-lingual NER Tasks

**NOTE**: modify all the `"/path/to/"` in our code to your own code/data path.

1. Translated Target Translation data

```bash
bash /path/to/pipeline/step1_prepare_tgt_translation_data.sh
```

2. Translated Target data to the Source data 

```bash
bash /path/to/pipeline/step2_tgt2src_translation.sh
```

3. Prepare Translated NER Data

```bash
bash /path/to/pipeline/step3_preapre_src_ner_data.sh
```

4. Source NER

```bash
bash /path/to/pipeline/step4_src_ner.sh
```

5. Prepare Source Translation Data

```bash
bash /path/to/pipeline/step5_prepare_src_translation_data.sh
```

6. Labeled Translation

```bash
bash /path/to/pipeline/step6_labeled_transation.sh
```

7. Prepare and Filter the multilingual NER Data

```bash
bash /path/to/pipeline/step7_prepare_pseudo_ner_data1.sh
bash /path/to/pipeline/step7_prepare_pseudo_ner_data2.sh
```

### II. The Multilingaul Labeled Sequence Translation Model

```bash
NODES=8
GPUS=8
LR=1e-4
UPDATE_FREQ=16
MAX_TOKENS=1024
MAX_EPOCH=100
MAX_UPDATES=400000
SUBWORD_PROB=0.0
SLOT_METHOD="insert"
SLOT_PROB=0.85
MAX_SLOT_NUM=14

TEXT="/path/to/data-bin-1/:/path/to/data-bin-2/"  # dataset directories, seperated by colons ":"
PRETRAINED_MODEL="/path/to/pretrained_model.pt"
SPM_MODEL="/path/to/m2m/spm.model"

LANGS="af,am,ar,as,ast,ay,az,ba,be,bg,bn,br,bs,ca,ceb,cjk,cs,cy,da,de,dyu,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kac,kam,kea,kg,kk,km,kmb,kmr,kn,ko,ku,ky,lb,lg,ln,lo,lt,luo,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,no,ns,ny,oc,om,or,pa,pl,ps,pt,qu,ro,ru,sd,shn,si,sk,sl,sn,so,sq,sr,ss,su,sv,sw,ta,te,tg,th,ti,tl,tn,tr,uk,umb,ur,uz,vi,wo,xh,yi,yo,zh,zu,eu"
LANG_PAIRS="en-af,af-en,en-ar,ar-en,en-bg,bg-en,en-bn,bn-en,en-de,de-en,en-el,el-en,en-es,es-en,en-et,et-en,en-eu,eu-en,en-fa,fa-en,en-fi,fi-en,en-fr,fr-en,en-he,he-en,en-hi,hi-en,en-hu,hu-en,en-id,id-en,en-it,it-en,en-ja,ja-en,en-jv,jv-en,en-ka,ka-en,en-kk,kk-en,en-ko,ko-en,en-ml,ml-en,en-mr,mr-en,en-ms,ms-en,en-my,my-en,en-nl,nl-en,en-no,no-en,en-pt,pt-en,en-ru,ru-en,en-sw,sw-en,en-ta,ta-en,en-te,te-en,en-th,th-en,en-tl,tl-en,en-tr,tr-en,en-ur,ur-en,en-vi,vi-en,en-yo,yo-en,en-zh,zh-en"

SAVE_DIR="/path/to/save_dir/"
mkdir -p ${SAVE_DIR}

echo "Start Training ..."
if [ ! -f ${SAVE_DIR}/checkpoint_last.pt ]; then
  python -m torch.distributed.launch --nproc_per_node=${GPUS} --nnodes=${NODES} \
    --node_rank=${OMPI_COMM_WORLD_RANK} --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} train.py ${TEXT} \
    --save-dir ${SAVE_DIR} --restore-file ${PRETRAINED_MODEL} --arch "transformer_wmt_en_de_big" \
    --encoder-normalize-before --decoder-normalize-before --encoder-layers 12 --decoder-layers 12 \
    --task translation_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 \
    --min-sampling-temperature 1.0 --warmup-epoch 5 \
    --encoder-langtok "src" --langtoks '{"main":("src","tgt")}' \
    --langs ${LANGS} --lang-pairs ${LANG_PAIRS} \
    --truncate-source --enable-reservsed-directions-shared-datasets --load-alignments \
    --share-all-embeddings --max-source-positions 1024 --max-target-positions 1024 \
    --criterion "label_smoothed_cross_entropy" --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr ${LR} --warmup-init-lr 1e-07 --stop-min-lr 1e-09 \
    --warmup-updates 4000 --max-update ${MAX_UPDATES} --max-epoch ${MAX_EPOCH} \
    --attention-dropout 0.1 --dropout 0.1 --max-tokens ${MAX_TOKENS} --update-freq ${UPDATE_FREQ} \
    --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test --fp16 \
    --bpe sentencepiece --sentencepiece-model ${SPM_MODEL} --subword-prob ${SUBWORD_PROB} \
    --max-slot-num ${MAX_SLOT_NUM} --slot-prob ${SLOT_PROB} --slot-method ${SLOT_METHOD} \
    --reset-optimizer --reset-meters --reset-lr-scheduler --reset-dataloader \
    --ddp-backend=no_c10d 2>&1 | tee -a ${SAVE_DIR}/train.log
else
  python -m torch.distributed.launch --nproc_per_node=${GPUS} --nnodes=${NODES} \
    --node_rank=${OMPI_COMM_WORLD_RANK} --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} train.py ${TEXT} \
    --save-dir ${SAVE_DIR} --arch "transformer_wmt_en_de_big" \
    --encoder-normalize-before --decoder-normalize-before --encoder-layers 12 --decoder-layers 12 \
    --task translation_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 \
    --min-sampling-temperature 1.0 --warmup-epoch 5 \
    --encoder-langtok "src" --langtoks '{"main":("src","tgt")}' \
    --langs ${LANGS} --lang-pairs ${LANG_PAIRS} \
    --truncate-source --enable-reservsed-directions-shared-datasets --load-alignments \
    --share-all-embeddings --max-source-positions 1024 --max-target-positions 1024 \
    --criterion "label_smoothed_cross_entropy" --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr ${LR} --warmup-init-lr 1e-07 --stop-min-lr 1e-09 \
    --warmup-updates 4000 --max-update ${MAX_UPDATES} --max-epoch ${MAX_EPOCH} \
    --attention-dropout 0.1 --dropout 0.1 --max-tokens ${MAX_TOKENS} --update-freq ${UPDATE_FREQ} \
    --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test --fp16 \
    --bpe sentencepiece --sentencepiece-model ${SPM_MODEL} --subword-prob ${SUBWORD_PROB} \
    --max-slot-num ${MAX_SLOT_NUM} --slot-prob ${SLOT_PROB} --slot-method ${SLOT_METHOD} \
    --ddp-backend=no_c10d 2>&1 | tee -a ${SAVE_DIR}/train.log
fi
```

### III. The Source NER Model

```bash
SEED=1
EPOCH=20
MAX_LENGTH=128
EVALUATE_STEPS=1000

DATA_DIR="/path/to/xlmr/data-bin/"
PRETRAINED_ENCODER="xlm-roberta-base"
PRETRAINED_ENCODER_PATH="/path/to/pretrained-model/${PRETRAINED_ENCODER}"

if [ ${PRETRAINED_ENCODER} == "xlm-roberta-large" ]; then
    BATCH_SIZE=32
    GRAD_ACC=1
    LR=7e-6
else
    BATCH_SIZE=32
    GRAD_ACC=1
    LR=1e-5
fi

LANGS="de,en,es,nl,no"
PREDICT_LANGS="de,es,nl,no"

OUTPUT_DIR=/path/to/output_dir/
mkdir -p ${OUTPUT_DIR}

python xTune/src/run_tag.py --model_type xlmr --model_name_or_path ${PRETRAINED_ENCODER_PATH} \
  --do_train --do_eval --do_predict --do_predict_dev --predict_langs ${PREDICT_LANGS} \
  --train_langs en --data_dir ${DATA_DIR} --labels ${DATA_DIR}/labels.txt \
  --per_gpu_train_batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${GRAD_ACC} \
  --per_gpu_eval_batch_size 256 --learning_rate ${LR} --num_train_epochs ${EPOCH} \
  --max_seq_length ${MAX_LENGTH} --noised_max_seq_length ${MAX_LENGTH} \
  --output_dir ${OUTPUT_DIR} --overwrite_output_dir --evaluate_during_training --logging_steps 50 \
  --evaluate_steps $EVALUATE_STEPS --seed ${SEED} --warmup_steps -1 \
  --save_only_best_checkpoint --eval_all_checkpoints --eval_patience -1 \
  --fp16 --fp16_opt_level O2 --hidden_dropout_prob 0.1 \
  --original_loss 2>&1 | tee -a ${OUTPUT_DIR}/log.txt
```


<!-- ## Inference & Evaluation -->


<!-- ## Experiments -->


## Citation

* arXiv: https://arxiv.org/abs/2210.07022
<!-- * ACL Anthology: https://aclanthology.org/ -->

```bibtex
@inproceedings{crop,
  title     = {CROP: Zero-shot Cross-lingual Named Entity Recognition with Multilingual Labeled Sequence Translation},
  author    = {Yang, Jian and Huang, Shaohan and Ma, Shuming and Yin, Yuwei and Dong, Li and Zhang, Dongdong and Guo, Hongcheng and Li, Zhoujun and Wei, Furu},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2022},
  year      = {2022},
}
```

## License

Please refer to the [LICENSE](./LICENSE) file for more details.


## Contact

If there is any question, feel free to create a GitHub issue or contact us by [Email](mailto:seckexyin@gmail.com).
