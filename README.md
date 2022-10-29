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
cd m2m
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

**NOTE**: modify all the `"/path/to/"` in our code to your own code/data path.

### Train the NER and Translation models

**The Source NER Model**

```bash
bash /path/to/CROP/pipeline/step0_train_source_ner_model.sh
```

**The Multilingual Labeled Sequence Translation Model**

```bash
bash /path/to/CROP/pipeline/step0_train_translation_model.sh
```

- [Clound Storage](https://pan.baidu.com/s/1YQjJEIVevEHXk-wpxcA8wg?pwd=jp4b)
  - Trained model
    - Trained Baseline Translation Model: `m2m_checkpoint_baseline.pt`
    - Trained Insert-based Translation Model: `m2m_checkpoint_insert_avg_41_60.pt`
    - Trained Replace-based Translation Model: `m2m_checkpoint_replace_avg_11_20.pt`
  - Dictionary for Tokenization (used by all three models above): `dict.txt`
  - SentencePiece Model: `spm.model`

### CROP Pipeline

1. Translated Target Translation data

```bash
bash /path/to/CROP/pipeline/step1_prepare_tgt_translation_data.sh
```

2. Translated Target data to the Source data

(use the Baseline Translation Model or Insert-based Translation Model or Replace-based Translation Model)

```bash
bash /path/to/CROP/pipeline/step2_tgt2src_translation.sh
```

3. Prepare Translated NER Data

```bash
bash /path/to/CROP/pipeline/step3_preapre_src_ner_data.sh
```

4. Source NER

```bash
bash /path/to/CROP/pipeline/step4_src_ner.sh
```

5. Prepare Source Translation Data

```bash
bash /path/to/CROP/pipeline/step5_prepare_src_translation_data.sh
```

6. Labeled Translation

(use the Insert-based Translation Model)

```bash
bash /path/to/CROP/pipeline/step6_labeled_transation.sh
```

7. Prepare and Filter the multilingual NER Data

```bash
bash /path/to/CROP/pipeline/step7_prepare_pseudo_ner_data1.sh
bash /path/to/CROP/pipeline/step7_prepare_pseudo_ner_data2.sh
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
