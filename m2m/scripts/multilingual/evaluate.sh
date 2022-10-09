set -ex

data_bin=$1
langs=$2
lang_pairs=$3
src=$4
tgt=$5
model_file=$6
spm_model=$7
input_file=$8
reference_file=$9
log_dir=${10}

cat ${input_file} | python scripts/spm_encode.py --model ${spm_model} |
  python scripts/multilingual/truncate.py | python fairseq_cli/interactive.py \
    ${data_bin} \
    --path ${model_file} \
    --task "translation_multi_simple_epoch" \
    --encoder-langtok "tgt" \
    --langs ${langs} \
    --lang-pairs ${lang_pairs} \
    --source-lang ${src} --target-lang ${tgt} \
    --buffer-size 1024 --batch-size 64 --beam 5 --lenpen 1.0 \
    --remove-bpe=sentencepiece --no-progress-bar | grep -P "^H" | cut -f 3- >${log_dir}/translation.${src}-${tgt}

cat ${log_dir}/translation.$src-${tgt} | sacrebleu -l ${src}-${tgt} ${reference_file} 2>&1 |
  tee -a ${log_dir}/bleu.${src}-${tgt}
