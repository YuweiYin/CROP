set -ex

data_bin=$1
model_file=$2
spm_model=$3
input_file=$4
reference_file=$5
log_dir=$6
cuda_device=$7
src=$8
tgt=$9
extra_args=${10}

cat ${input_file} | python scripts/spm_encode.py --model ${spm_model} |
  python scripts/multilingual/truncate.py | CUDA_VISIBLE_DEVICES=${cuda_device} python fairseq_cli/interactive.py \
    ${data_bin} \
    --path ${model_file} \
    --buffer-size 1024 --batch-size 64 --beam 5 --lenpen 1.0 \
    --remove-bpe=sentencepiece --no-progress-bar \
    ${extra_args} | grep -P "^H" | cut -f 3- >$log_dir/translation.${src}-${tgt}

cat ${log_dir}/translation.${src}-${tgt} | sacrebleu -l ${src}-${tgt} ${reference_file} 2>&1 | \
  tee -a ${log_dir}/bleu.${src}-${tgt}
