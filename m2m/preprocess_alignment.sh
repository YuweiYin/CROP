RAW="/path/to/data/xlmr-data/valid-set/"
TRAIN="/path/to/data/xlmr-data/valid-set-spm/"
SPM_MODEL="/path/to/xlmr.base/sentencepiece.bpe.model"
src="en"

for tgt in "fr" "de" "fi" "cs" "et" "tr" "lv" "ro" "hi" "gu"; do
  echo "Start binarize ${TRAIN}/train.${src}-${tgt}..."
  spm_encode --model=${SPM_MODEL} --output_format=piece < ${RAW}/valid.${src}-${tgt}.${src} \
    > ${TRAIN}/valid.${src}-${tgt}.${src}
  spm_encode --model=${SPM_MODEL} --output_format=piece < ${RAW}/valid.${src}-${tgt}.${tgt} \
    > ${TRAIN}/valid.${src}-${tgt}.${tgt}
done
