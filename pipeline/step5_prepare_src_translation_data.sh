LANGS=("de" "es" "nl" "no")

GENERATE_SCRIPT="/path/to/xTune/m2m/scripts/pattern/m2m/ner/prepare_insert_pattern_data.py"
SPM_MODEL="/path/to/m2m/spm.model"
INPUT_DIR="/path/to/xlmr_augment_v1/translation/NER/"
OUTPUT_DIR="/path/to/xlmr_augment_v1/translation/LABELED_EN/"
mkdir -p ${OUTPUT_DIR}

for lg in ${LANGS[@]}; do
  INPUT="${INPUT_DIR}/$lg/test.xlmr.tsv"
  OUTPUT="${OUTPUT_DIR}/${lg}.txt0000"

  echo "${INPUT} -> ${OUTPUT} + ${ENTITY}"
  python ${GENERATE_SCRIPT} -input ${INPUT} -output ${OUTPUT} \
    -entity "" -raw-sentences "" -lang "en" -sentencepiece-model ${SPM_MODEL}
done
