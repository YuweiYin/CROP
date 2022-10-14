LANGS=("de" "es" "nl" "no")

GENERATE_SCRIPT="/path/to/xTune/m2m/scripts/pattern/m2m/ner/prepare_ner_data.py"
SPM_MODEL="/path/to/m2m/spm.model"
INPUT_DIR="/path/to/xlmr_augment_v1/translation/BT/"
OUTPUT_DIR="/path/to/xlmr_augment_v1/translation/NER/"
mkdir -p ${OUTPUT_DIR}

for lg in ${LANGS[@]}; do
  INPUT="${INPUT_DIR}/${lg}0000.2en"
  OUTPUT="${OUTPUT_DIR}/${lg}/test.xlmr"
  IDX="$OUTPUT_DIR/${lg}/test.xlmr.idx"

  echo "${INPUT} -> ${OUTPUT} + ${IDX}"
  python ${GENERATE_SCRIPT} -input ${INPUT} -output ${OUTPUT} -idx ${IDX}
done
