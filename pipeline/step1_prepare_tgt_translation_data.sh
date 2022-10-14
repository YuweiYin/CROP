LANGS=("de" "es" "nl" "no")

GENERATE_SCRIPT="/path/to/xTune/m2m/scripts/pattern/m2m/ner/prepare_insert_pattern_data.py"
INPUT_DIR="/path/to/xlmr_augment_v1/"
OUTPUT_DIR="/path/to/xlmr_augment_v1/translation/X/"
mkdir -p ${OUTPUT_DIR}

for lg in ${LANGS[@]}; do
  INPUT="${INPUT_DIR}/train-${lg}.tsv"
  OUTPUT="${OUTPUT_DIR}/${lg}.txt0000"

  echo "Converting ${INPUT} -> ${OUTPUT}"
  python ${GENERATE_SCRIPT} -input ${INPUT} -raw-sentence ${OUTPUT} -output "" -entity ""
done
