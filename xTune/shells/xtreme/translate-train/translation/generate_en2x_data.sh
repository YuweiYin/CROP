LANGS=(en)
GENERATE_SCRIPT=/path/to/xTune/m2m/scripts/pattern/m2m/ner/prepare_insert_pattern_data.py
INPUT_DIR=/path/to/NER/xtreme_v1/translate-train/NER/
OUTPUT_DIR=/path/to/NER/xtreme_v1/translate-train/LABELED_EN/
PYTHON=/path/to/miniconda3/envs/amlt8/bin/python
mkdir -p $OUTPUT_DIR $RAW_SENTENCES_DIR

for lg in ${LANGS[@]}; do
    INPUT=$INPUT_DIR/$lg/test.xlmr.tsv
    OUTPUT=$OUTPUT_DIR/${lg}.txt0000
    echo "${INPUT} -> ${OUTPUT} + ${ENTITY}"
    $PYTHON $GENERATE_SCRIPT -input $INPUT -output $OUTPUT -entity "" -raw-sentences "" -lang "en"
done
