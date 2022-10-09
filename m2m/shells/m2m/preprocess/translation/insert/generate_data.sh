LANGS=(af ar bg bn de el es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo zh)
GENERATE_SCRIPT=/path/to/xTune/m2m/scripts/pattern/m2m/translate_train/prepare_insert_pattern_data.py
INPUT_DIR=/path/to/NER/xtreme_v0/
OUTPUT_DIR=/path/to/NER/xtreme_v0/pattern/insert_pattern/X/
RAW_SENTENCES_DIR=/path/to/NER/xtreme_v0/translation/X/
PYTHON=/path/to/miniconda3/envs/amlt8/bin/python
mkdir -p $OUTPUT_DIR $RAW_SENTENCES_DIR
for lg in ${LANGS[@]}; do
    INPUT=$INPUT_DIR/train-$lg.tsv
    OUTPUT=$OUTPUT_DIR/${lg}.txt0000
    ENTITY=$OUTPUT_DIR/${lg}.txt0001
    echo "${INPUT} -> ${OUTPUT} + ${ENTITY}"
    $PYTHON $GENERATE_SCRIPT -input $INPUT -output $OUTPUT -entity $ENTITY -raw-sentences $RAW_SENTENCES_DIR/$lg.txt0000
done
