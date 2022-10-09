INPUT_DIR=/path/to/NER/unitrans-data/xlmr_augment_v1/
OUTPUT_DIR=/path/to/NER/unitrans-data/xlmr_augment_v1/translation/X_NER/

GENERATE_SCRIPR=/path/to/xTune/m2m/scripts/pattern/m2m/ner/generate_ner_idx.py
LANGS=(de es nl no)
for lg in ${LANGS[@]}; do
    INPUT=$INPUT_DIR/train-${lg}.tsv
    OUTPUT=$OUTPUT_DIR/$lg/test.xlmr
    mkdir -p $OUTPUT_DIR/$lg/
    echo "Copying $INPUT -> $OUTPUT"
    cp $INPUT $OUTPUT

    echo "Generating IDX FLLE"
    python $GENERATE_SCRIPR -input $OUTPUT_DIR/$lg/test.xlmr
done