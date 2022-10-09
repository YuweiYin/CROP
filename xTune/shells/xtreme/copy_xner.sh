INPUT_DIR=/path/to/NER/xtreme_v1/
OUTPUT_DIR=/path/to/NER/xtreme_v1/translation/X_NER/

PYTHON=/path/to/miniconda3/envs/amlt8/bin/python
GENERATE_SCRIPR=/path/to/xTune/m2m/scripts/pattern/m2m/translate_train/generate_ner_idx.py
LANGS=(af ar bg bn de el es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo zh)
for lg in ${LANGS[@]}; do
    INPUT=$INPUT_DIR/train-${lg}.tsv
    OUTPUT=$OUTPUT_DIR/$lg/test.xlmr
    mkdir -p $OUTPUT_DIR/$lg/
    echo "Copying $INPUT -> $OUTPUT"
    cp $INPUT $OUTPUT

    echo "Generating IDX FLLE"
    $PYTHON $GENERATE_SCRIPR -input $OUTPUT_DIR/$lg/test.xlmr
done
