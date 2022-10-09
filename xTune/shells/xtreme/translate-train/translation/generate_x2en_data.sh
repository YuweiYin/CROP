LANGS=(af ar bg bn de el es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo zh)
GENERATE_SCRIPT=/path/to/xTune/m2m/scripts/pattern/m2m/translate_train/prepare_translation_data.py
INPUT_DIR=/path/to/NER/xtreme_v0/translation/NER/
OUTPUT_DIR=/path/to/NER/xtreme_v0/pattern/insert_pattern/X2EN/
PYTHON=/path/to/miniconda3/envs/amlt8/bin/python
mkdir -p $OUTPUT_DIR $RAW_SENTENCES_DIR

for lg in ${LANGS[@]}; do
    INPUT=$INPUT_DIR/$lg/test.xlmr
    OUTPUT=$OUTPUT_DIR/$lg/en.txt0000
    ENTITY=$OUTPUT_DIR/$lg/
    echo "Copying ${INPUT} -> ${OUTPUT}"
    $PYTHON $GENERATE_SCRIPT -input $INPUT -output $OUTPUT -entity $ENTITY
done
