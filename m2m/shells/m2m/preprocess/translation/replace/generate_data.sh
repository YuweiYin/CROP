LANGS=(af ar bg bn de el es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo zh)
GENERATE_SCRIPT=./scripts/pattern/m2m/translate_train/prepare_replace_pattern_data.py
INPUT_DIR=/path/to/NER/xtreme_v0/
OUTPUT_DIR=/path/to/NER/pattern/replace_pattern/X/
PYTHON=/path/to/miniconda3/envs/amlt8/bin/python
mkdir -p $OUTPUT_DIR
for lg in ${LANGS[@]}; do
    echo "$INPUT -> $OUTPUT $ENTITY"
    INPUT=$INPUT_DIR/train-$lg.tsv
    OUTPUT=$OUTPUT_DIR/${lg}.txt0000
    ENTITY=$OUTPUT_DIR/${lg}.txt0001
    $PYTHON $GENERATE_SCRIPT -input $INPUT -output $OUTPUT -entity $ENTITY
    exit
done
