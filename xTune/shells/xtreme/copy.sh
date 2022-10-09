INPUT_DIR=/path/to/NER/xtreme/panx_processed_maxlen128/
#af ar bg bn de el es et fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo zh
LANGS=(en)

for lg in ${LANGS[@]}; do
    INPUT=$INPUT_DIR/$lg/train.xlm-roberta-large
    OUTPUT=$INPUT_DIR/$lg/train.xlm-roberta-base
    echo "Copying $INPUT -> $OUTPUT"
    cp $INPUT $OUTPUT
    
    INPUT=$INPUT_DIR/$lg/train.xlm-roberta-large.idx
    OUTPUT=$INPUT_DIR/$lg/train.xlm-roberta-base.idx
    echo "Copying $INPUT -> $OUTPUT"
    cp $INPUT $OUTPUT
done
