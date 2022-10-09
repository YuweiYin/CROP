LANGS=(af ar bg bn de el es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo zh)
GENERATE_SCRIPT1=/path/to/xTune/m2m/scripts/pattern/translate_train/prepare_translate_train_ner_pseudo_data.py
ROOT=/path/to/NER/xtreme_translate_train/
EN_INPUT_DIR=$ROOT/translate_train/LABELED_EN/
INPUT_DIR=$ROOT/translate_train/LABELED_X/
NER_DIR=$ROOT/translate_train/NER/
OUTPUT_DIR=$ROOT/translate_train/LABELED_X_NER/
PYTHON=/path/to/miniconda3/envs/amlt8/bin/python
mkdir -p $OUTPUT_DIR

for lg in ${LANGS[@]}; do
    EN_INPUT=$EN_INPUT_DIR/en.txt0000
    NER=$NER_DIR/en/test.xlmr.tsv
    INPUT=$INPUT_DIR/en0000.2${lg}
    OUTPUT=$OUTPUT_DIR/train.${lg}.tsv
    IDX=$OUTPUT_DIR/train.${lg}.tsv.idx
    echo "${INPUT} ${NER} -> ${OUTPUT} + ${IDX}"
    $PYTHON $GENERATE_SCRIPT1 -en-input $EN_INPUT -input $INPUT -ner $NER -output $OUTPUT -idx $IDX -lang $lg
done

echo "Concating $ROOT/panx_processed_maxlen128/en/orig_data/train.xlmr $FINAL_DIR/train.*.tsv -> $ROOT/panx_processed_maxlen128/en/train.xlmr"
cat $ROOT/panx_processed_maxlen128/en/orig_data/train.xlmr $OUTPUT_DIR/train.*.tsv > $ROOT/panx_processed_maxlen128/en/train.xlmr
echo "Generating IDX FILE -> /$ROOT/panx_processed_maxlen128/en/train.xlmr.idx"
GENERATE_IDX_SCRIPTS=/path/to/xTune/m2m/scripts/pattern/m2m/ner/generate_ner_idx.py
$PYTHON $GENERATE_IDX_SCRIPTS -input $ROOT/panx_processed_maxlen128/en/train.xlmr
