LANGS=(af ar bg bn de el es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo zh)
GENERATE_SCRIPT=/path/to/xTune/m2m/scripts/pattern/UniTrans/generate_pseudo_ner_data.py
ROOT=/path/to/NER/xtreme_UniTrans/
NER=$ROOT/UniTrans/NER/en/test.xlmr.tsv
X_NER_DIR=$ROOT/UniTrans/X/
OUTPUT_DIR=$ROOT/UniTrans/FINAL/
PYTHON=/path/to/miniconda3/envs/amlt8/bin/python
TRAIN_DIR=$ROOT/panx_processed_maxlen128/
mkdir -p $OUTPUT_DIR
for lg in ${LANGS[@]}; do
    X_NER=$X_NER_DIR/en0000.2${lg}
    OUTPUT=$OUTPUT_DIR/train.${lg}.tsv
    IDX=$OUTPUT.idx
    echo "${NER} -> ${OUTPUT} + ${IDX}"
    $PYTHON $GENERATE_SCRIPT -ner $NER -translated-ner $X_NER -output $OUTPUT -idx $IDX -lang $lg
done
echo "Concating $TRAIN_DIR/en/orig_data/train.xlmr $OUTPUT_DIR/train.*.tsv -> $TRAIN_DIR/en/train.xlmr"
cat $TRAIN_DIR/en/orig_data/train.xlmr $OUTPUT_DIR/train.*.tsv > $TRAIN_DIR/en/train.xlmr
echo "Generating IDX FILE -> /path/to/NER/xtreme_v1/panx_processed_maxlen128/en/train.xlmr.idx"
GENERATE_IDX_SCRIPTS=/path/to/xTune/m2m/scripts/pattern/m2m/ner/generate_ner_idx.py
$PYTHON $GENERATE_IDX_SCRIPTS -input $TRAIN_DIR/en/train.xlmr