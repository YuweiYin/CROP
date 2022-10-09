LANGS=(af ar bg bn de el es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo zh)
GENERATE_SCRIPT=/path/to/xTune/m2m/scripts/pattern/MulDA/generate_MulDA_ner_data.py
ROOT=/path/to/NER/xtreme_MulDA/
NER=$ROOT/panx_processed_maxlen128/en/orig_data/train.xlmr
LABELED_EN=$ROOT/MulDA/En/
LABELED_X=$ROOT/MulDA/X/
OUTPUT_DIR=$ROOT/MulDA/FINAL/
PYTHON=/path/to/miniconda3/envs/amlt8/bin/python
TRAIN_DIR=$ROOT/panx_processed_maxlen128/
mkdir -p $OUTPUT_DIR
for lg in ${LANGS[@]}; do
    OUTPUT=$OUTPUT_DIR/train.${lg}.tsv
    IDX=$OUTPUT.idx
    echo "${NER} -> ${OUTPUT} + ${IDX}"
    $PYTHON $GENERATE_SCRIPT -ner $NER -source-pattern $LABELED_EN/en.txt0000 -translated-pattern $LABELED_X/en0000.2${lg} -translated-entities $LABELED_X/en0001.2${lg} -output $OUTPUT -idx $IDX -lang $lg
done
echo "Concating $TRAIN_DIR/en/orig_data/train.xlmr $OUTPUT_DIR/train.*.tsv -> $TRAIN_DIR/en/train.xlmr"
CONCAT_SCRIPT=/path/to/xTune/m2m/scripts/pattern/concat_ner_dataset.py
$PYTHON $CONCAT_SCRIPT -source-dataset $TRAIN_DIR/en/orig_data/train.xlmr -target-dataset $OUTPUT_DIR  -output $TRAIN_DIR/en/train.xlmr -idx $TRAIN_DIR/en/train.xlmr.idx -sampling-method "down-sampling"
#cat $TRAIN_DIR/en/orig_data/train.xlmr $OUTPUT_DIR/train.*.tsv > $TRAIN_DIR/en/train.xlmr
#echo "Generating IDX FILE -> /path/to/NER/xtreme_v1/panx_processed_maxlen128/en/train.xlmr.idx"
#GENERATE_IDX_SCRIPTS=/path/to/xTune/m2m/scripts/pattern/m2m/ner/generate_ner_idx.py
#$PYTHON $GENERATE_IDX_SCRIPTS -input $TRAIN_DIR/en/train.xlmr