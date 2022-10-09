LANGS=(af ar bg bn de el es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo zh)
GENERATE_SCRIPT1=/path/to/xTune/m2m/scripts/pattern/m2m/ner/prepare_ner_data_from_pattern_sentence_step1.py
GENERATE_SCRIPT2=/path/to/xTune/m2m/scripts/pattern/m2m/ner/prepare_ner_data_from_pattern_sentence_step2.py
ROOT=/path/to/NER/xtreme_v1/
EN_INPUT_DIR=$ROOT/translation/LABELED_EN/
INPUT_DIR=$ROOT/translation/LABELED_X/
GROUNDTRUTH_DIR=$ROOT
NER_DIR=$ROOT/translation/NER/
X_NER_DIR=$ROOT/translation/X_NER/
OUTPUT_DIR=$ROOT/translation/LABELED_X_NER/
LOG_DIR=/path/to/xTune/data/log/
PYTHON=/path/to/miniconda3/envs/amlt8/bin/python
STATE="STEP12" #STEP1 STEP2 STEP3 ALL
mkdir -p $OUTPUT_DIR
for lg in ${LANGS[@]}; do
    EN_INPUT=$EN_INPUT_DIR/${lg}.txt0000
    INPUT=$INPUT_DIR/en0000.2${lg}
    NER=$NER_DIR/${lg}/test.xlmr.tsv
    X_NER=$X_NER_DIR/${lg}/test.xlmr.tsv
    OUTPUT=$OUTPUT_DIR/train.${lg}.tsv
    FINAL_DIR=/path/to/NER/xtreme_v1/translation/FINAL/
    IDX=$OUTPUT_DIR/train.${lg}.tsv.idx
    IDX2=/path/to/NER/xtreme_v1/translation/FINAL/train.${lg}.idx
    echo "${INPUT} ${NER} -> ${OUTPUT} + ${IDX}"
    if [ $STATE = "STEP1" -o $STATE = "ALL" -o $STATE = "STEP12" ]; then 
        $PYTHON $GENERATE_SCRIPT1 -en-input $EN_INPUT -input $INPUT -ner $NER -output $OUTPUT -idx $IDX -lang $lg
    fi
    if [ $STATE = "STEP2" -o $STATE = "ALL" -o $STATE = "STEP12" ]; then 
        $PYTHON $GENERATE_SCRIPT2 -x-ner $X_NER -groundtruth-ner $GROUNDTRUTH_DIR/train-${lg}.tsv -translated-ner $OUTPUT -output $FINAL_DIR/train.${lg}.tsv -idx $IDX2 -log $LOG_DIR/train.${lg}.tsv.log -lang $lg
    fi
done
if [ $STATE = "STEP3" -o $STATE = "ALL" ]; then 
    echo "Concating $ROOT/panx_processed_maxlen128/en/orig_data/train.xlmr $FINAL_DIR/train.*.tsv -> $ROOT/panx_processed_maxlen128/en/train.xlmr"
    cat $ROOT/panx_processed_maxlen128/en/orig_data/train.xlmr $FINAL_DIR/train.*.tsv > $ROOT/panx_processed_maxlen128/en/train.xlmr
    echo "Generating IDX FILE -> $ROOT/panx_processed_maxlen128/en/train.xlmr.idx"
    GENERATE_IDX_SCRIPTS=/path/to/xTune/m2m/scripts/pattern/m2m/translate_train/generate_ner_idx.py
    $PYTHON $GENERATE_IDX_SCRIPTS -input $ROOT/panx_processed_maxlen128/en/train.xlmr
fi