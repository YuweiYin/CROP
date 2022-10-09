LANGS=(de es nl no)
GENERATE_SCRIPT1=/path/to/xTune/m2m/scripts/pattern/unitrans-data/our/prepare_ner_data_from_pattern_sentence_step1.py
GENERATE_SCRIPT2=/path/to/xTune/m2m/scripts/pattern/unitrans-data/our/prepare_ner_data_from_pattern_sentence_step2.py
ROOT=/path/to/NER/unitrans-data/xlmr_augment_v1/
EN_INPUT_DIR=${ROOT}/translation/LABELED_EN/
INPUT_DIR=${ROOT}/translation/LABELED_X/
GROUNDTRUTH_DIR=${ROOT}
NER_DIR=${ROOT}/translation/NER/
X_NER_DIR=${ROOT}/translation/X_NER/
OUTPUT_DIR=${ROOT}/translation/LABELED_X_NER/
FINAL_DIR=${ROOT}/translation/FINAL/
LOG_DIR=/path/to/xTune/data/log/
PYTHON=/path/to/miniconda3/envs/amlt8/bin/python
beam_size=1
ITER=2 # 1 2 3 ... n
mkdir -p $OUTPUT_DIR

for lg in ${LANGS[@]}; do
    EN_INPUT=${EN_INPUT}_DIR/${lg}.txt0000
    INPUT=${INPUT}_DIR/en0000.2${lg}
    NER=${NER}_DIR/${lg}/test.xlmr.tsv
    OUTPUT=$OUTPUT_DIR/train.${lg}.tsv
    IDX=$OUTPUT_DIR/train.${lg}.tsv.idx
    # IDX2=/path/to/NER/unitrans-data/xlmr_augment_v1/translation/FINAL/train.${lg}.idx
    echo "${INPUT} ${NER} -> ${OUTPUT} + ${IDX}"
    $PYTHON ${GENERATE_SCRIPT1} -en-input ${EN_INPUT} -input ${INPUT} -ner ${NER} -output $OUTPUT -idx ${IDX} -lang ${lg} -beam-size ${beam_size}
    if [ "$ITER" == "1" ]; then
        X_NER="None"
    else
        X_NER=$X_NER_DIR/${lg}/test.xlmr.tsv
    fi
    $PYTHON ${GENERATE_SCRIPT2} -x-ner $X_NER -groundtruth-ner $GROUNDTRUTH_DIR/train-${lg}.tsv -translated-ner $OUTPUT -output ${FINAL_DIR}/train.${lg}.tsv -idx ${IDX}2 -log $LOG_DIR/train.${lg}.tsv.log -lang ${lg} -beam-size ${beam_size}
done

CONCAT_SCRIPT=/path/to/xTune/m2m/scripts/pattern/concat_ner_dataset.py
$PYTHON $CONCAT_SCRIPT -source-dataset ${ROOT}/processed/en/orig_data/train.xlmr -target-dataset ${FINAL_DIR} -output ${ROOT}/processed/en/train.xlmr -idx ${ROOT}/processed/en/train.xlmr.idx -sampling-method "None"
