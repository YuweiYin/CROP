LANGS=(de es nl no)
GENERATE_SCRIPT=/path/to/xTune/m2m/scripts/pattern/m2m/ner/prepare_ner_data.py
INPUT_DIR=/path/to/NER/unitrans-data/xlmr_augment_v1/translation/BT/
OUTPUT_DIR=/path/to/NER/unitrans-data/xlmr_augment_v1/translation/NER/
PYTHON=/path/to/miniconda3/envs/amlt8/bin/python
SPM_MODEL=/path/to/NER/PretrainedModels/flores/flores101_mm100_615M/sentencepiece.bpe.model
mkdir -p $OUTPUT_DIR
for lg in ${LANGS[@]}; do
    INPUT=$INPUT_DIR/${lg}0000.2en
    OUTPUT=$OUTPUT_DIR/${lg}/test.xlmr
    IDX=$OUTPUT_DIR/${lg}/test.xlmr.idx
    echo "${INPUT} -> ${OUTPUT} + ${IDX}"
    $PYTHON $GENERATE_SCRIPT -input $INPUT -output $OUTPUT -idx $IDX
done
