LANGS=(de es nl no)
GENERATE_SCRIPT=/path/to/xTune/m2m/scripts/pattern/m2m/ner/prepare_insert_pattern_data.py
INPUT_DIR=/path/to/NER/unitrans-data/xlmr_augment_v1/translation/NER/
OUTPUT_DIR=/path/to/NER/unitrans-data/xlmr_augment_v1/translation/LABELED_EN/
SPM_MODEL=/path/to/NER/PretrainedModels/flores/flores101_mm100_615M/sentencepiece.bpe.model
mkdir -p $OUTPUT_DIR $RAW_SENTENCES_DIR

for lg in ${LANGS[@]}; do
    INPUT=$INPUT_DIR/$lg/test.xlmr.tsv
    OUTPUT=$OUTPUT_DIR/${lg}.txt0000
    echo "${INPUT} -> ${OUTPUT} + ${ENTITY}"
    python $GENERATE_SCRIPT -input $INPUT -output $OUTPUT -entity "" -raw-sentences "" -lang "en" -sentencepiece-model $SPM_MODEL
done
