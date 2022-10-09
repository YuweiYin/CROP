LANGS=(de en es nl no)
PYTHON=/path/to/miniconda3/envs/amlt8/bin/python
GENERATE_IDX_SCRIPT=/path/to/xTune/xTune/scripts/remove_DOCSTART_tag.py
INPUT_DIR=/path/to/NER/unitrans-data/unitrans/

for lg in ${LANGS[@]}; do
    for splt in dev test train; do
        OUTPUT_DIR=/path/to/NER/unitrans-data/xlmr/processed/${lg}/
        mkdir -p $OUTPUT_DIR
        INPUT=$INPUT_DIR/${splt}-${lg}.tsv
        OUTPUT=$OUTPUT_DIR/${splt}.xlmr
        echo "Copying ${INPUT} -> ${OUTPUT}"
        cp $INPUT $OUTPUT
        $PYTHON $GENERATE_IDX_SCRIPT -input $OUTPUT
    done
done
