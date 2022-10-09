LANGS=(af ar bg bn de el en es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo zh)
ROOT=/path/to/NER/xtreme_v1/
ANALYSIS=/path/to/NER/xtreme_analysis/data_size/
FINAL_DIR=/path/to/NER/xtreme_v1/translation/FINAL/
GENERATE_IDX_SCRIPTS=/path/to/xTune/m2m/scripts/pattern/m2m/ner/generate_ner_idx.py
SAMPLING_SCRIPT=/path/to/xTune/m2m/scripts/pattern/sampling_ner_dataset.py
mkdir -p $ANALYSIS
PYTHON=/path/to/miniconda3/envs/amlt8/bin/python
MIN_SIZE=1000
MAX_SIZE=10000
STEP=1000
CreateDir(){
    echo "Copying $ROOT -> $ANALYSIS/$1/"
    mkdir -p $ANALYSIS/$1/   
    for lg in ${LANGS[@]}; do
        cp -r $ROOT/panx_processed_maxlen128/$lg/ $ANALYSIS/$1/panx_processed_maxlen128/
    done
    cp $ROOT/panx_processed_maxlen128/labels.txt $ANALYSIS/$1/panx_processed_maxlen128/
    echo "Random Sampling $1 sentences"
    cat $FINAL_DIR/train.*.tsv > $ANALYSIS/train.all.tsv
    $PYTHON $SAMPLING_SCRIPT -input $ANALYSIS/train.all.tsv -output $ANALYSIS/$1/panx_processed_maxlen128/en/train.xlmr -max-sentences $1
    cat $ROOT/panx_processed_maxlen128/en/orig_data/train.xlmr >> $ANALYSIS/$1/panx_processed_maxlen128/en/train.xlmr
    echo "Generating IDX FILE -> $ANALYSIS/$1/panx_processed_maxlen128/en/train.xlmr.idx"
    $PYTHON $GENERATE_IDX_SCRIPTS -input $ANALYSIS/$1/panx_processed_maxlen128/en/train.xlmr
}
for((SIZE=$MIN_SIZE;SIZE<=$MAX_SIZE;SIZE+=$STEP)); do
    CreateDir $SIZE
done
