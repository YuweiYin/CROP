INPUT_DIR=/path/to/NER/xtreme/
OUTPUT_DIR=/path/to/NER/xtreme_v0/
LANGS=(ar he vi id jv ms tl eu ml ta te af nl en de el bn hi mr ur fa fr it pt es bg ru ja ka ko th sw yo my zh kk tr et fi hu)

CopySrc2Trg(){
    INPUT=$1
    OUTPUT=$2
    echo "Copying $INPUT -> $OUTPUT"
    cp $INPUT $OUTPUT
}

for lg in ${LANGS[@]}; do
    echo "Copying $INPUT_DIR/train-$lg.tsv -> $OUTPUT_DIR/train-$lg.tsv"
    cp $INPUT_DIR/train-$lg.tsv $OUTPUT_DIR/train-$lg.tsv
    if [ $lg = "en" ]; then
        splts=(dev test train) 
    else
        splts=(dev test)
    fi
    mkdir -p $OUTPUT_DIR/$lg
    for splt in ${splts[@]}; do
        INPUT=$INPUT_DIR/panx_processed_maxlen128/$lg/$splt.xlm-roberta-large
        OUTPUT=$OUTPUT_DIR/panx_processed_maxlen128/$lg/$splt.xlmr
        CopySrc2Trg $INPUT $OUTPUT
        
        INPUT=$INPUT_DIR/panx_processed_maxlen128/$lg/$splt.xlm-roberta-large.idx
        OUTPUT=$OUTPUT_DIR/panx_processed_maxlen128/$lg/$splt.xlmr.idx
        CopySrc2Trg $INPUT $OUTPUT
    done 
done
