export PATH=$PATH:~/eflomal/bin/
INPUT_DIR=/path/to/NER/m2m/train-20M/
OUTPUT_DIR=/path/to/NER/m2m/align-20M/
ALIGN=./scripts/align.py
MAKE_PRIORS=./scripts/makepriors.py
PREPROCESS_ALIGN=./scripts/pattern/preprocess_alignments.py
STATE="ALL" #ALL CONCAT ALIGN
mkdir -p $OUTPUT_DIR
lg=$1
if [ $STATE = "ALL" -o $STATE = "CONCAT" ]; then 
    echo "Concating $INPUT_DIR/train.en-${lg}.en $INPUT_DIR/train.en-${lg}.${lg} -> $OUTPUT_DIR/train.en-${lg}"
    paste -d " ||| " $INPUT_DIR/train.en-${lg}.en /dev/null /dev/null /dev/null /dev/null /dev/null $INPUT_DIR/train.en-${lg}.${lg} > $OUTPUT_DIR/train.en-${lg}
fi

if [ $STATE = "ALL" -o $STATE = "ALIGN" ]; then
    rm $OUTPUT_DIR/train.align.en-${lg}.*
    echo "Aligning $OUTPUT_DIR/align.en-${lg} -> $OUTPUT_DIR/train.align.en-${lg}"
    python $ALIGN -i $OUTPUT_DIR/train.en-${lg} --model 3 -f $OUTPUT_DIR/train.align.en-${lg}.fwd -r $OUTPUT_DIR/train.align.en-${lg}.rev
    ~/fast_align/build/atools -c grow-diag-final-and -i $OUTPUT_DIR/train.align.en-${lg}.fwd -j $OUTPUT_DIR/train.align.en-${lg}.rev > $OUTPUT_DIR/train.align.en-${lg}
    echo "Generating Priors -> $OUTPUT_DIR/train.align.en-${lg}.priors"
    python $MAKE_PRIORS -i $OUTPUT_DIR/train.en-${lg} -f $OUTPUT_DIR/train.align.en-${lg}.fwd -r $OUTPUT_DIR/train.align.en-${lg}.rev --priors $OUTPUT_DIR/train.align.en-${lg}.priors
    echo "Preprocessing Alignments: $OUTPUT_DIR/train.align.en-${lg} -> $INPUT_DIR/train.align.en-${lg}"
    python $PREPROCESS_ALIGN -input $OUTPUT_DIR/train.align.en-${lg} -output $INPUT_DIR/train.align.en-${lg}
fi