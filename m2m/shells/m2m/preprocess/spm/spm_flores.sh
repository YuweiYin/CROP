XLMR_SPM_MODEL=/path/to/NER/PretrainedModels/xlm-roberta-base/sentencepiece.bpe.model
M2M_SPM_MODEL=/path/to/NER/PretrainedModels/flores/flores101_mm100_615M/sentencepiece.bpe.model
INPUT_DIR=/path/to/NER/flores/cc_bpe/
RAW_DIR=/path/to/NER/flores/raw/
SPM_DIR=/path/to/NER/flores/m2m_bpe/
OUTPUT_DIR=/path/to/NER/flores/20M/train/
mkdir -p $RAW_DIR $SPM_DIR $OUTPUT_DIR
FILTER=./scripts/pattern/filter.py
#LANGS=(af ar bg bn de el es et fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl no pt ru sw ta te th tl tr ur vi yo zh)
lg=$1
STATE="FILTER" #ALL DETOK TOK FILTER
if [ $STATE = "ALL" -o $STATE = "DETOK" ]; then
    INPUT=$INPUT_DIR/train.${lg}-en.${lg}
    OUTPUT=$RAW_DIR/train.en-${lg}.${lg}
    echo "Detokenizing $INPUT -> $OUTPUT"
    cat $INPUT | spm_decode --model=$XLMR_SPM_MODEL --input_format=piece > $OUTPUT

    INPUT=$INPUT_DIR/train.${lg}-en.en
    OUTPUT=$RAW_DIR/train.en-${lg}.en
    echo "Detokenizing $INPUT -> $OUTPUT"
    cat $INPUT | spm_decode --model=$XLMR_SPM_MODEL --input_format=piece > $OUTPUT
fi

if [ $STATE = "ALL" -o $STATE = "TOK" ]; then 
    INPUT=$RAW_DIR/train.en-${lg}.en
    OUTPUT=$SPM_DIR/train.en-${lg}.en
    echo "Tokenizing $INPUT -> $OUTPUT"
    cat $INPUT | spm_encode --model=$M2M_SPM_MODEL --output_format=piece > $OUTPUT

    INPUT=$RAW_DIR/train.en-${lg}.${lg}
    OUTPUT=$SPM_DIR/train.en-${lg}.${lg}
    echo "Tokenizing $INPUT -> $OUTPUT"
    cat $INPUT | spm_encode --model=$M2M_SPM_MODEL --output_format=piece > $OUTPUT
fi

if [ $STATE = "ALL" -o $STATE = "FILTER" ]; then 
    SRC=$SPM_DIR/train.en-${lg}.en
    TGT=$SPM_DIR/train.en-${lg}.${lg}
    NEW_SRC=$OUTPUT_DIR/train.en-${lg}.en
    NEW_TGT=$OUTPUT_DIR/train.en-${lg}.${lg}
    echo "Filtering $SRC || $TGT -> $NEW_SRC || $NEW_TGT"
    python $FILTER -src $SRC -tgt $TGT -new-src $NEW_SRC -new-tgt $NEW_TGT -length-ratio 1.5 -max-length 250 -max-sentences 20000000
fi

