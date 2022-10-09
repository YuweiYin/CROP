XLMR_SPM_MODEL=/path/to/NER/PretrainedModels/xlm-roberta-base/sentencepiece.bpe.model
M2M_SPM_MODEL=/path/to/NER/PretrainedModels/flores/flores101_mm100_615M/sentencepiece.bpe.model
INPUT_DIR=/path/to/NER/flores/cc_bpe/
RAW_DIR=/path/to/NER/flores/raw/
SPM_DIR=/path/to/NER/flores/m2m_bpe/
OUTPUT_DIR=/path/to/NER/flores/All/train/
mkdir -p $RAW_DIR $SPM_DIR $OUTPUT_DIR
FILTER=./scripts/pattern/filter.py
#LANGS=(af ar bg bn de el es et fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo zh)
lg1=$1
lg2=$2
STATE="TOK" #ALL DETOK TOK FILTER
if [ $STATE = "ALL" -o $STATE = "DETOK" ]; then
    INPUT=$INPUT_DIR/train.${lg1}-en.${lg2}
    OUTPUT=$RAW_DIR/train.en-${lg1}.${lg2}
    echo "Detokenizing $INPUT -> $OUTPUT"
    cat $INPUT | spm_decode --model=$XLMR_SPM_MODEL --input_format=piece > $OUTPUT
fi

if [ $STATE = "ALL" -o $STATE = "TOK" ]; then 
    INPUT=$RAW_DIR/train.en-${lg1}.${lg2}
    OUTPUT=$SPM_DIR/train.en-${lg1}.${lg2}
    echo "Tokenizing $INPUT -> $OUTPUT"
    cat $INPUT | spm_encode --model=$M2M_SPM_MODEL --output_format=piece > $OUTPUT
fi


