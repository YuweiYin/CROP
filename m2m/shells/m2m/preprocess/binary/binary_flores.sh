src=$1
tgt=$2
TRAIN=$3
DATA_BIN=$4
VALID_RAW=/path/to/SharedTask/thunder/flores101_dataset/devtest-code/
DICT=/path/to/NER/PretrainedModels/flores/flores101_mm100_615M/dict.txt  #add eu language symbol
M2M_SPM_MODEL=/path/to/NER/PretrainedModels/flores/flores101_mm100_615M/sentencepiece.bpe.model
if [ $src = "en" -a $tgt = "eu" ]; then #Can not find en-eu test set
    echo "Copying $TRAIN/train.$src-$tgt.$src top 1012 lines -> $TRAIN/valid.$src-$tgt.$src"
    head $TRAIN/train.$src-$tgt.$src -n 1012 > $TRAIN/valid.$src-$tgt.$src
    echo "Copying $TRAIN/train.$src-$tgt.$tgt top 1012 lines -> $TRAIN/valid.$src-$tgt.$tgt"
    head $TRAIN/train.$src-$tgt.$tgt -n 1012 > $TRAIN/valid.$src-$tgt.$tgt
else
    echo "Tokenizing $VALID_RAW/valid.$src-$tgt.$src -> $TRAIN/valid.$src-$tgt.$src"
    cat $VALID_RAW/valid.$src | spm_encode --model=$M2M_SPM_MODEL --output_format=piece > $TRAIN/valid.$src-$tgt.$src
    echo "Tokenizing $VALID_RAW/valid.$src-$tgt.$tgt -> $TRAIN/valid.$src-$tgt.$tgt"
    cat $VALID_RAW/valid.$tgt | spm_encode --model=$M2M_SPM_MODEL --output_format=piece > $TRAIN/valid.$src-$tgt.$tgt
fi

echo "Copying $TRAIN/train.align.${src}-${tgt}.npy -> $DATA_BIN/train.align.${src}-${tgt}.npy"
cp $TRAIN/train.align.${src}-${tgt}.npy $DATA_BIN/


echo "Start binarizing $TRAIN/train.${tgt}-${src}..."    
python ./fairseq_cli/preprocess.py  \
    --trainpref $TRAIN/train.${src}-${tgt} \
    --source-lang $src --target-lang $tgt \
    --destdir $DATA_BIN \
    --srcdict $DICT \
    --tgtdict $DICT \
    --workers 40

python ./fairseq_cli/preprocess.py  \
    --validpref $TRAIN/valid.${src}-${tgt} \
    --source-lang $src --target-lang $tgt \
    --destdir $DATA_BIN \
    --srcdict $DICT \
    --tgtdict $DICT \
    --workers 40