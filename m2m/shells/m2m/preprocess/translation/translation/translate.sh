export CUDA_VISIBLE_DEVICES=0
DATA_BIN=/path/to/NER/m2m/20M/data-bin-split10/data-bin0/
INPUT_DIR=/path/to/NER/xtreme_v1/translation/X/
OUTPUT_DIR=/path/to/NER/xtreme_v1/translation/BT/
mkdir -p $OUTPUT_DIR
src=$1
tgt=$2
beam=$3
nbest=$4
MODEL=$5
INPUT_FILE=$6
OUTPUT_FILE=$7
BATCH_SIZE=$8
INPUT=$INPUT_DIR/$INPUT_FILE
OUTPUT=$OUTPUT_DIR/$OUTPUT_FILE
BUFFER_SIZE=10000
echo "${src}->${tgt} | beam $beam | MODEL $MODEL"
echo "INPUT $INPUT | OUTPUT $OUTPUT"
echo "BATCH_SIZE $BATCH_SIZE | BUFFER_SIZE $BUFFER_SIZE"
lenpen=1.0

LANGS="af,am,ar,ast,az,ba,be,bg,bn,br,bs,ca,ceb,cs,cy,da,de,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kk,km,kn,ko,lb,lg,ln,lo,lt,lv,mg,mk,ml,mn,mr,ms,my,ne,nl,no,ns,oc,or,pa,pl,ps,pt,ro,ru,sd,si,sk,sl,so,sq,sr,ss,su,sv,sw,ta,th,tl,tn,tr,uk,ur,uz,vi,wo,xh,yi,yo,zh,zu,eu,te"
LANG_PAIRS="en-af,af-en,en-ar,ar-en,en-bg,bg-en,en-bn,bn-en,en-de,de-en,en-el,el-en,en-es,es-en,en-et,et-en,en-eu,eu-en,en-fa,fa-en,en-fi,fi-en,en-fr,fr-en,en-he,he-en,en-hi,hi-en,en-hu,hu-en,en-id,id-en,en-it,it-en,en-ja,ja-en,en-jv,jv-en,en-ka,ka-en,en-kk,kk-en,en-ko,ko-en,en-ml,ml-en,en-mr,mr-en,en-ms,ms-en,en-my,my-en,en-nl,nl-en,en-pt,pt-en,en-ru,ru-en,en-sw,sw-en,en-ta,ta-en,en-te,te-en,en-th,th-en,en-tl,tl-en,en-tr,tr-en,en-ur,ur-en,en-vi,vi-en,en-yo,yo-en,en-zh,zh-en"
SPM_MODEL=/path/to/NER/PretrainedModels/m2m-100/spm.128k.model
cat $INPUT | python ./fairseq_cli/interactive.py $DATA_BIN \
    --path $MODEL \
    --encoder-langtok "src" --langtoks '{"main":("src", "tgt")}' \
    --task translation_multi_simple_epoch \
    --langs $LANGS --truncate-source \
    --lang-pairs $LANG_PAIRS --max-len-b 256 --min-len 2 --nbest $nbest \
    --source-lang $src --target-lang $tgt \
    --buffer-size $BUFFER_SIZE --batch-size $BATCH_SIZE --beam $beam --lenpen $lenpen --unkpen 10000 \
    --no-progress-bar --fp16 --bpe sentencepiece --sentencepiece-model $SPM_MODEL > $OUTPUT.log
#--bpe sentencepiece --sentencepiece-model $SPM_MODEL
#sed "s/	: /	/g" | sed "s/	, /	/g" |
cat $OUTPUT.log | grep -P "^D" | cut -f 3- > $OUTPUT
echo "Successfully saving $INPUT to $OUTPUT..."
