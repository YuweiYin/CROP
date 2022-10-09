TEXT=/path/to/NER/flores/20M/data-bin-split10/data-bin0/:/path/to/NER/flores/20M/data-bin-split10/data-bin1/:/path/to/NER/flores/20M/data-bin-split10/data-bin2/:/path/to/NER/flores/20M/data-bin-split10/data-bin3/:/path/to/NER/flores/20M/data-bin-split10/data-bin4/:/path/to/NER/flores/20M/data-bin-split10/data-bin5/:/path/to/NER/flores/20M/data-bin-split10/data-bin6/:/path/to/NER/flores/20M/data-bin-split10/data-bin7/:/path/to/NER/flores/20M/data-bin-split10/data-bin8/:/path/to/NER/flores/20M/data-bin-split10/data-bin9/


PRETRAINED_MODEL=/path/to/NER/PretrainedModels/flores/flores101_mm100_615M/model.pt
SPM_MODEL=/path/to/NER/PretrainedModels/flores/flores101_mm100_615M/sentencepiece.bpe.model
LANGS="af,am,ar,as,ast,ay,az,ba,be,bg,bn,br,bs,ca,ceb,cjk,cs,cy,da,de,dyu,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kac,kam,kea,kg,kk,km,kmb,kmr,kn,ko,ku,ky,lb,lg,ln,lo,lt,luo,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,no,ns,ny,oc,om,or,pa,pl,ps,pt,qu,ro,ru,sd,shn,si,sk,sl,sn,so,sq,sr,ss,su,sv,sw,ta,te,tg,th,ti,tl,tn,tr,uk,umb,ur,uz,vi,wo,xh,yi,yo,zh,zu,eu"
LANG_PAIRS="en-af,af-en,en-ar,ar-en,en-bg,bg-en,en-bn,bn-en,en-de,de-en,en-el,el-en,en-es,es-en,en-et,et-en,en-eu,eu-en,en-fa,fa-en,en-fi,fi-en,en-fr,fr-en,en-he,he-en,en-hi,hi-en,en-hu,hu-en,en-id,id-en,en-it,it-en,en-ja,ja-en,en-jv,jv-en,en-ka,ka-en,en-kk,kk-en,en-ko,ko-en,en-ml,ml-en,en-mr,mr-en,en-ms,ms-en,en-my,my-en,en-nl,nl-en,en-no,no-en,en-pt,pt-en,en-ru,ru-en,en-sw,sw-en,en-ta,ta-en,en-te,te-en,en-th,th-en,en-tl,tl-en,en-tr,tr-en,en-ur,ur-en,en-vi,vi-en,en-yo,yo-en,en-zh,zh-en"
LR=1e-4
NODES=5
GPUS=8
MAX_SLOT_NUM=14
SUBWORD_PROB=0.0
UPDATE_FREQ=16
MAX_TOKENS=1536
MAX_EPOCH=100
MAX_UPDATES=400000
SLOT_METHOD="insert"
SLOT_PROB=0.85
#MODEL=/path/to/NER/flores/20M/model/insert/lr_${LR}-SLOT_PROB_${SLOT_PROB}-subword_prob_${SUBWORD_PROB}-MAX_TOKENS_${MAX_TOKENS}-UPDATE_FREQ_${UPDATE_FREQ}-MAX_SLOT_NUM_${MAX_SLOT_NUM}-NODES_${NODES}-GPUS_${GPUS}-SLOT_METHOD-${SLOT_METHOD}/
MODEL=/path/to/NER/flores/20M/model/insert/lr_1e-4-SLOT_PROB_0.85-subword_prob_0.0-MAX_TOKENS_1024-UPDATE_FREQ_16-MAX_SLOT_NUM_14-NODES_8-GPUS_8-SLOT_METHOD-insert/
mkdir -p $MODEL
echo "Start Training ..."
if [ ! -f $MODEL/checkpoint_last.pt ]; then  #compatible with preemptible job
    python -m torch.distributed.launch --nproc_per_node=$GPUS --nnodes=$NODES --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py $TEXT \
        --save-dir $MODEL --restore-file $PRETRAINED_MODEL --arch transformer_wmt_en_de_big \
        --encoder-normalize-before --decoder-normalize-before --encoder-layers 12 --decoder-layers 12  \
        --task translation_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 --warmup-epoch 5 \
        --encoder-langtok "src" --langtoks '{"main":("src","tgt")}' --langs $LANGS \
        --lang-pairs $LANG_PAIRS --truncate-source --enable-reservsed-directions-shared-datasets --load-alignments \
        --share-all-embeddings --max-source-positions 1024 --max-target-positions 1024 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr $LR --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-updates 4000 \
        --max-update $MAX_UPDATES --max-epoch $MAX_EPOCH --attention-dropout 0.1 --dropout 0.1 --max-tokens $MAX_TOKENS --update-freq $UPDATE_FREQ \
        --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test --fp16 --bpe sentencepiece --sentencepiece-model $SPM_MODEL \
        --subword-prob $SUBWORD_PROB --max-slot-num $MAX_SLOT_NUM --slot-prob $SLOT_PROB --slot-method $SLOT_METHOD \
        --reset-optimizer --reset-meters --reset-lr-scheduler --reset-dataloader --ddp-backend=no_c10d 2>&1 | tee -a $MODEL/train.log
else
    python -m torch.distributed.launch --nproc_per_node=$GPUS --nnodes=$NODES --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py $TEXT \
        --save-dir $MODEL --arch transformer_wmt_en_de_big \
        --encoder-normalize-before --decoder-normalize-before --encoder-layers 12 --decoder-layers 12  \
        --task translation_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 --warmup-epoch 5 \
        --encoder-langtok "src" --langtoks '{"main":("src","tgt")}' --langs $LANGS \
        --lang-pairs $LANG_PAIRS --truncate-source --enable-reservsed-directions-shared-datasets --load-alignments \
        --share-all-embeddings --max-source-positions 1024 --max-target-positions 1024 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr $LR --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-updates 4000 \
        --max-update $MAX_UPDATES --max-epoch $MAX_EPOCH --attention-dropout 0.1 --dropout 0.1 --max-tokens $MAX_TOKENS --update-freq $UPDATE_FREQ \
        --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test --fp16 --bpe sentencepiece --sentencepiece-model $SPM_MODEL \
        --subword-prob $SUBWORD_PROB --max-slot-num $MAX_SLOT_NUM --slot-prob $SLOT_PROB --slot-method $SLOT_METHOD \
        --ddp-backend=no_c10d 2>&1 | tee -a $MODEL/train.log
fi
#--same-lang-per-batch --enable-lang-ids
