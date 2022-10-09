TEXT=/path/to/NER/m2m/20M/data-bin-split10/data-bin0/:/path/to/NER/m2m/20M/data-bin-split10/data-bin1/:/path/to/NER/m2m/20M/data-bin-split10/data-bin2/:/path/to/NER/m2m/20M/data-bin-split10/data-bin3/:/path/to/NER/m2m/20M/data-bin-split10/data-bin4/:/path/to/NER/m2m/20M/data-bin-split10/data-bin5/:/path/to/NER/m2m/20M/data-bin-split10/data-bin6/:/path/to/NER/m2m/20M/data-bin-split10/data-bin7/:/path/to/NER/m2m/20M/data-bin-split10/data-bin8/:/path/to/NER/m2m/20M/data-bin-split10/data-bin9/

PRETRAINED_MODEL=/path/to/NER/PretrainedModels/m2m-100/418M_last_checkpoint.pt
SPM_MODEL=/path/to/NER/PretrainedModels/m2m-100/spm.128k.model
LANGS="af,am,ar,ast,az,ba,be,bg,bn,br,bs,ca,ceb,cs,cy,da,de,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kk,km,kn,ko,lb,lg,ln,lo,lt,lv,mg,mk,ml,mn,mr,ms,my,ne,nl,no,ns,oc,or,pa,pl,ps,pt,ro,ru,sd,si,sk,sl,so,sq,sr,ss,su,sv,sw,ta,th,tl,tn,tr,uk,ur,uz,vi,wo,xh,yi,yo,zh,zu,eu,te"
LANG_PAIRS="en-af,af-en,en-ar,ar-en,en-bg,bg-en,en-bn,bn-en,en-de,de-en,en-el,el-en,en-es,es-en,en-et,et-en,en-eu,eu-en,en-fa,fa-en,en-fi,fi-en,en-fr,fr-en,en-he,he-en,en-hi,hi-en,en-hu,hu-en,en-id,id-en,en-it,it-en,en-ja,ja-en,en-jv,jv-en,en-ka,ka-en,en-kk,kk-en,en-ko,ko-en,en-ml,ml-en,en-mr,mr-en,en-ms,ms-en,en-my,my-en,en-nl,nl-en,en-pt,pt-en,en-ru,ru-en,en-sw,sw-en,en-ta,ta-en,en-te,te-en,en-th,th-en,en-tl,tl-en,en-tr,tr-en,en-ur,ur-en,en-vi,vi-en,en-yo,yo-en,en-zh,zh-en"
LR=1e-4
NODES=4
GPUS=8
SUBWORD_PROB=0.85
UPDATE_FREQ=4
MAX_TOKENS=5120
MAX_EPOCH=100
MAX_UPDATES=400000
MODEL=/path/to/NER/m2m/model/M2M101-lr_${LR}-subword_prob_${SUBWORD_PROB}-MAX_TOKENS_${MAX_TOKENS}-UPDATE_FREQ_${MAX_UPDATES}/
mkdir -p $MODEL
echo "Start Training ..."
if [ ! -f $MODEL/checkpoint_last.pt ]; then  #compatible with preemptible job
    python -m torch.distributed.launch --nproc_per_node=$GPUS --nnodes=$NODES --node_rank=$OMPI_COMM_WORLD_RANK --master_addr="$MASTER_ADDR" --master_port=$MASTER_PORT train.py $TEXT \
        --save-dir $MODEL --restore-file $PRETRAINED_MODEL --arch transformer_wmt_en_de_big \
        --encoder-normalize-before --decoder-normalize-before --encoder-layers 12 --decoder-layers 12  \
        --task translation_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 --warmup-epoch 5 \
        --encoder-langtok "src" --langtoks '{"main":("src","tgt")}' --langs $LANGS \
        --lang-pairs $LANG_PAIRS --truncate-source --enable-reservsed-directions-shared-datasets --load-alignments \
        --share-all-embeddings --max-source-positions 1024 --max-target-positions 1024 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr $LR --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-updates 4000 \
        --max-update $MAX_UPDATES --max-epoch $MAX_EPOCH --attention-dropout 0.1 --dropout 0.1 --max-tokens $MAX_TOKENS --update-freq $UPDATE_FREQ \
        --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test --fp16 --subword-prob $SUBWORD_PROB \
        --reset-optimizer --reset-meters --reset-lr-scheduler --reset-dataloader --ddp-backend=no_c10d 2>&1 | tee -a $MODEL/train.log
else
    python -m torch.distributed.launch --nproc_per_node=$GPUS --nnodes=$NODES --node_rank=$OMPI_COMM_WORLD_RANK --master_addr="$MASTER_ADDR" --master_port=$MASTER_PORT train.py $TEXT \
        --save-dir $MODEL --restore-file $PRETRAINED_MODEL --arch transformer_wmt_en_de_big \ 
        --encoder-normalize-before --decoder-normalize-before --encoder-layers 12 --decoder-layers 12  \
        --task translation_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 --warmup-epoch 5 \
        --encoder-langtok "src" --langtoks '{"main":("src","tgt")}' --langs $LANGS \
        --lang-pairs $LANG_PAIRS --truncate-source --enable-reservsed-directions-shared-datasets --load-alignments \
        --share-all-embeddings --max-source-positions 1024 --max-target-positions 1024 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr $LR --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-updates 4000 \
        --max-update $MAX_UPDATES --max-epoch $MAX_EPOCH --attention-dropout 0.1 --dropout 0.1 --max-tokens $MAX_TOKENS --update-freq $UPDATE_FREQ \
        --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test --fp16 --subword-prob $SUBWORD_PROB \
        --reset-optimizer --reset-meters --reset-lr-scheduler --reset-dataloader --ddp-backend=no_c10d 2>&1 | tee -a $MODEL/train.log
fi
#--same-lang-per-batch --enable-lang-ids
