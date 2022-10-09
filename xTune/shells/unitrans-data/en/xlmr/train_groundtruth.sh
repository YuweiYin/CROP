DATA_DIR=/path/to/NER/xtreme_vg/panx_processed_maxlen128/
mkdir -p $OUTPUT_DIR
PRETRAINED_ENCODER=xlm-roberta-base
SEED=1
PRETRAINED_ENCODER_PATH=/path/to/NER/PretrainedModels/$PRETRAINED_ENCODER
EPOCH=10
MAX_LENGTH=128
LANGS="ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu"
EVALUATE_STEPS=10000
if [ $PRETRAINED_ENCODER == "xlm-roberta-large" ]; then
    BATCH_SIZE=32
    GRAD_ACC=1
    LR=7e-6
else
    BATCH_SIZE=32
    GRAD_ACC=1
    LR=1e-5
fi
OUTPUT_DIR=/path/to/NER/xtreme_vg/model/augment_vg/xlmr-bsz${BATCH_SIZE}-maxlen${MAX_LENGTH}/

python src/run_tag.py --model_type xlmr --model_name_or_path $PRETRAINED_ENCODER_PATH \
      --do_train --do_eval --do_predict --do_predict_dev --predict_langs $LANGS \
      --train_langs en --data_dir $DATA_DIR --labels $DATA_DIR/labels.txt \
      --per_gpu_train_batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACC --per_gpu_eval_batch_size 256 \
      --learning_rate $LR --num_train_epochs $EPOCH \
      --max_seq_length $MAX_LENGTH --noised_max_seq_length $MAX_LENGTH \
      --output_dir $OUTPUT_DIR --overwrite_output_dir --evaluate_during_training --logging_steps 50 \
      --evaluate_steps $EVALUATE_STEPS --seed $SEED --warmup_steps -1 \
      --save_only_best_checkpoint --eval_all_checkpoints --eval_patience -1 --fp16 --fp16_opt_level O2 --hidden_dropout_prob 0.1 --original_loss | tee -a $OUTPUT/log.txt \
