DATA_DIR=/path/to/NER/unitrans-data/xlmr_augment_v1/processed/
PRETRAINED_ENCODER=xlm-roberta-base
PRETRAINED_ENCODER_PATH=/path/to/NER/PretrainedModels/$PRETRAINED_ENCODER
EPOCH=$1
MAX_LENGTH=$2
LR=$3
BATCH_SIZE=$4
DROPOUT=$5
WEIGHT_DECAY=$6
SEED=$7
GRAD_ACC=1
#LANGS="de,en,es,nl,no"
PREDICT_LANGS="de,es,nl,no"
EVALUATE_STEPS=1000
if [ ! $EPOCH ]; then
    EPOCH=30
fi
if [ ! $MAX_LENGTH ]; then
    MAX_LENGTH=128
fi
if [ ! $SEED ]; then
    SEED=1
fi
if [ ! $DROPOUT ]; then
    DROPOUT=0.1
fi
if [ ! $WEIGHT_DECAY ]; then
    DROPOUT=0
fi

echo "Training Setting: LR ${LR}, MAX_LENGTH: ${MAX_LENGTH}, BATCH_SIZE: ${BATCH_SIZE}, GRAD_ACC: ${GRAD_ACC}"
OUTPUT_DIR=/path/to/NER/unitrans-data/xlmr_augment_v1/model/augment_v1/grid_search/xlmr-bsz${BATCH_SIZE}-maxlen${MAX_LENGTH}-lr${LR}-epoch${EPOCH}-seed${SEED}-dropout${DROPOUT}-weight-decay${WEIGHT_DECAY}-pooling/
mkdir -p $OUTPUT_DIR
python src/run_tag.py --model_type xlmr --model_name_or_path $PRETRAINED_ENCODER_PATH \
      --do_train --do_eval --do_predict --do_predict_dev --predict_langs $PREDICT_LANGS \
      --train_langs en --data_dir $DATA_DIR --labels $DATA_DIR/labels.txt \
      --per_gpu_train_batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACC --per_gpu_eval_batch_size 256 \
      --learning_rate $LR --num_train_epochs $EPOCH \
      --max_seq_length $MAX_LENGTH --noised_max_seq_length $MAX_LENGTH \
      --output_dir $OUTPUT_DIR --overwrite_output_dir --evaluate_during_training --logging_steps 100 --use_pooling_strategy --weight_decay ${WEIGHT_DECAY} \
      --evaluate_steps $EVALUATE_STEPS --seed $SEED --warmup_steps -1 \
      --save_only_best_checkpoint --eval_all_checkpoints --eval_patience 10 --fp16 --fp16_opt_level O2 --hidden_dropout_prob $DROPOUT --original_loss | tee -a $OUTPUT_DIR/log.txt \
