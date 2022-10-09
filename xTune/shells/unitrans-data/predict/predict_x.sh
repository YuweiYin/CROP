export CUDA_VISIBLE_DEVICES=0
LANG=$1
TRAIN_DIR=/path/to/NER/unitrans-data/xlmr_augment_v1/processed/
DATA_DIR=/path/to/NER/unitrans-data/xlmr_augment_v1/translation/X_NER/
MODEL=/path/to/NER/unitrans-data/xlmr_augment_v1/model/augment_v1/iter1/xlmr-bsz32-maxlen128-lr1e-5-epoch20/checkpoint-best/
MAX_LENGTH=128
TEST_OUTPUT_FILE=/path/to/NER/unitrans-data/xlmr_augment_v1/translation/X_NER/$LANG/test.xlmr.tsv
python ./src/run_tag.py --model_type xlmr \
    --model_name_or_path $MODEL \
    --init_checkpoint $MODEL --do_predict --predict_langs $LANG --train_langs en --data_dir $DATA_DIR \
    --labels $TRAIN_DIR/labels.txt --per_gpu_train_batch_size 2 --gradient_accumulation_steps 1 --per_gpu_eval_batch_size 256 \
    --learning_rate 1e-5 --num_train_epochs 1 --max_seq_length $MAX_LENGTH --noised_max_seq_length $MAX_LENGTH \
    --output_dir $MODEL --overwrite_output_dir --overwrite_cache --evaluate_during_training --test_output_file $TEST_OUTPUT_FILE \
    --logging_steps 50 --seed 1 --warmup_steps -1 --save_only_best_checkpoint --eval_all_checkpoints --eval_patience -1 --fp16 --fp16_opt_level O2 --original_loss 