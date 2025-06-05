ulimit -c unlimited

export WANDB_WATCH=gradients
export PYTHONPATH=.

MODEL_NAME='facebook/opt-125m'
CONTEXT='all'
DESCRIPTION=${MODEL_NAME}-${CONTEXT}

CUDA_VISIBLE_DEVICES=6,1 python language_modelling/run_generation.py \
    --dataset wikiweb2m \
    --neighbor_mode raw \
    --model_name_or_path ${MODEL_NAME} \
    --context ${CONTEXT} \
    --peft_type none \
    --position_type none \
    --max_input_length 1024 \
    --max_output_length 128 \
    --epochs 50 \
    --steps_per_epoch 10000 \
    --val_steps_per_epoch 400 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 8 \
    --per_device_val_batch_size 8 \
    --dataloader_num_workers 8 \
    --grad_accumulation_steps 16 \
    --fp16 \
    --wandb_project Graph4MM \
    --wandb_run ${DESCRIPTION}
