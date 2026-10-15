#!/bin/env bash

# ---- Tokens ----
export WANDB_API_KEY="" # Set your wandb token here
export HF_TOKEN="" # Set your huggingface token here

# ---- TorchInductor cache ----
export TORCHINDUCTOR_CACHE_DIR=/dev/shm/tmp # RAM
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
export TORCHINDUCTOR_CACHE_SIZE_LIMIT_MB=32768  
export TORCH_LOGS="recompiles"

# ---- Compilation / threading ----
export TORCHINDUCTOR_COMPILE_THREADS=8 # tune ~ (#CPU cores / GPUs)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# ---- CUDA / perf ----
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NVIDIA_TF32_OVERRIDE=1 # or 1 if you want TF32 matmul

# ---- Debug / logging (optional) ----
export TORCH_LOGS="output_code"
export TORCH_COMPILE_DEBUG=0

# ---- Training Configurations

PRETRAINED_MODEL_NAME_OR_PATH="black-forest-labs/FLUX.1-Kontext-dev" # Pretrained Flux Model
CUSTOM_PIPELINE="pipelines/pipeline_flux_kontext.py" # Custom Pipeline for thinking-in-boxes
DATASET_NAME="pradhaansbhat/Thinking-In-Boxes" # Dataset
DATASET_CONFIG=("100K-Syn") # Stage 1; ("10K-Objectron" "10K-Syn")  # Stage 2
IMAGE_COLUMN="rgb_1.png" # Target Image
REFERENCE_IMAGE_COLUMN="rgb_0.png" # Source Image
SOURCE_PRIMITIVE_COLUMN="bbox_0.png" # Source Boxes
TARGET_PRIMITIVE_COLUMN="bbox_1.png" # Target Boxes

CHECKPOINTING_STEPS=2500 # Checkpoint every N steps
CHECKPOINTING_LIMIT=10 # Store at max C checkpoints
OUTPUT_DIR="Thinking-In-Boxes-R32-Stage1" # Store LoRA on local dir
DATALOADER_NUM_WORKERS=8 # Number of dataloaders
DATASET_NUM_PROCESSES=16 # Number of dataset processors

# Generation Arguments
RESOLUTION=512 # Image Resolution
GUIDANCE_SCALE=1 # Disabled

# LoRA Arguments
RANK=32 
ALPHA=32
TARGET_REGEX="attn.to_k, attn.to_q, attn.to_v, attn.to_out.0, ff.net.2, ff.net.0.proj" # Attn Matrices + MLPs
INIT_LORA_WEIGHTS="gaussian"

# Optimization Arguments
OPTIMIZER="prodigy"
WEIGHT_DECAY=0.01
LEARNING_RATE=1
TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
NUM_TRAIN_EPOCHS=8
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=250
DROP_IMAGE_PROB=0.1
CROP_PROB=0.1

# Validation Arguments
VALIDATION_STEPS=2500 # Validation every N steps
VALIDATION_PROMPT="" # Disable prompt
VALIDATION_REFERENCE_IMAGE="examples/A/src.jpg"
VALIDATION_SOURCE_PRIMITIVE="examples/A/src_box.png"
VALIDATION_TARGET_PRIMITIVE="examples/A/tgt_box_camera.png"
NUM_VALIDATION_IMAGES=4

COMMENT_STRING="Rank=32; Thinking In Boxes LoRA Rank=32 Stage-1" # Any comment to store in output dir

# Additional Args

REPORT_TO="wandb"
MIXED_PRECISION="bf16"
COMPILE_MODE="max-autotune-no-cudagraphs"

########################################
#           STAGE 1 Training 
########################################

uv run accelerate launch train.py \
  --pretrained_model_name_or_path="$PRETRAINED_MODEL_NAME_OR_PATH"  \
  --custom_pipeline="$CUSTOM_PIPELINE" \
  --dataset_name="$DATASET_NAME" \
  --dataset_config_name "${DATASET_CONFIG[@]}" \
  --image_column="$IMAGE_COLUMN" \
  --reference_image_column="$REFERENCE_IMAGE_COLUMN" \
  --source_primitive_column="$SOURCE_PRIMITIVE_COLUMN" \
  --target_primitive_column="$TARGET_PRIMITIVE_COLUMN" \
  --output_dir="$OUTPUT_DIR" \
  --dataloader_num_workers=$DATALOADER_NUM_WORKERS \
  --mixed_precision="$MIXED_PRECISION" \
  --resolution=$RESOLUTION \
  --guidance_scale=$GUIDANCE_SCALE \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --learning_rate=$LEARNING_RATE \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --rank=$RANK \
  --lora_alpha=$ALPHA \
  --init_lora_weights=$INIT_LORA_WEIGHTS \
  --lora_layers="$TARGET_REGEX" \
  --drop_image_prob=$DROP_IMAGE_PROB \
  --crop_prob=$CROP_PROB \
  --optimizer="$OPTIMIZER" \
  --adam_weight_decay=$WEIGHT_DECAY \
  --report_to="$REPORT_TO" \
  --lr_scheduler="$LR_SCHEDULER" \
  --lr_warmup_steps=$LR_WARMUP_STEPS \
  --num_train_epochs=$NUM_TRAIN_EPOCHS \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --resume_from_checkpoint="latest" \
  --validation_steps=$VALIDATION_STEPS \
  --validation_prompt="$VALIDATION_PROMPT" \
  --num_validation_images=$NUM_VALIDATION_IMAGES \
  --validation_reference_image="$VALIDATION_REFERENCE_IMAGE" \
  --validation_source_primitive="$VALIDATION_SOURCE_PRIMITIVE" \
  --validation_target_primitive="$VALIDATION_TARGET_PRIMITIVE" \
  --compile \
  --compile_mode="$COMPILE_MODE" \
  --comment_string="$COMMENT_STRING" \
  --dataset_num_proc $DATASET_NUM_PROCESSES

mkdir -p Thinking-In-Boxes-R32-Stage2
cp -r Thinking-In-Boxes-R32-Stage1/checkpoint-25000 Thinking-In-Boxes-R32-Stage2
mv Thinking-In-Boxes-R32-Stage2/checkpoint-25000 Thinking-In-Boxes-R32-Stage2/checkpoint-00000

OUTPUT_DIR="Thinking-In-Boxes-R32-Stage2" # Store LoRA on local dir
DATASET_CONFIG=("10K-Objectron" "10K-Syn") # Stage 2;  ("100K-Syn") # Stage 1
COMMENT_STRING="Rank=32; Thinking In Boxes LoRA Rank=32 Stage-1" # Any comment to store in output dir

########################################
#           STAGE 2 Training 
########################################

uv run accelerate launch train.py \
  --pretrained_model_name_or_path="$PRETRAINED_MODEL_NAME_OR_PATH"  \
  --custom_pipeline="$CUSTOM_PIPELINE" \
  --dataset_name="$DATASET_NAME" \
  --dataset_config_name "${DATASET_CONFIG[@]}" \
  --image_column="$IMAGE_COLUMN" \
  --reference_image_column="$REFERENCE_IMAGE_COLUMN" \
  --source_primitive_column="$SOURCE_PRIMITIVE_COLUMN" \
  --target_primitive_column="$TARGET_PRIMITIVE_COLUMN" \
  --output_dir="$OUTPUT_DIR" \
  --dataloader_num_workers=$DATALOADER_NUM_WORKERS \
  --mixed_precision="$MIXED_PRECISION" \
  --resolution=$RESOLUTION \
  --guidance_scale=$GUIDANCE_SCALE \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --learning_rate=$LEARNING_RATE \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --rank=$RANK \
  --lora_alpha=$ALPHA \
  --init_lora_weights=$INIT_LORA_WEIGHTS \
  --lora_layers="$TARGET_REGEX" \
  --drop_image_prob=$DROP_IMAGE_PROB \
  --crop_prob=$CROP_PROB \
  --optimizer="$OPTIMIZER" \
  --adam_weight_decay=$WEIGHT_DECAY \
  --report_to="$REPORT_TO" \
  --lr_scheduler="$LR_SCHEDULER" \
  --lr_warmup_steps=$LR_WARMUP_STEPS \
  --num_train_epochs=$NUM_TRAIN_EPOCHS \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --resume_from_checkpoint="latest" \
  --validation_steps=$VALIDATION_STEPS \
  --validation_prompt="$VALIDATION_PROMPT" \
  --num_validation_images=$NUM_VALIDATION_IMAGES \
  --validation_reference_image="$VALIDATION_REFERENCE_IMAGE" \
  --validation_source_primitive="$VALIDATION_SOURCE_PRIMITIVE" \
  --validation_target_primitive="$VALIDATION_TARGET_PRIMITIVE" \
  --compile \
  --compile_mode="$COMPILE_MODE" \
  --comment_string="$COMMENT_STRING" \
  --dataset_num_proc $DATASET_NUM_PROCESSES