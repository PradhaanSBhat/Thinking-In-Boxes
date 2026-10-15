#!/bin/env bash

# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
export HF_TOKEN="" # Add HF_TOKEN here (Optional)

PRETRAINED_MODEL_NAME_OR_PATH="black-forest-labs/FLUX.1-Kontext-dev" # Huggingface Model Name or Path
PIPELINE_PATH="pipelines/pipeline_flux_kontext.py" # Custom Pipeline Path

PROMPT="" # Prompt to guide generation; Leave empty, Not used
NEGATIVE_PROMPT="" # Negative Prompt; Leave empty, Not used

DTYPE="bfloat16" # Tensor Data Type
DEVICE_MAP="cuda" # Devices to map the tensors to

HEIGHT=512 # Image Height; Model was primarily trained for 512x512, use this resolution for best results
WIDTH=512 # Image Width; Model was primarily trained for 512x512, use this resolution for best results
MAX_AREA=$(($HEIGHT*$WIDTH))
GUIDANCE_SCALE=3.5 # Guidance Scale for generation
NUM_INFERENCE_STEPS=28 # Number of inference Steps
TOTAL=4 # Total number of images to be generated
NUM_IMAGES_PER_PROMPT=4 # Number of images to be generated per prompt
SEED=42 # Seed to guide generation
MAX_SEQUENCE_LENGTH=512 # Sequence length of text tensor

LORA_ADAPTER_PATH="pradhaansbhat/Thinking-In-Boxes" # LoRA Checkpoint Path
WEIGHT_NAME="pytorch_lora_weights.safetensors" # LoRA Saved Weight Name
ADAPTER_NAME="100K+20K-LoRA" # LoRA Adapter Name
SCALE=1 # LoRA Scale

REFERENCE_IMAGE="examples/A/src.jpg" # Reference Image
SOURCE_PRIMITIVE="examples/A/src_box.png" # Source Boxes
TARGET_PRIMITIVE="examples/A/tgt_box_camera.png" # Target Boxes

SAVE_DIR="Outputs/Thinking-In-Boxes-100K+20K-LoRA/A" # Where to save images

# ---------------------- Run Command ----------------------------

(
    CUDA_VISIBLE_DEVICES=0 uv run python inference.py \
    --pretrained_model_name_or_path "$PRETRAINED_MODEL_NAME_OR_PATH" \
    --pipeline_path "$PIPELINE_PATH" \
    --height $HEIGHT \
    --width $WIDTH \
    --max_area $MAX_AREA \
    --prompt "$PROMPT" \
    --guidance_scale $GUIDANCE_SCALE \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --num_images_per_prompt $NUM_IMAGES_PER_PROMPT \
    --total $TOTAL \
    --device_map "$DEVICE_MAP" \
    --save_dir "$SAVE_DIR" \
    --seed $SEED \
    --lora_adapter_path "$LORA_ADAPTER_PATH" \
    --weight_name "$WEIGHT_NAME" \
    --adapter_name "$ADAPTER_NAME" \
    --scale $SCALE \
    --image "$REFERENCE_IMAGE" \
    --source_primitive "$SOURCE_PRIMITIVE" \
    --target_primitive "$TARGET_PRIMITIVE" \
    --compile \
    --verbose \
    # --offload \
    # --local \
    # --vae_slicing \
) 