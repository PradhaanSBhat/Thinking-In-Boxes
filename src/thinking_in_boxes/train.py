#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "diffusers @ git+https://github.com/huggingface/diffusers.git",
#     "torch>=2.0.0",
#     "accelerate>=0.31.0",
#     "transformers>=4.41.2",
#     "ftfy",
#     "tensorboard",
#     "Jinja2",
#     "peft>=0.11.1",
#     "sentencepiece",
#     "torchvision",
#     "datasets",
#     "bitsandbytes",
#     "prodigyopt",
# ]
# ///

import argparse
import copy
import itertools
import json
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import diffusers
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    TorchDynamoPlugin,
    gather_object,
    set_seed,
)
from datasets import concatenate_datasets, load_dataset
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxKontextPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    EMAModel,
    _collate_lora_metadata,
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
    load_image,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo, hf_hub_download, upload_folder
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.37.0")

logger = get_logger(__name__)


def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    train_text_encoder=False,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {
                    "text": validation_prompt if validation_prompt else " ",
                    "output": {"url": f"image_{i}.png"},
                }
            )

    model_description = f"""
# Flux.1-Kontext LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} LoRA weights for {base_model}.

The weights were trained with the [Flux diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md).

Was LoRA for the text encoder enabled? {train_text_encoder}.

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch
pipeline = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to('cuda')
pipeline.load_lora_weights('{repo_id}', weight_name='pytorch_lora_weights.safetensors')
image = pipeline('{validation_prompt if validation_prompt else instance_prompt}').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev/blob/main/LICENSE.md).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora",
        "flux",
        "flux-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
    )
    return text_encoder_one, text_encoder_two


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    global_step,
    torch_dtype,
    is_final_validation=False,
):
    num_validation_images = (
        args.num_validation_images if args.num_validation_images else 1
    )
    total_indices = list(range(num_validation_images))

    with accelerator.split_between_processes(total_indices) as local_indices:
        logger.info(
            f"Running validation... \n Generating {num_validation_images // accelerator.num_processes} images with prompt:"
            f" {args.validation_prompt}."
        )
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        seed = args.seed + accelerator.process_index if args.seed is not None else None
        generator = (
            torch.Generator(device=accelerator.device).manual_seed(seed)
            if seed is not None
            else None
        )

        autocast_ctx = (
            torch.autocast(accelerator.device.type)
            if not is_final_validation
            else nullcontext()
        )

        # pre-calculate  prompt embeds, pooled prompt embeds, text ids because t5 does not support autocast
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                pipeline_args["prompt"], prompt_2=pipeline_args["prompt"]
            )

        local_images = []
        for _ in local_indices:
            with autocast_ctx:
                image = pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    generator=generator,
                    image=pipeline_args["image"],
                    source_primitive=pipeline_args["source_primitive"],
                    target_primitive=pipeline_args["target_primitive"],
                ).images[0]
                local_images.append(image)

    images = gather_object(local_images)

    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            phase_name = "test" if is_final_validation else "validation"
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images(
                    phase_name, np_images, global_step, dataformats="NHWC"
                )
            if tracker.name == "wandb":
                tracker.log(
                    {
                        phase_name: [
                            wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                            for i, image in enumerate(images)
                        ]
                    },
                    step=global_step,
                )

    del pipeline
    free_memory()

    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--custom_pipeline", type=str, help="Custom Pipeline Path used for Validation"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        nargs='+',
        default=None,
        help="One or more dataset configs. Multiple configs are concatenated.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--dataset_num_proc",
        default=16,
        type=int,
        help="Number of parallel processes to use when downloading/preparing a HF dataset",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="rgb_1.png",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'rgb_1.png'.",
    )
    parser.add_argument(
        "--reference_image_column",
        type=str,
        default="rgb_0.png",
        help="The column of the dataset containing the source image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'rgb_0.png'.",
    )
    ##########
    parser.add_argument(
        "--source_primitive_column",
        type=str,
        default="bbox_0.png",
        help="The column of the dataset containing the source boxes. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'bbox_0.png'.",
    )
    parser.add_argument(
        "--target_primitive_column",
        type=str,
        default="bbox_1.png",
        help="The column of the dataset containing the target boxes. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'bbox_1.png'.",
    )
    ##########
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to repeat the training data.",
    )

    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_reference_image",
        type=str,
        default=None,
        help="A reference image that is used during validation to initiate the model editing.",
    )
    ##########
    parser.add_argument(
        "--validation_source_primitive",
        type=str,
        default=None,
        help="Source primitive image that is used during validation to initiate the model editing.",
    )
    parser.add_argument(
        "--validation_target_primitive",
        type=str,
        default=None,
        help="Target primitive image that is used during validation to obtain the required model edit.",
    )
    ##########
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help="LoRA alpha to be used for additional scaling.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="gaussian",
        help="https://huggingface.co/docs/peft/package_reference/lora#peft.LoraConfig.init_lora_weights",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="Dropout probability for LoRA layers",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-kontext-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--drop_image_prob",
        type=float,
        default=0.1,
        help="Drop Text with a specified probability during training",
    )
    parser.add_argument(
        "--crop_prob",
        type=float,
        default=0.1,
        help="Randomly crop images with a specified probability during training",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1,
        help="Distilled Guidance Scale",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument("--use_ema", action="store_true", help="To use EMAModel")
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.995,
        help="Decay Factor for Exponential Moving Average",
    )
    parser.add_argument(
        "--use_ema_warmup", action="store_true", help="To use EMA Warmup"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=(
            'We default to the "none" weighting scheme for uniform sampling and uniform loss'
        ),
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_decouple",
        type=bool,
        default=True,
        help="Use AdamW style decoupled weight decay",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for unet params",
    )
    parser.add_argument(
        "--adam_weight_decay_text_encoder",
        type=float,
        default=1e-03,
        help="Weight decay to use for text_encoder",
    )

    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_npu_flash_attention",
        action="store_true",
        help="Enabla Flash Attention for NPU",
    )
    ##########
    parser.add_argument("--compile", action="store_true", help="To use torch.compile")
    parser.add_argument(
        "--compile_mode",
        type=str,
        choices=[
            "max-autotune",
            "reduce-overhead",
            "max-autotune-no-cudagraphs",
            "default",
        ],
    )
    parser.add_argument(
        "--comment_string",
        type=str,
        default="Training Initialized!",
        help="Information about the Training Run",
    )
    ##########

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError(
            "Specify only one of `--dataset_name` or `--instance_data_dir`"
        )

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def get_subset_count(repo_id, subset_names):
    if isinstance(subset_names, str):
        subset_names = [subset_names]

    metadata_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="metadata.json",
    )
    with open(metadata_path) as f:
        subsets = json.load(f)["subsets"]

    missing = [name for name in subset_names if name not in subsets]
    if missing:
        raise KeyError(
            f"Subset(s) {missing} not found in metadata.json. Available: {list(subsets)}"
        )

    return sum(subsets[name]["num_samples"] for name in subset_names)


##########
class CameraLoraDataset(Dataset):
    def __init__(
        self,
        instance_data_root,
        size=512,
        repeats=1,
        drop_image_prob=0.1,
        crop_prob=0.1,
    ):
        self.size = size
        self.drop_image_prob = drop_image_prob
        self.crop_prob = crop_prob
        self.custom_instance_prompts = True

        if args.dataset_name is not None:
            self.instance_data_root = None
            configs = args.dataset_config_name or [None]
            parts = [
                load_dataset(
                    args.dataset_name,
                    cfg,
                    cache_dir=args.cache_dir,
                    num_proc=args.dataset_num_proc,
                )["train"]
                for cfg in configs
            ]
            column_names = parts[0].column_names
            parts = [p.select_columns(column_names) for p in parts]
            self.dataset = parts[0] if len(parts) == 1 else concatenate_datasets(parts)

            self.image_column = (
                args.image_column if args.image_column is not None else column_names[3]
            )
            self.reference_image_column = (
                args.reference_image_column
                if args.reference_image_column is not None
                else column_names[2]
            )
            self.source_primitive_column = (
                args.source_primitive_column
                if args.source_primitive_column is not None
                else column_names[0]
            )
            self.target_primitive_column = (
                args.target_primitive_column
                if args.target_primitive_column is not None
                else column_names[1]
            )
            self.num_instance_images = get_subset_count(
                args.dataset_name, args.dataset_config_name
            )

        else:
            self.instance_data_root = Path(instance_data_root)
            jsonl_path = self.instance_data_root / "metadata.jsonl"

            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")
            if not jsonl_path.exists():
                raise ValueError(f"metadata.jsonl not found in {instance_data_root}")

            self.dataset = []
            with open(jsonl_path, "r") as f:
                for line in f:
                    self.dataset.append(json.loads(line))

            self.image_column = "image"
            self.reference_image_column = "reference_image"
            self.source_primitive_column = "source_primitive"
            self.target_primitive_column = "target_primitive"
            self.num_instance_images = len(self.dataset)

        self._length = self.num_instance_images * repeats

        self.train_transforms = v2.Compose(
            [
                v2.Resize((size, size), interpolation=v2.InterpolationMode.BILINEAR),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.5], [0.5]),
            ]
        )

        self.random_crop = v2.RandomResizedCrop(self.size)

    def __len__(self):
        return self._length

    def _apply_sync_transform(self, image):
        """Helper to ensure all images in a batch are transformed identically."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        return self.train_transforms(image)

    def __getitem__(self, index):
        idx = index % self.num_instance_images

        try:
            example_row = self.dataset[idx]
            if self.instance_data_root:

                def get_path(key):
                    return str(self.instance_data_root / example_row[key])

                instance_image = load_image(get_path(self.image_column))
                reference_image = load_image(get_path(self.reference_image_column))
                source_primitive = load_image(get_path(self.source_primitive_column))
                target_primitive = load_image(get_path(self.target_primitive_column))
            else:
                instance_image = example_row[self.image_column]
                reference_image = example_row[self.reference_image_column]
                source_primitive = example_row[self.source_primitive_column]
                target_primitive = example_row[self.target_primitive_column]

            instance_image = self._apply_sync_transform(instance_image)
            reference_image = self._apply_sync_transform(reference_image)
            source_primitive = self._apply_sync_transform(source_primitive)
            target_primitive = self._apply_sync_transform(target_primitive)

            instance_prompt = ""

            if random.random() < self.drop_image_prob:
                zero_img = torch.zeros_like(instance_image)
                reference_image, source_primitive, target_primitive = (
                    zero_img,
                    zero_img,
                    zero_img,
                )

            if random.random() < self.crop_prob:
                instance_image = self.random_crop(instance_image)
                reference_image = self.random_crop(reference_image)
                source_primitive = self.random_crop(source_primitive)
                target_primitive = self.random_crop(target_primitive)

            return {
                "instance_images": instance_image,
                "reference_images": reference_image,
                "source_primitives": source_primitive,
                "target_primitives": target_primitive,
                "instance_prompt": instance_prompt,
            }
        except Exception as e:
            print(f"Error {e}")
            new_index = random.randint(0, self._length - 1)
            return self.__getitem__(new_index)


def collate_fn(examples):
    return {
        "pixel_values": torch.stack(
            [example["instance_images"] for example in examples]
        ),
        "reference_pixels": torch.stack(
            [example["reference_images"] for example in examples]
        ),
        "source_primitives": torch.stack(
            [example["source_primitives"] for example in examples]
        ),
        "target_primitives": torch.stack(
            [example["target_primitives"] for example in examples]
        ),
        "prompts": [example["instance_prompt"] for example in examples],
    }
##########


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if hasattr(text_encoders[0], "module"):
        dtype = text_encoders[0].module.dtype
    else:
        dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    # kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    kwargs = DistributedDataParallelKwargs()

    ##########
    if args.compile:
        dynamo_plugin = TorchDynamoPlugin(
            backend="inductor",
            mode=args.compile_mode,
            fullgraph=False,
            dynamic=False,
        )
    else:
        dynamo_plugin = None

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        dynamo_plugin=dynamo_plugin,
        kwargs_handlers=[kwargs],
    )
    ##########
    
    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info(args.comment_string)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
    )

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            transformer.set_attention_backend("_native_npu")
        else:
            raise ValueError(
                "npu flash attention requires torch_npu extensions and is supported only on npu device "
            )

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype, memory_format=torch.channels_last)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]

    # now we will add new LoRA weights the transformer layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights=args.init_lora_weights,
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    if args.use_ema:
        transformer_lora_parameters = list(
            filter(lambda p: p.requires_grad, transformer.parameters())
        )
        ema_transformer = EMAModel(
            transformer_lora_parameters,
            decay=args.ema_decay,
            use_ema_warmup=args.use_ema_warmup,
            foreach=True,
            model_cls=FluxTransformer2DModel,
            model_config=transformer.config,
        )
        ema_transformer.to(accelerator.device, dtype=weight_dtype)

    ##########
    if args.compile:
        vae = torch.compile(
            vae,
            mode=args.compile_mode,
            fullgraph=False,
            dynamic=False,
        )
        text_encoder_one = torch.compile(
            text_encoder_one,
            mode=args.compile_mode,
            fullgraph=False,
            dynamic=False,
        )
        text_encoder_two = torch.compile(
            text_encoder_two,
            mode=args.compile_mode,
            fullgraph=False,
            dynamic=False,
        )
    ##########

    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights=args.init_lora_weights,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            modules_to_save = {}
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["transformer"] = model
                elif isinstance(model, type(unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(
                        model
                    )
                    modules_to_save["text_encoder"] = model
                else:
                    raise TypeError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            FluxKontextPipeline.save_lora_weights(
                os.path.join(output_dir, "transformer"),
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )

            ##########
            if args.use_ema:
                ema_transformer.store(
                    transformer_lora_parameters
                )  # Store Original Parameters to Restore Later
                ema_transformer.copy_to(
                    transformer_lora_parameters
                )  # Copy EMAModel's Parameters to Transformer Temporarily
                ema_transformer_lora_layers_to_save = get_peft_model_state_dict(
                    unwrap_model(transformer)
                )
                modules_to_save["transformer"] = unwrap_model(transformer)

                FluxKontextPipeline.save_lora_weights(
                    os.path.join(output_dir, "ema_transformer"),
                    transformer_lora_layers=ema_transformer_lora_layers_to_save,
                    **_collate_lora_metadata(modules_to_save),
                )

                ema_transformer.restore(
                    transformer_lora_parameters
                )  # Restore Original Parameters of Transformer
                torch.save(
                    ema_transformer.state_dict(),
                    os.path.join(output_dir, "ema_transformer", "ema_state.pt"),
                )  # Saves EMA State dict so that warmup and step state is preserved across resumptions without which it would restart from 0
            ##########

    def load_model_hook(models, input_dir):
        transformer_ = None
        text_encoder_one_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            else:
                raise TypeError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxKontextPipeline.lora_state_dict(
            os.path.join(input_dir, "transformer")
        )

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            transformer_, transformer_state_dict, adapter_name="default"
        )
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
        if args.train_text_encoder:
            # Do we need to call `scale_lora_layers()` here?
            _set_state_dict_into_text_encoder(
                lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_
            )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_])
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

        ##########
        if args.use_ema:
            ema_path = os.path.join(input_dir, "ema_transformer", "ema_state.pt")
            if os.path.exists(ema_path):
                ema_state_dict = torch.load(ema_path, map_location=accelerator.device)
                ema_transformer.load_state_dict(ema_state_dict)
                logger.info(f"Resumed EMA state from {ema_path}")
            else:
                logger.warning(
                    f"EMA enabled but no ema_state.pt found in {input_dir}. "
                    "EMA will restart from scratch — decay warmup will repeat."
                )
        ##########

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        if args.train_text_encoder:
            models.extend([text_encoder_one])
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(
        filter(lambda p: p.requires_grad, transformer.parameters())
    )
    if args.train_text_encoder:
        text_lora_parameters_one = list(
            filter(lambda p: p.requires_grad, text_encoder_one.parameters())
        )

    # Optimization parameters
    transformer_parameters_with_lr = {
        "params": transformer_lora_parameters,
        "lr": args.learning_rate,
    }
    if args.train_text_encoder:
        # different learning rate for text encoder and unet
        text_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize = [
            transformer_parameters_with_lr,
            text_parameters_one_with_lr,
        ]
    else:
        params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and args.optimizer.lower() != "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError(
                "To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`"
            )

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning(
                f"Learning rates were provided both for the transformer and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and DataLoaders creation:
    train_dataset = CameraLoraDataset(
        instance_data_root=args.instance_data_dir,
        size=args.resolution,
        repeats=args.repeats,
        drop_image_prob=args.drop_image_prob,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        persistent_workers=args.dataloader_num_workers > 0,
        prefetch_factor=2 if args.dataloader_num_workers > 0 else None,
    )

    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders, tokenizers, prompt, args.max_sequence_length
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                text_ids = text_ids.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds, text_ids

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        (
            instance_prompt_hidden_states,
            instance_pooled_prompt_embeds,
            instance_text_ids,
        ) = compute_text_embeddings(args.instance_prompt, text_encoders, tokenizers)

    # Clear the memory here
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
        free_memory()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.

    if not train_dataset.custom_instance_prompts:
        if not args.train_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            pooled_prompt_embeds = instance_pooled_prompt_embeds
            text_ids = instance_text_ids
        # if we're optimizing the text encoder (both if instance prompt is used for all images or custom prompts)
        # we need to tokenize and encode the batch prompts on all training steps
        else:
            tokens_one = tokenize_prompt(
                tokenizer_one, args.instance_prompt, max_sequence_length=77
            )
            tokens_two = tokenize_prompt(
                tokenizer_two,
                args.instance_prompt,
                max_sequence_length=args.max_sequence_length,
            )

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(
            len(train_dataloader) / accelerator.num_processes
        )
        num_update_steps_per_epoch = math.ceil(
            len_train_dataloader_after_sharding / args.gradient_accumulation_steps
        )
        num_training_steps_for_scheduler = (
            args.num_train_epochs
            * accelerator.num_processes
            * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = (
            args.max_train_steps * accelerator.num_processes
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        (
            transformer,
            text_encoder_one,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            transformer,
            text_encoder_one,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = args.output_dir
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    scheduler_sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device)
    scheduler_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)

    def get_sigmas(timesteps, dtype):
        step_indices = (
            (scheduler_timesteps[None] == timesteps[:, None]).int().argmax(dim=1)
        )
        return scheduler_sigmas[step_indices].to(dtype=dtype).reshape(-1, 1, 1, 1)

    def prepare_condition_latents(latents):
        packed_latents = FluxKontextPipeline._pack_latents(
            latents,
            batch_size=latents.shape[0],
            num_channels_latents=latents.shape[1],
            height=latents.shape[2],
            width=latents.shape[3],
        )
        latent_ids = FluxKontextPipeline._prepare_latent_image_ids(
            latents.shape[0],
            latents.shape[2] // 2,
            latents.shape[3] // 2,
            accelerator.device,
            weight_dtype,
        )
        return packed_latents, latent_ids

    # Handle guidance
    if unwrap_model(transformer).config.guidance_embeds:
        guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
        guidance = guidance.expand(args.train_batch_size)
    else:
        guidance = None

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            # set top parameter requires_grad = True for gradient checkpointing works
            unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            if args.train_text_encoder:
                models_to_accumulate.extend([text_encoder_one])
            with accelerator.accumulate(models_to_accumulate):
                with accelerator.autocast():
                    prompts = batch["prompts"]

                    # encode batch prompts when custom prompts are provided for each image -
                    if train_dataset.custom_instance_prompts:
                        if not args.train_text_encoder:
                            prompt_embeds, pooled_prompt_embeds, text_ids = (
                                compute_text_embeddings(
                                    prompts, text_encoders, tokenizers
                                )
                            )
                        else:
                            tokens_one = tokenize_prompt(
                                tokenizer_one, prompts, max_sequence_length=77
                            )
                            tokens_two = tokenize_prompt(
                                tokenizer_two,
                                prompts,
                                max_sequence_length=args.max_sequence_length,
                            )
                            prompt_embeds, pooled_prompt_embeds, text_ids = (
                                encode_prompt(
                                    text_encoders=[text_encoder_one, text_encoder_two],
                                    tokenizers=[None, None],
                                    text_input_ids_list=[tokens_one, tokens_two],
                                    max_sequence_length=args.max_sequence_length,
                                    device=accelerator.device,
                                    prompt=prompts,
                                )
                            )
                    else:
                        elems_to_repeat = len(prompts)
                        if args.train_text_encoder:
                            prompt_embeds, pooled_prompt_embeds, text_ids = (
                                encode_prompt(
                                    text_encoders=[text_encoder_one, text_encoder_two],
                                    tokenizers=[None, None],
                                    text_input_ids_list=[
                                        tokens_one.repeat(elems_to_repeat, 1),
                                        tokens_two.repeat(elems_to_repeat, 1),
                                    ],
                                    max_sequence_length=args.max_sequence_length,
                                    device=accelerator.device,
                                    prompt=args.instance_prompt,
                                )
                            )
                        else:
                            prompt_embeds, pooled_prompt_embeds, text_ids = (
                                compute_text_embeddings(
                                    prompts, text_encoders, tokenizers
                                )
                            )

                    # Convert images to latent space
                    pixel_values = batch["pixel_values"].to(
                        dtype=vae.dtype,
                        non_blocking=True,
                        memory_format=torch.channels_last,
                    )
                    reference_pixels = batch["reference_pixels"].to(
                        dtype=vae.dtype,
                        non_blocking=True,
                        memory_format=torch.channels_last,
                    )
                    ##########
                    source_primitives = batch["source_primitives"].to(
                        dtype=vae.dtype,
                        non_blocking=True,
                        memory_format=torch.channels_last,
                    )
                    target_primitives = batch["target_primitives"].to(
                        dtype=vae.dtype,
                        non_blocking=True,
                        memory_format=torch.channels_last,
                    )

                    all_pixels = torch.cat(
                        [
                            pixel_values,
                            reference_pixels,
                            source_primitives,
                            target_primitives,
                        ],
                        dim=0,
                    )
                    all_latents_dist = vae.encode(all_pixels).latent_dist

                    means = all_latents_dist.mean.chunk(4, dim=0)
                    logvars = all_latents_dist.logvar.chunk(4, dim=0)

                    model_input = means[0] + torch.exp(
                        0.5 * logvars[0]
                    ) * torch.randn_like(
                        means[0]
                    )  # Sample = mean + std * gaussian_noise

                    # Process Reference Image for FLUX.1-Kontext
                    # https://github.com/huggingface/diffusers/blob/16e7067647673ffe7e8d75f34c420c4f03dd542f/src/diffusers/pipelines/flux/pipeline_flux_kontext.py#L598
                    reference_latents = means[1]  # Mode = mean
                    source_primitive_latents = means[2]
                    target_primitive_latents = means[3]

                    # Vae Scale and Shift Condition Latents
                    reference_latents = (
                        reference_latents - vae_config_shift_factor
                    ) * vae_config_scaling_factor
                    source_primitive_latents = (
                        source_primitive_latents - vae_config_shift_factor
                    ) * vae_config_scaling_factor
                    target_primitive_latents = (
                        target_primitive_latents - vae_config_shift_factor
                    ) * vae_config_scaling_factor

                    # Prepare Condition Latents and IDs
                    reference_latents, reference_latent_ids = prepare_condition_latents(
                        reference_latents
                    )
                    source_primitive_latents, source_primitive_latent_ids = (
                        prepare_condition_latents(source_primitive_latents)
                    )
                    target_primitive_latents, target_primitive_latent_ids = (
                        prepare_condition_latents(target_primitive_latents)
                    )

                    # image ids are the same as latent ids with the first dimension set to 1,2,3 instead of 0
                    reference_latent_ids[..., 0] = 1
                    source_primitive_latent_ids[..., 0] = 2
                    target_primitive_latent_ids[..., 0] = 3
                    ##########

                    model_input = (
                        model_input - vae_config_shift_factor
                    ) * vae_config_scaling_factor
                    model_input = model_input.to(dtype=weight_dtype)

                    latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                        model_input.shape[0],
                        model_input.shape[2] // 2,
                        model_input.shape[3] // 2,
                        accelerator.device,
                        weight_dtype,
                    )

                    # Concatenate Latent IDs Along Sequence Length (dim=0)
                    ##########
                    latent_image_ids = torch.cat(
                        [
                            latent_image_ids,
                            reference_latent_ids,
                            source_primitive_latent_ids,
                            target_primitive_latent_ids,
                        ],
                        dim=0,
                    )
                    ##########

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]

                    # Sample a random timestep for each image
                    # for weighting schemes where we sample timesteps non-uniformly
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (
                        u * noise_scheduler_copy.config.num_train_timesteps
                    ).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(
                        device=model_input.device
                    )

                    # Add noise according to flow matching.
                    # zt = (1 - texp) * x + texp * z1
                    sigmas = get_sigmas(timesteps, dtype=model_input.dtype)
                    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                    packed_noisy_model_input = FluxKontextPipeline._pack_latents(
                        noisy_model_input,
                        batch_size=model_input.shape[0],
                        num_channels_latents=model_input.shape[1],
                        height=model_input.shape[2],
                        width=model_input.shape[3],
                    )
                    B, S, D = packed_noisy_model_input.shape
                    ##########
                    packed_noisy_model_input = torch.cat(
                        [
                            packed_noisy_model_input,
                            reference_latents,
                            source_primitive_latents,
                            target_primitive_latents,
                        ],
                        dim=1,
                    )
                    ##########

                    # Predict the noise residual
                    model_pred = transformer(
                        hidden_states=packed_noisy_model_input,
                        # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
                        timestep=timesteps / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]

                    model_pred = model_pred[
                        :, :S
                    ]  # Extracting only noisy latent from [Noisy_latents,Reference_image_latents,Source_primitive_latents,Target_primitive_latents]
                    model_pred = FluxKontextPipeline._unpack_latents(
                        model_pred,
                        height=model_input.shape[2] * vae_scale_factor,
                        width=model_input.shape[3] * vae_scale_factor,
                        vae_scale_factor=vae_scale_factor,
                    )

                    # these weighting schemes use a uniform timestep sampling
                    # and instead post-weight the loss
                    weighting = compute_loss_weighting_for_sd3(
                        weighting_scheme=args.weighting_scheme, sigmas=sigmas
                    )

                    # flow matching loss
                    target = noise - model_input

                    # Compute regular loss.
                    loss = torch.mean(
                        (
                            weighting.float()
                            * (model_pred.float() - target.float()) ** 2
                        ).reshape(target.shape[0], -1),
                        1,
                    )
                    loss = loss.mean()

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(
                                transformer.parameters(), text_encoder_one.parameters()
                            )
                            if args.train_text_encoder
                            else transformer.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_transformer.step(transformer_lora_parameters)
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if (
                args.validation_prompt is not None
                and global_step % args.validation_steps == 0
            ):
                # create pipeline
                if not args.train_text_encoder:
                    text_encoder_one, text_encoder_two = load_text_encoders(
                        text_encoder_cls_one, text_encoder_cls_two
                    )
                    text_encoder_one.to(weight_dtype)
                    text_encoder_two.to(weight_dtype)

                ##########
                if args.use_ema:
                    ema_transformer.store(
                        transformer_lora_parameters
                    )  # Store Original Parameters
                    ema_transformer.copy_to(
                        transformer_lora_parameters
                    )  # Copy EMA Parameters to Transformer
                ##########

                pipeline = FluxKontextPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    custom_pipeline=args.custom_pipeline,
                    vae=vae._orig_mod if is_compiled_module(vae) else vae,
                    text_encoder=unwrap_model(text_encoder_one),
                    text_encoder_2=unwrap_model(text_encoder_two),
                    transformer=unwrap_model(transformer),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                    trust_remote_code=True
                )
                ##########
                pipeline_args = {
                    "prompt": args.validation_prompt,
                    "image": load_image(args.validation_reference_image),
                    "source_primitive": load_image(args.validation_source_primitive),
                    "target_primitive": load_image(args.validation_target_primitive),
                }
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=pipeline_args,
                    global_step=global_step,
                    torch_dtype=weight_dtype,
                )
                ##########

                if args.use_ema:
                    ema_transformer.restore(transformer_lora_parameters)

                if not args.train_text_encoder:
                    del text_encoder_one, text_encoder_two
                    free_memory()

                images = None
                del pipeline
                free_memory()
                accelerator.wait_for_everyone()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        modules_to_save = {}
        transformer = unwrap_model(transformer)

        if args.use_ema:
            ema_transformer.copy_to(
                transformer_lora_parameters
            )  # Copy EMA Parameters to Transformer before Saving

        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        modules_to_save["transformer"] = transformer

        if args.train_text_encoder:
            text_encoder_one = unwrap_model(text_encoder_one)
            text_encoder_lora_layers = get_peft_model_state_dict(
                text_encoder_one.to(torch.float32)
            )
            modules_to_save["text_encoder"] = text_encoder_one
        else:
            text_encoder_lora_layers = None

        FluxKontextPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )

        if args.comment_string:
            with open(os.path.join(args.output_dir, "comments.txt"), "w") as f:
                f.write(args.comment_string)

        # Final inference
        # Load previous pipeline
        pipeline = FluxKontextPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            custom_pipeline=args.custom_pipeline,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
            trust_remote_code=True
        )
        # load attention processors
        pipeline.load_lora_weights(args.output_dir)

        # run inference
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline_args = {
                "prompt": args.validation_prompt,
                "image": load_image(args.validation_reference_image),
                "source_primitive": load_image(args.validation_source_primitive),
                "target_primitive": load_image(args.validation_target_primitive),
            }
            images = log_validation(
                pipeline=pipeline,
                args=args,
                accelerator=accelerator,
                pipeline_args=pipeline_args,
                global_step=global_step,
                is_final_validation=True,
                torch_dtype=weight_dtype,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                train_text_encoder=args.train_text_encoder,
                validation_prompt=args.validation_prompt,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        images = None
        del pipeline

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
