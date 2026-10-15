import argparse
import inspect
import json
import os
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from diffusers import DiffusionPipeline
from PIL import Image
from tqdm import tqdm

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "half": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
}


def parse_args():
    parser = argparse.ArgumentParser()
    # Model arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model ID from https://huggingface.co",
    )
    parser.add_argument(
        "--pipeline_path",
        type=str,
        default=None,
        help="Custom pipeline path for Dispersive Inference",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Devices to map the components of the pipeline",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="To offload models that are not currently being used to the CPU. Useful to decrease memory utilization",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Compile model for inference speed up"
    )
    parser.add_argument("--local", action="store_true", help="Local file loading")
    parser.add_argument(
        "--vae_tiling",
        action="store_true",
        help="""Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images""",
    )
    parser.add_argument(
        "--vae_slicing",
        action="store_true",
        help="""Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """,
    )

    # Prompt arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt to guide Image generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt to be used for classifier-free guidance",
    )

    # Device and Dtype
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to store tensors on"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float", "float16", "half", "bfloat16", "double"],
        help="Data type of tensors",
    )

    # Inference arguments
    parser.add_argument(
        "--save_dir", default=None, type=str, help="Path to store the generated images"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Distilled Guidance Scale",
    )
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=1.0,
        help="When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.",
    )
    parser.add_argument(
        "--height", type=int, default=512, help="Height of image generated"
    )
    parser.add_argument(
        "--width", type=int, default=512, help="Width of image generated"
    )
    parser.add_argument(
        "--max_area", type=int, default=512**2, help="Area of the image generated"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help="Number of inference timesteps for diffusion",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Batch inference for N images",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Max sequence length of T5 embeddings. 512 for Flux, 256/77 for SD3.5, 77 for SD3",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for replicatable generations, -1 for random seed",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Prints Inference arguments on console"
    )
    parser.add_argument(
        "--total", type=int, default=1, help="Total number of images to generate"
    )

    # Camera LoRA arguments
    parser.add_argument("--lora_adapter_path", type=str, help="LoRA Adapter Path")
    parser.add_argument(
        "--weight_name",
        type=str,
        default="pytorch_lora_weights.safetensors",
        help="Name of the weight in the path",
    )
    parser.add_argument("--adapter_name", type=str, help="LoRA Adapter Name")
    parser.add_argument(
        "--scale", type=float, default=1.0, help="LoRA Scale for inference"
    )

    # Conditioning arguments
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default=None,
        help="Local path to folder storing multiple images and metadata or Huggingface dataset path",
    )
    parser.add_argument("--image", type=str, help="Source Reference Image")
    parser.add_argument("--source_primitive", type=str, help="Source Primitive")
    parser.add_argument("--target_primitive", type=str, help="Target Primitive")

    args = parser.parse_args()
    return args


def load(args):
    if args.device_map is None:
        # Load pipeline
        pipe = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            custom_pipeline=args.pipeline_path,
            local_files_only=args.local,
            dtype = args.dtype,
            trust_remote_code=True
        )

        if args.lora_adapter_path:
            pipe.transformer.load_lora_adapter(
                args.lora_adapter_path,
                weight_name=args.weight_name,
                adapter_name=args.adapter_name,
            )
            print(
                f"\033[31mLoRA Adapter {args.adapter_name} initialized using weights from {args.lora_adapter_path}!\033[0m"
            )

        if args.offload:
            # Moves unused models (text_encoders) to RAM for memory optimization as transformer + text_encoders dont fit within 24GB VRAM
            pipe.enable_model_cpu_offload(device=args.device)
        else:
            # If the transformer + text_encoders fit within 24GB VRAM as in the case of SD3.5, keep all tensors on the same GPU
            pipe = pipe.to(args.device)
    else:
        pipe = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            custom_pipeline=args.pipeline_path,
            device_map=args.device_map,
            local_files_only=args.local,
            dtype=args.dtype,
            trust_remote_code=True
        )

        if args.lora_adapter_path:
            pipe.transformer.load_lora_adapter(
                args.lora_adapter_path,
                weight_name=args.weight_name,
                adapter_name=args.adapter_name,
            )
            print(
                f"LoRA Adapter {args.adapter_name} initialized using weights from {args.lora_adapter_path}!"
            )

    if args.compile:
        pipe.transformer.compile_repeated_blocks(fullgraph=False, dynamic=False)

    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    if args.vae_tiling:
        pipe.vae.enable_vae_tiling()

    if args.vae_slicing:
        pipe.vae.enable_vae_slicing()

    return pipe


def load_data(args):

    ref_image_exists = args.image and os.path.exists(args.image)
    src_primitive_exists = args.source_primitive and os.path.exists(
        args.source_primitive
    )
    target_primitive_exists = args.target_primitive and os.path.exists(
        args.target_primitive
    )

    if ref_image_exists:
        image = Image.open(args.image).convert("RGB") if ref_image_exists else None
        source_primitive = (
            Image.open(args.source_primitive).convert("RGB")
            if src_primitive_exists
            else None
        )
        target_primitive = (
            Image.open(args.target_primitive).convert("RGB")
            if target_primitive_exists
            else None
        )
        prompt = args.prompt
        dataset = Dataset.from_dict(
            {
                "image": [image],
                "source_primitive": [source_primitive],
                "target_primitive": [target_primitive],
                "prompt": [prompt],
            }
        )
        return dataset

    elif args.dataset_name_or_path:
        if os.path.isdir(args.dataset_name_or_path):
            print(f"Loading local directory:{args.dataset_name_or_path}")
            return load_dataset(
                "imagefolder", data_dir=args.dataset_name_or_path, split='train'
            )

        elif os.path.isfile(args.dataset_name_or_path):
            print(f"Loading local file:{args.dataset_name_or_path}")
            extension = os.path.splitext(args.dataset_name_or_path)[1].lstrip(".")
            return load_dataset(extension, data_files=args.dataset_name_or_path, split='train')

        else:
            print(
                f"Attempting to load dataset from Huggingface hub:{args.dataset_name_or_path}"
            )
            return load_dataset(args.dataset_name_or_path, split='test')

    else:
        raise ValueError("You must provide either a dataset path or a image paths.")


def call_pipe(pipe, **kwargs):
    sig = inspect.signature(pipe.__call__)
    valid_args = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return pipe(**valid_args).images


def generate(pipe, args):
    save_path = Path(f"{save_dir}")
    save_path.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print("Model:", args.pretrained_model_name_or_path)
        print("Seed:", args.seed)
        print("Prompt:", args.prompt)
        print("Negative Prompt:", args.negative_prompt)
        print("Guidance Scale:", args.guidance_scale)
        print("Dataset Provided:", args.dataset_name_or_path is not None)
        print("Source Image Provided:", args.image is not None)
        print("Source Primitive Provided:", args.source_primitive is not None)
        print("Target Primitive Provided:", args.target_primitive is not None)

    dataset = load_data(args)
    joint_attention_kwargs = {"scale": args.scale}

    for idx, sample in enumerate(dataset):
        args.image = sample["image"]
        args.prompt = sample["prompt"]
        args.source_primitive = sample["source_primitive"]
        args.target_primitive = sample["target_primitive"]

        sample_save_path = save_path / f"{idx}"
        sample_save_path.mkdir(parents=True, exist_ok=True)

        current = len([f for f in sample_save_path.iterdir() if f.suffix == ".png"])
        progress_bar = tqdm(
            range(current, args.total),
            initial=current,
            desc=f"Generating images for '{args.prompt}'",
            file=sys.stdout,
        )

        generator_state_path = sample_save_path / "generator_state.pt"
        if generator_state_path.exists():
            print(f"Resuming RNG from {generator_state_path}")
            generator = torch.Generator()
            generator.set_state(torch.load(generator_state_path))
        else:
            generator = torch.Generator().manual_seed(args.seed)
            
        args.generator = generator

        while current < args.total:
            images = call_pipe(
                pipe, **vars(args), joint_attention_kwargs=joint_attention_kwargs
            )

            for _, image in enumerate(images):
                if current >= args.total:
                    break
                image.save(f"{sample_save_path}/{current}.png")
                current += 1
                progress_bar.update(1)
                sys.stdout.flush()

            torch.save(generator.get_state(), generator_state_path)


if __name__ == "__main__":
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if args.seed == -1:
        args.seed = random.randint(0, sys.maxsize)

    with open(f"{args.save_dir}/args.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)

    assert args.dtype in DTYPE_MAP, (
        f"Invalid dtype '{args.dtype}' provided. Choose from: {list(DTYPE_MAP.keys())}"
    )
    args.dtype = DTYPE_MAP[args.dtype]

    pipe = load(args)

    print("--------Generating---------")
    generate(pipe, args)
    print("--------Finished Generations--------")
