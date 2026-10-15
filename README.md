<div align="center">

# Thinking in Boxes: 3D Editing in Real Images Made Easy

### NeurIPS 2026

<img src="assets/thumbnail.png" alt="Thinking in Boxes" width="100%">

[**Pradhaan S Bhat**](https://pradhaansbhat.github.io/)<sup>1</sup><sup>∗</sup> · [**Naveen Chandra R**](https://www.linkedin.com/in/naveen-chandra-r-7230aa192)<sup>1</sup><sup>∗</sup> · [**Rishubh Parihar**](https://rishubhpar.github.io/)<sup>1</sup> · [**Vaibhav Vavilala**](https://www.linkedin.com/in/vaibhav-vavilala)<sup>2</sup> · [**R. Venkatesh Babu**](https://cds.iisc.ac.in/faculty/venky/)<sup>1</sup> · [**D. A. Forsyth**](http://luthuli.cs.uiuc.edu/~daf/)<sup>3</sup> · [**Anand Bhattad**](https://anandbhattad.github.io/)<sup>4</sup>

<sup>1</sup> Indian Institute of Science &nbsp;·&nbsp; <sup>2</sup> Apple &nbsp;·&nbsp; <sup>3</sup> UIUC &nbsp;·&nbsp; <sup>4</sup> Johns Hopkins University

<sup>∗</sup> Equal Contribution

[![arXiv](https://img.shields.io/badge/arXiv-2606.20556-b31b1b.svg)](https://arxiv.org/abs/2606.20556)
[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb.svg)](https://thinking-in-boxes.github.io)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/pradhaansbhat/Thinking-In-Boxes)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/pradhaansbhat/Thinking-In-Boxes)
[![Demo](https://img.shields.io/badge/%F0%9F%8E%AC%20Demo-blue.svg)](#-demo)

</div>

---

## 📚 Abstract

Text and 2D-conditioning interfaces provide weak, ambiguous control over spatial
transformations in image editing — particularly under large object motions and
camera changes. Prior work has used 3D primitives such as boxes, but only as loose
conditioning signals indicating approximate object location rather than specifying
the transformation.

We instead use **3D boxes as structured specifications**: the user provides the input
and output boxes of the edit, casting editing as a well-posed geometry problem. This
"thinking in boxes" interface, where each box face is color-coded to convey 3D
orientation, gives precise control over translation, rotation, scaling, and viewpoint
changes in real images while preserving scene and object identity, and recovering
previously unseen object regions. To ground transformations in scene appearance, we
introduce a **depth-aligned planar floor** as a global reference frame, shaded with
depth-aware cues.

Conditioned on this structure, an image generator produces consistent results under
large transformations. Trained in two stages — on synthetic multi-object scenes and a
small set of real-world videos from Objectron — the system generalizes to complex,
in-the-wild real images. Our method operates directly on real photographs and
substantially **outperforms recent state-of-the-art methods** on large 3D edits.

<div align="center">
  <img src="assets/results.gif" alt="Qualitative results" width="100%">
</div>

---

## 🗺️ Navigation

| Section | Description |
| :--- | :--- |
| [🛠️ Configuration](#%EF%B8%8F-configuration) | Install and set up the environment |
| [⚙️ Usage](#%EF%B8%8F-usage) | Train and run inference |
| [🤗 Demo](#-demo) | Launch the interactive Gradio demo |
| [🤝 Citation](#-citation) | Cite our work |

---

## 🛠️ Configuration

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) for other platforms.

### 2. Set up the environment

```bash
uv sync
```

### 3. Hugging Face token

Some model and dataset repositories are gated. Create an access token
([guide](https://huggingface.co/docs/hub/en/security-tokens)) and set `HF_TOKEN`
before training or inference.

<details>
<summary>📦 Dependency tree (<code>uv tree</code>)</summary>

```tree
thinking-in-boxes v0.1.0
├── accelerate v1.15.0
│   ├── huggingface-hub v1.33.0
│   │   ├── click v8.5.0
│   │   ├── filelock v4.0.12
│   │   ├── fsspec v2026.7.0
│   │   ├── hf-xet v1.7.0
│   │   ├── httpx v0.28.1
│   │   │   ├── anyio v4.15.1
│   │   │   │   ├── idna v3.20
│   │   │   │   └── typing-extensions v4.16.0
│   │   │   ├── certifi v2026.7.22
│   │   │   ├── httpcore v1.0.9
│   │   │   │   ├── certifi v2026.7.22
│   │   │   │   └── h11 v0.16.0
│   │   │   └── idna v3.20
│   │   ├── packaging v26.3
│   │   ├── pyyaml v6.0.3
│   │   ├── tqdm v4.70.1
│   │   └── typing-extensions v4.16.0
│   ├── numpy v1.26.4
│   ├── packaging v26.3
│   ├── psutil v7.2.2
│   ├── pyyaml v6.0.3
│   ├── safetensors v0.8.0
│   └── torch v2.14.1+cu130
│       ├── cuda-bindings v13.4.3
│       │   └── cuda-pathfinder v1.8.3
│       ├── cuda-toolkit[cublas, cudart, cufft, cufile, cupti, curand, cusolver, cusparse, nvjitlink, nvrtc, nvtx] v13.0.3.0
│       │   ├── nvidia-cublas v13.1.1.3 (extra: cublas)
│       │   │   └── nvidia-cuda-nvrtc v13.0.88
│       │   ├── nvidia-cuda-nvrtc v13.0.88 (extra: cublas)
│       │   ├── nvidia-cuda-runtime v13.0.96 (extra: cudart)
│       │   ├── nvidia-cufft v12.0.0.61 (extra: cufft)
│       │   │   └── nvidia-nvjitlink v13.4.92
│       │   ├── nvidia-nvjitlink v13.4.92 (extra: cufft)
│       │   ├── nvidia-cufile v1.15.1.6 (extra: cufile)
│       │   ├── nvidia-cuda-cupti v13.0.85 (extra: cupti)
│       │   ├── nvidia-curand v10.4.0.35 (extra: curand)
│       │   ├── nvidia-cublas v13.1.1.3 (extra: cusolver) (*)
│       │   ├── nvidia-cusolver v12.0.4.66 (extra: cusolver)
│       │   │   ├── nvidia-cublas v13.1.1.3 (*)
│       │   │   ├── nvidia-cusparse v12.6.3.3
│       │   │   │   └── nvidia-nvjitlink v13.4.92
│       │   │   └── nvidia-nvjitlink v13.4.92
│       │   ├── nvidia-cusparse v12.6.3.3 (extra: cusolver) (*)
│       │   ├── nvidia-nvjitlink v13.4.92 (extra: cusolver)
│       │   ├── nvidia-cusparse v12.6.3.3 (extra: cusparse) (*)
│       │   ├── nvidia-nvjitlink v13.4.92 (extra: cusparse)
│       │   ├── nvidia-nvjitlink v13.4.92 (extra: nvjitlink)
│       │   ├── nvidia-cuda-nvrtc v13.0.88 (extra: nvrtc)
│       │   └── nvidia-nvtx v13.0.85 (extra: nvtx)
│       ├── filelock v4.0.12
│       ├── fsspec v2026.7.0
│       ├── jinja2 v3.1.6
│       │   └── markupsafe v3.0.4
│       ├── networkx v3.7
│       ├── nvidia-cudnn-cu13 v9.24.0.43
│       │   └── nvidia-cublas v13.1.1.3 (*)
│       ├── nvidia-cusparselt-cu13 v0.8.1
│       ├── nvidia-nccl-cu13 v2.30.7
│       ├── nvidia-nvshmem-cu13 v3.4.5
│       ├── setuptools v79.0.1
│       ├── sympy v1.14.0
│       │   └── mpmath v1.3.0
│       ├── triton v3.8.0
│       └── typing-extensions v4.16.0
├── datasets v5.1.0
│   ├── dill v0.4.1
│   ├── filelock v4.0.12
│   ├── fsspec[http] v2026.7.0
│   │   └── aiohttp v3.14.4 (extra: http)
│   │       ├── aiohappyeyeballs v2.7.1
│   │       ├── aiosignal v1.4.0
│   │       │   ├── frozenlist v1.8.0
│   │       │   └── typing-extensions v4.16.0
│   │       ├── attrs v26.1.0
│   │       ├── frozenlist v1.8.0
│   │       ├── multidict v6.9.1
│   │       ├── propcache v0.5.4
│   │       ├── typing-extensions v4.16.0
│   │       └── yarl v1.25.1
│   │           ├── idna v3.20
│   │           ├── multidict v6.9.1
│   │           └── propcache v0.5.4
│   ├── huggingface-hub v1.33.0 (*)
│   ├── multiprocess v0.70.19
│   │   └── dill v0.4.1
│   ├── numpy v1.26.4
│   ├── packaging v26.3
│   ├── pandas v3.0.6
│   │   ├── numpy v1.26.4
│   │   └── python-dateutil v2.9.0.post0
│   │       └── six v1.17.0
│   ├── pyarrow v25.0.1
│   ├── pyyaml v6.0.3
│   ├── tqdm v4.70.1
│   └── xxhash v4.0.1
├── diffusers v0.41.0
│   ├── filelock v4.0.12
│   ├── huggingface-hub v1.33.0 (*)
│   ├── importlib-metadata v9.0.1
│   │   └── zipp v4.1.1
│   ├── numpy v1.26.4
│   ├── pillow v12.3.0
│   ├── regex v2026.9.29
│   ├── requests v2.34.2
│   │   ├── certifi v2026.7.22
│   │   ├── charset-normalizer v3.5.2
│   │   ├── idna v3.20
│   │   └── urllib3 v2.8.0
│   └── safetensors v0.8.0
├── einops v0.8.2
├── ftfy v6.3.1
│   └── wcwidth v0.9.2
├── gradio v6.30.0
│   ├── anyio v4.15.1 (*)
│   ├── brotli v1.2.0
│   ├── fastapi v0.143.1
│   │   ├── annotated-doc v0.0.5
│   │   ├── opentelemetry-api v1.45.1
│   │   │   └── typing-extensions v4.16.0
│   │   ├── pydantic v2.13.5
│   │   │   ├── annotated-types v0.8.0
│   │   │   ├── pydantic-core v2.46.5
│   │   │   │   └── typing-extensions v4.16.0
│   │   │   ├── typing-extensions v4.16.0
│   │   │   └── typing-inspection v0.4.4
│   │   │       └── typing-extensions v4.16.0
│   │   ├── starlette v1.8.0
│   │   │   ├── anyio v4.15.1 (*)
│   │   │   └── typing-extensions v4.16.0
│   │   ├── typing-extensions v4.16.0
│   │   └── typing-inspection v0.4.4 (*)
│   ├── gradio-client v2.7.2
│   │   ├── fsspec v2026.7.0
│   │   ├── httpx v0.28.1 (*)
│   │   ├── huggingface-hub v1.33.0 (*)
│   │   ├── packaging v26.3
│   │   └── typing-extensions v4.16.0
│   ├── groovy v0.1.2
│   ├── hf-gradio v0.4.1
│   │   ├── gradio-client v2.7.2 (*)
│   │   └── typer v0.27.3
│   │       ├── annotated-doc v0.0.5
│   │       ├── rich v15.0.0
│   │       │   ├── markdown-it-py v4.2.0
│   │       │   │   └── mdurl v0.1.2
│   │       │   └── pygments v2.21.0
│   │       └── shellingham v1.5.4
│   ├── httpx v0.28.1 (*)
│   ├── huggingface-hub v1.33.0 (*)
│   ├── jinja2 v3.1.6 (*)
│   ├── markupsafe v3.0.4
│   ├── numpy v1.26.4
│   ├── orjson v3.13.0
│   ├── packaging v26.3
│   ├── pandas v3.0.6 (*)
│   ├── pillow v12.3.0
│   ├── pydantic v2.13.5 (*)
│   ├── pydub v0.25.1
│   ├── python-multipart v0.0.32
│   ├── pytz v2026.5
│   ├── pyyaml v6.0.3
│   ├── safehttpx v0.1.7
│   │   └── httpx v0.28.1 (*)
│   ├── semantic-version v2.10.0
│   ├── starlette v1.8.0 (*)
│   ├── tomlkit v0.14.0
│   ├── typer v0.27.3 (*)
│   ├── typing-extensions v4.16.0
│   └── uvicorn v0.54.0
│       ├── click v8.5.0
│       └── h11 v0.16.0
├── iopath v0.1.10
│   ├── portalocker v4.4.0
│   ├── tqdm v4.70.1
│   └── typing-extensions v4.16.0
├── kernels v0.17.2
│   ├── huggingface-hub v1.33.0 (*)
│   ├── packaging v26.3
│   ├── pyyaml v6.0.3
│   ├── sigstore v4.5.0
│   │   ├── cryptography v50.0.2
│   │   │   └── cffi v2.1.1
│   │   │       └── pycparser v3.11
│   │   ├── id v1.6.1
│   │   │   └── urllib3 v2.8.0
│   │   ├── platformdirs v4.12.3
│   │   ├── pyasn1 v0.6.4
│   │   ├── pydantic v2.13.5 (*)
│   │   ├── pyjwt v2.15.1
│   │   ├── pyopenssl v26.4.0
│   │   │   ├── cryptography v50.0.2 (*)
│   │   │   └── typing-extensions v4.16.0
│   │   ├── requests v2.34.2 (*)
│   │   ├── rfc3161-client v1.0.9
│   │   │   └── cryptography v50.0.2 (*)
│   │   ├── rfc8785 v0.1.4
│   │   ├── rich v15.0.0 (*)
│   │   ├── sigstore-models v0.0.6
│   │   │   ├── pydantic v2.13.5 (*)
│   │   │   └── typing-extensions v4.16.0
│   │   ├── sigstore-rekor-types v0.0.18
│   │   │   └── pydantic v2.13.5 (*)
│   │   └── tuf v7.0.1
│   │       ├── securesystemslib v1.5.1
│   │       └── urllib3 v2.8.0
│   └── tomlkit v0.14.0
├── peft v0.21.2
│   ├── accelerate v1.15.0 (*)
│   ├── huggingface-hub v1.33.0 (*)
│   ├── numpy v1.26.4
│   ├── packaging v26.3
│   ├── psutil v7.2.2
│   ├── pyyaml v6.0.3
│   ├── safetensors v0.8.0
│   ├── torch v2.14.1+cu130 (*)
│   ├── tqdm v4.70.1
│   └── transformers v5.19.0
│       ├── huggingface-hub v1.33.0 (*)
│       ├── numpy v1.26.4
│       ├── packaging v26.3
│       ├── pyyaml v6.0.3
│       ├── regex v2026.9.29
│       ├── safetensors v0.8.0
│       ├── tokenizers v0.23.2
│       │   └── huggingface-hub v1.33.0 (*)
│       ├── tqdm v4.70.1
│       └── typer v0.27.3 (*)
├── prodigyopt v1.1.2
├── setuptools v79.0.1
├── torch v2.14.1+cu130 (*)
├── torchvision v0.29.1+cu130
│   ├── numpy v1.26.4
│   ├── pillow v12.3.0
│   └── torch v2.14.1+cu130 (*)
├── transformers v5.19.0 (*)
├── utils3d v1.7
│   ├── moderngl v5.13.0
│   │   └── glcontext v3.1.0
│   ├── numpy v1.26.4
│   └── scipy v1.17.1
│       └── numpy v1.26.4
├── vis4d v1.0.0
│   ├── absl-py v2.5.1
│   ├── appdirs v1.4.4
│   ├── bdd100k v1.0.1
│   │   ├── gmplot v1.4.1
│   │   │   └── requests v2.34.2 (*)
│   │   ├── joblib v1.6.0
│   │   │   └── cloudpickle v3.1.2
│   │   ├── matplotlib v3.11.2
│   │   │   ├── contourpy v1.3.3
│   │   │   │   └── numpy v1.26.4
│   │   │   ├── cycler v0.12.1
│   │   │   ├── fonttools v4.66.1
│   │   │   ├── kiwisolver v1.5.1
│   │   │   ├── numpy v1.26.4
│   │   │   ├── packaging v26.3
│   │   │   ├── pillow v12.3.0
│   │   │   ├── pyparsing v3.3.3
│   │   │   └── python-dateutil v2.9.0.post0 (*)
│   │   ├── motmetrics v1.4.0
│   │   │   ├── numpy v1.26.4
│   │   │   ├── pandas v3.0.6 (*)
│   │   │   ├── scipy v1.17.1 (*)
│   │   │   └── xmltodict v1.0.4
│   │   ├── numpy v1.26.4
│   │   ├── pandas v3.0.6 (*)
│   │   ├── pillow v12.3.0
│   │   ├── pycocotools v2.0.11
│   │   │   └── numpy v1.26.4
│   │   ├── scalabel v0.3.1
│   │   │   ├── boto3 v1.43.109
│   │   │   │   ├── botocore v1.43.109
│   │   │   │   │   ├── jmespath v1.1.0
│   │   │   │   │   ├── python-dateutil v2.9.0.post0 (*)
│   │   │   │   │   └── urllib3 v2.8.0
│   │   │   │   ├── jmespath v1.1.0
│   │   │   │   └── s3transfer v0.19.2
│   │   │   │       └── botocore v1.43.109 (*)
│   │   │   ├── joblib v1.6.0 (*)
│   │   │   ├── matplotlib v3.11.2 (*)
│   │   │   ├── motmetrics v1.4.0 (*)
│   │   │   ├── nanoid v2.0.0
│   │   │   ├── numpy v1.26.4
│   │   │   ├── pandas v3.0.6 (*)
│   │   │   ├── pillow v12.3.0
│   │   │   ├── plyfile v1.1.3
│   │   │   │   └── numpy v1.26.4
│   │   │   ├── psutil v7.2.2
│   │   │   ├── pycocotools v2.0.11 (*)
│   │   │   ├── pydantic v2.13.5 (*)
│   │   │   ├── pyparsing v3.3.3
│   │   │   ├── python-dateutil v2.9.0.post0 (*)
│   │   │   ├── pyyaml v6.0.3
│   │   │   ├── requests v2.34.2 (*)
│   │   │   ├── scikit-image v0.26.0
│   │   │   │   ├── imageio v2.38.0
│   │   │   │   │   ├── numpy v1.26.4
│   │   │   │   │   └── pillow v12.3.0
│   │   │   │   ├── lazy-loader v0.6
│   │   │   │   │   └── packaging v26.3
│   │   │   │   ├── networkx v3.7
│   │   │   │   ├── numpy v1.26.4
│   │   │   │   ├── packaging v26.3
│   │   │   │   ├── pillow v12.3.0
│   │   │   │   ├── scipy v1.17.1 (*)
│   │   │   │   └── tifffile v2026.3.3
│   │   │   │       └── numpy v1.26.4
│   │   │   ├── tabulate v0.10.0
│   │   │   ├── toml v0.10.2
│   │   │   └── tqdm v4.70.1
│   │   ├── scikit-image v0.26.0 (*)
│   │   ├── tabulate v0.10.0
│   │   ├── toml v0.10.2
│   │   └── tqdm v4.70.1
│   ├── cloudpickle v3.1.2
│   ├── cython v3.3.0
│   ├── devtools v0.12.2
│   │   ├── asttokens v2.4.1
│   │   │   └── six v1.17.0
│   │   ├── executing v2.2.1
│   │   └── pygments v2.21.0
│   ├── h5py v3.16.0
│   │   └── numpy v1.26.4
│   ├── jsonargparse[signatures] v4.52.0
│   │   ├── pyyaml v6.0.3
│   │   ├── docstring-parser v0.18.0 (extra: signatures)
│   │   └── typeshed-client v2.14.0 (extra: signatures)
│   │       └── typing-extensions v4.16.0
│   ├── lightning v2.6.6
│   │   ├── fsspec[http] v2026.7.0 (*)
│   │   ├── lightning-utilities v0.15.3
│   │   │   ├── packaging v26.3
│   │   │   └── typing-extensions v4.16.0
│   │   ├── packaging v26.3
│   │   ├── pytorch-lightning v2.6.6
│   │   │   ├── fsspec[http] v2026.7.0 (*)
│   │   │   ├── lightning-utilities v0.15.3 (*)
│   │   │   ├── packaging v26.3
│   │   │   ├── pyyaml v6.0.3
│   │   │   ├── torch v2.14.1+cu130 (*)
│   │   │   ├── torchmetrics v1.9.0
│   │   │   │   ├── lightning-utilities v0.15.3 (*)
│   │   │   │   ├── numpy v1.26.4
│   │   │   │   ├── packaging v26.3
│   │   │   │   └── torch v2.14.1+cu130 (*)
│   │   │   ├── tqdm v4.70.1
│   │   │   └── typing-extensions v4.16.0
│   │   ├── pyyaml v6.0.3
│   │   ├── torch v2.14.1+cu130 (*)
│   │   ├── torchmetrics v1.9.0 (*)
│   │   ├── tqdm v4.70.1
│   │   └── typing-extensions v4.16.0
│   ├── ml-collections v1.1.0
│   │   ├── absl-py v2.5.1
│   │   └── pyyaml v6.0.3
│   ├── numpy v1.26.4
│   ├── nuscenes-devkit v1.2.0
│   │   ├── cachetools v7.2.1
│   │   ├── descartes v1.1.0
│   │   │   └── matplotlib v3.11.2 (*)
│   │   ├── fire v0.7.1
│   │   │   └── termcolor v3.3.0
│   │   ├── matplotlib v3.11.2 (*)
│   │   ├── numpy v1.26.4
│   │   ├── opencv-python-headless v4.11.0.86
│   │   │   └── numpy v1.26.4
│   │   ├── parameterized v0.9.0
│   │   ├── pillow v12.3.0
│   │   ├── pycocotools v2.0.11 (*)
│   │   ├── pyquaternion v0.9.9
│   │   │   └── numpy v1.26.4
│   │   ├── scikit-learn v1.9.1
│   │   │   ├── joblib v1.6.0 (*)
│   │   │   ├── narwhals v2.27.1
│   │   │   ├── numpy v1.26.4
│   │   │   ├── scipy v1.17.1 (*)
│   │   │   └── threadpoolctl v3.7.0
│   │   ├── scipy v1.17.1 (*)
│   │   ├── shapely v2.0.7
│   │   │   └── numpy v1.26.4
│   │   └── tqdm v4.70.1
│   ├── opencv-python v4.11.0.86
│   │   └── numpy v1.26.4
│   ├── pandas v3.0.6 (*)
│   ├── pillow v12.3.0
│   ├── plyfile v1.1.3 (*)
│   ├── pycocotools v2.0.11 (*)
│   ├── pydantic v2.13.5 (*)
│   ├── scalabel v0.3.1 (*)
│   ├── scipy v1.17.1 (*)
│   ├── scipy-stubs v1.18.1.1
│   │   └── optype[numpy] v0.17.1
│   │       ├── typing-extensions v4.16.0
│   │       ├── numpy v1.26.4 (extra: numpy)
│   │       └── numpy-typing-compat v20251206.1.25 (extra: numpy)
│   │           └── numpy v1.26.4
│   ├── setuptools v79.0.1
│   ├── tensorboard v2.21.0
│   │   ├── absl-py v2.5.1
│   │   ├── grpcio v1.84.0
│   │   │   └── typing-extensions v4.16.0
│   │   ├── markdown v3.11.1
│   │   ├── numpy v1.26.4
│   │   ├── packaging v26.3
│   │   ├── pillow v12.3.0
│   │   ├── protobuf v7.36.2
│   │   ├── setuptools v79.0.1
│   │   ├── tensorboard-data-server v0.7.2
│   │   └── werkzeug v3.1.9
│   │       └── markupsafe v3.0.4
│   ├── termcolor v3.3.0
│   ├── terminaltables v3.1.10
│   ├── timm v1.0.30
│   │   ├── huggingface-hub v1.33.0 (*)
│   │   ├── pyyaml v6.0.3
│   │   ├── safetensors v0.8.0
│   │   ├── torch v2.14.1+cu130 (*)
│   │   └── torchvision v0.29.1+cu130 (*)
│   ├── torch v2.14.1+cu130 (*)
│   ├── torchvision v0.29.1+cu130 (*)
│   ├── tqdm v4.70.1
│   ├── utm v0.9.0
│   └── wheel v0.48.0
│       └── packaging v26.3
├── vis4d-cuda-ops v0.0.0
│   └── ninja v1.13.2
└── wandb v0.30.0
    ├── click v8.5.0
    ├── opentelemetry-api v1.45.1 (*)
    ├── opentelemetry-exporter-otlp-proto-http v1.45.1
    │   ├── googleapis-common-protos v1.75.5
    │   │   └── protobuf v7.36.2
    │   ├── opentelemetry-api v1.45.1 (*)
    │   ├── opentelemetry-exporter-http-transport[requests] v0.66b1
    │   │   ├── opentelemetry-api v1.45.1 (*)
    │   │   └── requests v2.34.2 (extra: requests) (*)
    │   ├── opentelemetry-exporter-otlp-common v0.66b1
    │   │   └── opentelemetry-sdk v1.45.1
    │   │       ├── opentelemetry-api v1.45.1 (*)
    │   │       ├── opentelemetry-semantic-conventions v0.66b1
    │   │       │   ├── opentelemetry-api v1.45.1 (*)
    │   │       │   └── typing-extensions v4.16.0
    │   │       └── typing-extensions v4.16.0
    │   ├── opentelemetry-exporter-otlp-proto-common v1.45.1
    │   │   └── opentelemetry-proto v1.45.1
    │   │       └── protobuf v7.36.2
    │   ├── opentelemetry-proto v1.45.1 (*)
    │   ├── opentelemetry-sdk v1.45.1 (*)
    │   ├── requests v2.34.2 (*)
    │   └── typing-extensions v4.16.0
    ├── opentelemetry-sdk v1.45.1 (*)
    ├── packaging v26.3
    ├── platformdirs v4.12.3
    ├── protobuf v7.36.2
    ├── pydantic v2.13.5 (*)
    ├── pyyaml v6.0.3
    ├── requests v2.34.2 (*)
    ├── typing-extensions v4.16.0
    └── xxhash v4.0.1
(*) Package tree already displayed
```

</details>

---

## ⚙️ Usage

All source code lives under `src/thinking_in_boxes`.

> **Tip:** Code blocks wrapped in `##########` markers indicate our changes on top of
> the original diffusers code, in `src/thinking_in_boxes/pipelines/pipeline_flux_kontext.py`
> and `src/thinking_in_boxes/train.py`.

### 📊 Data pipeline

Code and instructions for rendering your own data live under
`src/thinking_in_boxes/data/`. See the `README.md` there for details.

### 🔥 Training

Training runs in **two stages** — Stage 1 on purely synthetic data, Stage 2 on a mix of
synthetic data and a real subset from Objectron.

1. `cd src/thinking_in_boxes`
2. Set `HF_TOKEN` and `WANDB_TOKEN` in `scripts/train.sh`.
3. (Optional) Configure Accelerate for your machine: `accelerate config`.
4. Launch: `bash scripts/train.sh`.

The dataset downloads automatically on first run (preparation may take a while).
Checkpoints are saved under `Thinking-In-Boxes-R32-Stage<n>`.

### ❄️ Inference

1. `cd src/thinking_in_boxes`
2. Set `HF_TOKEN` in `scripts/inference.sh`.
3. Configure `REFERENCE_IMAGE`, `SOURCE_PRIMITIVE`, `TARGET_PRIMITIVE`, and `SAVE_DIR`.
4. Run: `bash scripts/inference.sh`.

---

## 🤗 Demo

### (Optional) Set up WildDet3D for auto-detection

The demo can auto-fit 3D boxes to objects using
[WildDet3D](https://github.com/allenai/WildDet3D). This step is optional; you can always
place boxes by hand.

```bash
git clone --recurse-submodules https://github.com/allenai/WildDet3D.git
export WILDDET3D_REPO="$(cd WildDet3D && pwd)"
```

> **Notes**
> - Detected boxes may not be perfectly accurate — we recommend refining them manually.
> - The first detection run sets up WildDet3D automatically and may take some time.
> - The editor starts with one 3D box; delete it (select it and press `−` or `Delete`) if not needed.
> - We recommend **≈48 GB VRAM** total to run the full pipeline + detector.

### Run the Gradio demo

```bash
cd src/thinking_in_boxes
uv run demo.py
```

> **Note:** Set `HF_TOKEN` and `HF_HOME` or have `black-forest-labs/FLUX.1-Kontext-dev` downloaded.


### Using the editor

1. Check the **left rail** for shortcuts and hover over icons for tooltips. A little
   familiarity with the UI makes editing much easier.
2. Fit boxes to the scene, then click the **bookmark** button to set the *source* boxes.
   Edit the boxes into the *target* pose, then click **"Grab image + boxes from editor"**.
3. Use individual edit modes, or **Free** mode, depending on your need:

   | Key   | Mode        |
   | :---- | :---------- |
   | `T`   | Translate   |
   | `R`   | Rotate      |
   | `S`   | Scale       |
   | `C`   | Camera      |
   | Floor | Edit floor  |
   | Free  | T · R · S · C |

4. **Box face colors indicate 3D orientation**:

   | Color  | Face   |
   | :----- | :----- |
   | Red    | Front  |
   | Cyan   | Left   |
   | Green  | Back   |
   | Pink   | Right  |
   | Blue   | Top    |
   | Yellow | Bottom |

5. We recommend the default **512×512** resolution, resize your images accordingly for best results.
6. For a fullscreen 3D editor, open `box-editor.html` directly and export the boxes manually.
7. **Pro tip:** Export box layouts as JSON to re-import and continue editing later.

---

## 🙏 Acknowledgements

We thank the contributors of the [diffusers](https://github.com/huggingface/diffusers)
library for their open-source work.

---

## 🤝 Citation

If you find our work useful, please consider citing:

```bibtex
@misc{bhat2026thinkingboxes3dediting,
  title         = {Thinking in Boxes: 3D Editing in Real Images Made Easy},
  author        = {Pradhaan S Bhat and Naveen Chandra R and Rishubh Parihar and Vaibhav Vavilala and R. Venkatesh Babu and D. A. Forsyth and Anand Bhattad},
  year          = {2026},
  eprint        = {2606.20556},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2606.20556}
}
```
