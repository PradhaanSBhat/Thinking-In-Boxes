# Dataset Generation

Distributed, BlenderProc-based generation of multi-view synthetic scenes with
RGB, depth, 3D bounding-box overlays and per-object segmentation masks.

Train and test are one codebase with a `--split` switch. Every behavioural
difference between them lives in
[`config/profiles.py`](config/profiles.py) and nowhere else.

---

## Setup

### Installation

Create a Python 3.10 conda env and install the host-side dependencies:

```bash
conda create -n tib_data python=3.10 -y
conda activate tib_data
pip install -r requirements.txt
```

Then run the one-time BlenderProc setup, **before** any generate or render:

```bash
blenderproc pip install tqdm
```

This downloads BlenderProc's bundled Blender and adds `tqdm` to its Python,
which `convert_assets.sh` runs in.

### Configuration

Edit one line in [`config.yaml`](config.yaml):

```yaml
paths:
  data_root: /path/to/datasets      # where assets live
```

Then fetch the assets (phase 0, once):

```bash
./scripts/download_objaverse.sh                         # ~10k models
./scripts/download_arsg_assets.sh                       # HDRIs + materials
./scripts/convert_assets.sh --prune                     # .glb -> .blend
```

Everything downloads straight into `data_root`; nothing needs moving.

**Objaverse models.** The ARSG-110K sketchfab listing holds ~163,000 objects.
The default `--world-size 16 --rank 0` fetches ~10,200 of them; pass
`--world-size 1` for all of them, `--rank N` for a different shard (one per
machine), or `--limit N` for a quick sample. Re-running resumes. Needs
`pip install objaverse pandas huggingface_hub fsspec`.

**HDRIs and materials.** `download_arsg_assets.sh` pulls the
[ARSG-110K](https://huggingface.co/datasets/HorizonRobotics/ARSG-110K) bundles
from Hugging Face (7.5 GB hdrs, 8.3 GB floor, 16 GB wall) and unzips them into
`<data_root>/ARSG-110K/`; no account is needed. Pass component names to fetch
a subset, e.g. `./scripts/download_arsg_assets.sh hdrs`. HDRIs are required —
every scene picks one for lighting. Materials are optional; without them the
floor is plain grey. Re-running resumes the download and skips components
already extracted; `KEEP_ZIPS=1` keeps the archives.

**Conversion.** The generator loads `.blend`, not `.glb`. `convert_assets.sh`
un-parents each model's meshes while keeping their world transform (glTF
files store the up-axis correction on a root node), joins them, centres the
result at the origin, normalises its size and packs its textures. The scene
generator drops every object onto the floor itself. Resumable — converted
files are skipped; `--overwrite` redoes them. `--glob "000-000/*.glb"`
converts a subset, which is handy for a quick test.

`--prune` deletes each `.glb` once its `.blend` exists, so only one copy of
each model is left on disk. It also deletes the `.glb` of every `.blend` it
*skips*, so if `converted/` holds files from an older version of the
converter (which could leave models lying on their side), run once with
`--overwrite` and without `--prune` before pruning.

### Expected layout

```
<data_root>/
├── ObjaverseXL_sketchfab/
│   ├── metadata.csv                  # download progress
│   └── converted/...                 # .blend  (raw/*.glb if not pruned)
└── ARSG-110K/
    ├── hdrs/                         # .hdr / .exr
    └── materials_{floor,wall}/       # .blend
```

Models are found in **any** top-level directory containing `converted/`.
HDRIs and materials come from the directory lists at the top of
[`generation/assets.py`](generation/assets.py).

A one-time asset index is cached as `<data_root>/.asset_index_rel.json`; it
prints how many of each kind it found. Delete it after adding assets.

---

## Running

```bash
# Train
./scripts/generate_train.sh && ./scripts/render_train.sh

# Test (both movement modes)
./scripts/generate_test.sh && ./scripts/render_test.sh
```

Generation is CPU-only and cheap, so it fans out widely. Rendering is the
GPU-bound phase. Because phase 1 records **relative** asset paths, the
metadata is portable and phase 2 is reproducible and restartable — a scene
whose `rgb_0.png` exists is skipped.

| Phase | Script | GPU? | Output |
|---|---|---|---|
| 1 | `generation/generate_scenes.py` | no | `scene_NNNNNN/metadata.json` |
| 2 | `rendering/render_scenes.py` | yes | RGB / bbox / (depth / masks) |

Useful flags: `--fast` (256px @ 16spp), `--dry-run`, `--scenes N`,
`--min-obj N` / `--max-obj N` (generate only).

Each `metadata.json` records how its objects were loaded (`object_loading`:
rotation mode, rotation baking, normalisation size). The renderer reads that
back and renders each scene with the profile of the split that **generated**
it, whatever `--split` it is given, so an object's recorded rotation is
applied exactly as it was during generation. A mismatch is reported once as
an `[INFO]` line. Older metadata without that block falls back to its
`split` field, then to `mode` (test), then to train.

---

## Configuration

Everything lives in [`config.yaml`](config.yaml). Values can reference each
other with `${...}`, so setting `data_root` or `output_root` once cascades.

**Precedence: CLI argument > `config.yaml` > built-in default.**

```bash
./scripts/render_train.sh --fast --scenes 20        # quick smoke test
./scripts/generate_train.sh --global-count 100
python3 launching/launch.py --config configs/big.yaml render --split train
```

| Section | Key settings |
|---|---|
| `paths` | `data_root` (required), `output_root`, per-split scene directories |
| `scale` | `train_start`, `train_scenes`, `test_scenes`, `num_workers` |
| `objects` | `train_min`, `train_max`, `test_min`, `test_max` — objects per scene (override with `--min-obj` / `--max-obj`) |
| `render` | `gpu_ids`, `procs_per_gpu`, `batch_size`, `fast`, `masks` |
| `assets` | `dataset`, `glb_input`, `blend_output`, `pattern` (phase 0) |
| `ablations` | One block per single-factor dataset |

`data_root` has no default: if it is left as the shipped placeholder the
scripts stop and say so. `$DATA_ROOT` works as a fallback. PyYAML is used when
installed; a built-in parser covers this file otherwise, so the pipeline works
where BlenderProc's bundled Python lacks PyYAML.

---

## Splits

|  | `--split train` | `--split test` |
|---|---|---|
| Objects per scene (default, see `objects` in `config.yaml`) | 4 | 1 |
| Normalisation target size | 3.0 | 2.5 |
| Default mode | `mixed` | `camera_movement` |
| Per-view variation | camera moves **and** a random object is perturbed | exactly one factor, per mode |
| Translation / scaling | enabled | disabled by default |
| Force `rotation_mode='XYZ'` | no | **yes** |
| Bake rotation on normalise | no | **yes** |
| Objaverse UID + name | no | yes |
| 3D bbox in metadata | no | yes |
| Per-object masks | off by default | **on** by default |

> **Do not "harmonise" these two profiles.** The values reproduce the datasets
> as originally generated. The test split forces `rotation_mode='XYZ'` because
> many Objaverse `.blend` files ship with quaternion rotation, in which case
> assignments to `rotation_euler` are silently ignored. Applying that fix to
> the train profile would change train geometry.

**Movement modes.** `mixed` (train): camera moves per view, one random object
perturbed — only while translation or scaling is enabled; with both off the
objects stay fixed. `camera_movement`: camera moves, objects fixed.
`object_movement`: camera fixed, objects perturbed.

---

## Outputs

Per scene, per view `v`, per object `o`:

| File | Always? | Description |
|---|---|---|
| `metadata.json` | yes | Scene description from phase 1 (relative paths) |
| `rgb_{v}.png` | yes | HDRI-lit RGB render |
| `bbox_{v}.png` | yes | 3D bounding-box overlay on a checkerboard floor |
| `view_{v}.json` | yes | Camera intrinsics/extrinsics + per-object metadata |
| `depth_{v}.png` | masks only | Depth, uint16 millimetres |
| `mask_{v}_{o}.png` | masks only | Per-object bbox-face mask |
| `obj_mask_{v}_{o}.png` | masks only | Pixel-accurate per-object mask |
| `bbox_coords_{v}.json` | masks only | Tight 2D boxes from full silhouettes |

Masks are opt-in because they are expensive: each object is rendered solo, so
a scene with *N* objects costs *N* extra renders per view.

```bash
./scripts/render_train.sh --with-masks     # in the same pass
./scripts/render_masks_only.sh             # a separate pass over rendered scenes
```

`--masks-only` needs `rgb_*.png` and `bbox_*.png` to exist; if they are
missing the launcher offers a full render instead (`--yes` to accept
non-interactively). Set `render.masks: true` to make masks the default;
`--no-masks` forces them off.

`obj_mask_{v}_{o}.png` is pixel-accurate rather than a projected box: each
object is rendered alone with transparent film and its depth compared against
the full-scene depth. Where the two agree the object is the closest surface
and therefore visible.

---

## Distributing work

**`launching/launch.py`** — multi-GPU rendering and batched generation on one
machine. It keeps exactly `max_concurrent` workers busy, starting the next as
soon as one finishes, so no GPU idles at a wave boundary.

**`pipelines/distribute_generate.py`** — CPU-only phase-1 fan-out across
machines. Scene indices are **global**: give each machine a disjoint
`--global-start` / `--global-count` and numbering stays consistent cluster-wide.

```bash
# machine A                              # machine B
./scripts/generate_train.sh \            ./scripts/generate_train.sh \
    --global-start 0 --global-count 20000    --global-start 20000 --global-count 20000
```

---

## Ablations

Single-factor test sets that isolate translation, scaling and rotation.
Declared in `config.yaml`:

```yaml
ablations:
  only_rotate:
    scenes: 100
    translation: false
    scaling: false
    rotation: true
```

```bash
./scripts/generate_ablations.sh                  # all of them
./scripts/generate_ablations.sh only_rotate      # just one
./scripts/generate_ablations.sh --skip-render    # metadata only
./scripts/generate_ablations.sh --dry-run
```

Add a block to run a new combination — no code change needed. Unrecognised
factor flags are rejected up front, since a typo would otherwise yield a whole
dataset with the wrong factors enabled.

---

## Layout

One package per concern; each holds its own entry-point script, a thin wrapper
that hands over to the package.

```
config/       settings, split profiles, quality presets, argv scanning
blender/      Blender/BlenderProc helpers shared by both phases
generation/   phase 1 — placement, camera search, metadata.json
rendering/    phase 2 — RGB / bbox / depth / mask outputs
launching/    multi-GPU worker pool
pipelines/    asset conversion, CPU fan-out, ablation sets
preparation/  phase 0 — downloading the raw assets
scripts/      shell wrappers; all forward extra arguments
```

Producing `bbox_{v}.png` spans several modules: `blender/floor.py` builds the
checkerboard strips, `blender/world.py` switches to the flat grey world,
`rendering/scene.py` hides the objects, `blender/renderer.py` renders the
pass, `rendering/bbox.py` draws the coloured boxes, and `rendering/passes.py`
sequences it and writes the PNG.

`preparation/meshes/` is vendored from 3D-Fixer / TRELLIS — cloning either
repository is no longer required. See
[`preparation/THIRD_PARTY_NOTICES.md`](preparation/THIRD_PARTY_NOTICES.md).

---

## Requirements

Installation steps are under [Setup → Installation](#installation); the full
list is in [`requirements.txt`](requirements.txt).

BlenderProc (provides `blenderproc run` and a bundled Blender), plus `numpy`,
`opencv-python`, `imageio` and `tqdm`. `PyYAML` is recommended but optional.
The `objaverse` package (plus `pandas`, `huggingface_hub`, `fsspec`) is needed
only by the model download, and is also imported lazily to resolve
human-readable object names for the test split — lookup failures degrade to
`null`. Rendering assumes NVIDIA GPUs: Cycles is configured for `OPTIX` with
GPU compute and OptiX denoising.
