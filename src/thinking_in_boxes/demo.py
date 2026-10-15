"""Thinking in Boxes — unified Gradio app.

Embeds box-editor.html (fit boxes -> set source -> edit to target) and runs the
box-conditioned editing pipeline on (reference image, source boxes, target boxes,
prompt). Pipeline loading + call mirror the standalone inference script.

All model / pipeline settings are entered in the UI ("Model" panel) and loaded
on demand with the "Load model" button.
"""
import argparse
import base64
import inspect
import io
import json
import os
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

HF_ENDPOINT = "https://huggingface.co"
os.environ["HF_ENDPOINT"] = HF_ENDPOINT

import gradio as gr
import numpy as np
import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI
from fastapi.responses import FileResponse
from PIL import Image

FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<polygon points="16,3 29,10 29,22 16,29 3,22 3,10" fill="#C0604A" opacity="0.15"/>'
    '<polygon points="16,3 29,10 16,17 3,10" fill="#C0604A" opacity="0.55"/>'
    '<polygon points="3,10 16,17 16,29 3,22" fill="#C0604A" opacity="0.85"/>'
    '<polygon points="29,10 16,17 16,29 29,22" fill="#C0604A" opacity="0.65"/>'
    '<polygon points="16,3 29,10 29,22 16,29 3,22 3,10" fill="none" stroke="#C0604A" stroke-width="1.2"/>'
    '<line x1="16" y1="3" x2="16" y2="17" stroke="#C0604A" stroke-width="1" opacity="0.7"/>'
    '<line x1="3" y1="10" x2="16" y2="17" stroke="#C0604A" stroke-width="1" opacity="0.7"/>'
    '<line x1="29" y1="10" x2="16" y2="17" stroke="#C0604A" stroke-width="1" opacity="0.7"/></svg>'
)
HERE = Path(__file__).resolve().parent
EDITOR_HTML = HERE / "box-editor.html"   # the box editor at the repo root

# ── Defaults (auto-loaded + cached at startup). These mirror inference.sh; 
DEFAULT_MODEL      = "black-forest-labs/FLUX.1-Kontext-dev"
DEFAULT_PIPELINE   = "pipelines/pipeline_flux_kontext.py"
DEFAULT_DTYPE      = "bfloat16"
DEFAULT_DEVICE     = "cuda"
DEFAULT_DEVICE_MAP = "cuda"
DEFAULT_LORA       = "pradhaansbhat/Thinking-In-Boxes"
DEFAULT_WEIGHT     = "pytorch_lora_weights.safetensors"
DEFAULT_ADAPTER    = "100K+20K-LoRA"
DEFAULT_LOCAL      = False
DEFAULT_COMPILE    = False   # off by default (slow startup)
DEFAULT_FA3        = True    # flash-attn-3 backend, on by default
# generation defaults (inference.sh)
DEFAULT_STEPS    = 28
DEFAULT_GUIDANCE = 3.5
DEFAULT_TRUE_CFG = 1.0
DEFAULT_SCALE    = 1.0       # LoRA scale
DEFAULT_HEIGHT   = 512
DEFAULT_WIDTH    = 512
DEFAULT_MAX_SEQ  = 512

WILDDET3D_HF_REPO = "allenai/WildDet3D"
WILDDET3D_WEIGHT  = "wilddet3d_alldata_all_prompt_v1.0.pt"
WILDDET3D_REPO = os.environ.get("WILDDET3D_REPO", "")   # WildDet3D checkout (blank → rely on import path)
WILDDET3D_CKPT = os.environ.get("WILDDET3D_CKPT", "")   # optional local ckpt override (else HF Hub)
WILDDET3D_SAM3 = os.environ.get("WILDDET3D_SAM3", "")
WILDDET3D_SCORE = 0.3
FORCE_DETECT_MOCK = False   # set True to force the offline mock detector

GRADIO_TEMP_DIR = str(Path(__file__).resolve().parent.parent / ".gradio_tmp")
os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR


def _reset_gradio_tmp():
    """Clear + recreate the Gradio temp dir so stale/unwritable files don't linger."""
    d = Path(GRADIO_TEMP_DIR)
    try:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        print(f"[tib] gradio temp dir: {d}", flush=True)
    except Exception as e:
        print(f"[tib] could not reset gradio temp dir {d}: {e}", flush=True)

DTYPE_MAP = {
    "bfloat16": torch.bfloat16, "float16": torch.float16, "half": torch.float16,
    "float32": torch.float32, "float": torch.float32, "float64": torch.float64, "double": torch.float64,
}

# The loaded pipeline lives here (one per running process).
STATE = {"pipe": None, "device": "cuda", "max_sequence_length": 512}


# ─────────────────────────── pipeline loading (mirrors the inference script) ──
def pipe_signature(a):
    """Fields whose change requires reloading the pipeline (max_sequence_length excluded)."""
    return (a.pretrained_model_name_or_path, a.pipeline_path, a.dtype, a.device, a.device_map,
            a.offload, a.local, a.compile, a.vae_tiling, a.vae_slicing,
            a.lora_adapter_path, a.adapter_name, a.weight_name)


def load_pipe(a):
    dtype = DTYPE_MAP[a.dtype]
    if a.device_map:
        pipe = DiffusionPipeline.from_pretrained(
            a.pretrained_model_name_or_path, custom_pipeline=a.pipeline_path or None, trust_remote_code=True,
            device_map=a.device_map, local_files_only=a.local, torch_dtype=dtype, token=a.token or None)
    else:
        pipe = DiffusionPipeline.from_pretrained(
            a.pretrained_model_name_or_path, custom_pipeline=a.pipeline_path or None, trust_remote_code=True,
            local_files_only=a.local, token=a.token or None).to(dtype=dtype)
    if a.lora_adapter_path:
        pipe.transformer.load_lora_adapter(
            a.lora_adapter_path, weight_name=a.weight_name or None, adapter_name=a.adapter_name or None)
        print(f"LoRA adapter {a.adapter_name} loaded from {a.lora_adapter_path}")
    if not a.device_map:
        pipe.enable_model_cpu_offload(device=a.device) if a.offload else pipe.to(a.device)
    if DEFAULT_FA3:   # flash-attn-3 backend (matches inference.py); best-effort so a missing FA3 won't kill the load
        try:
            pipe.transformer.set_attention_backend("_flash_3_hub")
        except Exception as e:
            print(f"[tib] flash-attn-3 backend unavailable ({e}); using the default attention backend.")
    if a.compile:
        pipe.transformer.compile_repeated_blocks(fullgraph=False, dynamic=False)
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)
    if a.vae_tiling:  pipe.vae.enable_vae_tiling()
    if a.vae_slicing: pipe.vae.enable_vae_slicing()
    return pipe


def call_pipe(pipe, **kwargs):
    sig = inspect.signature(pipe.__call__)
    valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return pipe(**valid).images


def dataurl_to_pil(u):
    if not u:
        return None
    if "," in u:
        u = u.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(u))).convert("RGB")


# ── grab image + source/target boxes from the embedded editor (runs in browser) ──
GRAB_JS = """
() => {
  const f = document.getElementById('tib-editor');
  if (!f || !f.contentWindow) return ["", "", "", "⚠ Editor not loaded yet."];
  let st;
  try { st = f.contentWindow.getSourceTarget(); }
  catch (e) { return ["", "", "", "⚠ Editor not ready: " + e.message]; }
  if (!st || st.error) return ["", "", "", "⚠ " + ((st && st.error) || "no data")];
  return [st.image || "", st.source, st.target, "✓ Grabbed image + source + target from the editor."];
}
"""

# ── WildDet3D bridge (no endpoint): editor postMessage → hidden button → Python → editor ──
DETECT_BRIDGE_JS = """
() => {
  if (window.__tibDetectBridge) return;
  window.__tibDetectBridge = true;
  const setVal = (el, v) => {
    const proto = el.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype
                                            : window.HTMLInputElement.prototype;
    const d = Object.getOwnPropertyDescriptor(proto, 'value');
    d && d.set ? d.set.call(el, v) : (el.value = v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  };
  window.addEventListener('message', (e) => {
    const d = e.data; if (!d || d.type !== 'tib-detect') return;
    const box = document.querySelector('#tib-detect-payload');
    const ta = box && (box.querySelector('textarea') || box.querySelector('input'));
    if (!ta) return;
    setVal(ta, JSON.stringify(d));
    setTimeout(() => { const b = document.querySelector('#tib-detect-trigger');
      if (b) (b.querySelector('button') || b).click(); }, 50);
  });
}
"""

# runs after detect_objects returns; posts detections (or error) back to the editor iframe
DETECT_RESULT_JS = """
(resultJson) => {
  const f = document.getElementById('tib-editor');
  if (!f || !f.contentWindow) return;
  let p; try { p = JSON.parse(resultJson || '{}'); } catch (e) { p = { error: 'bad detector result' }; }
  const msg = (p && p.error) ? { type: 'tib-detect-result', error: p.error }
                             : { type: 'tib-detect-result', detections: p };
  try { f.contentWindow.postMessage(msg, '*'); } catch (e) {}
}
"""


# ─────────────────────────────────────────────────────────────────────── UI ──
def load_model(model_path, pipeline_path, dtype, device, device_map, offload, local, compile_,
               vae_tiling, vae_slicing, token, lora_path, adapter_name, weight_name, max_seq,
               progress=gr.Progress(track_tqdm=True)):
    if not (model_path or "").strip():
        raise gr.Error("Enter a --pretrained_model_name_or_path first.")
    a = SimpleNamespace(
        pretrained_model_name_or_path=model_path.strip(), pipeline_path=(pipeline_path or "").strip(),
        dtype=dtype, device=(device or "cuda").strip(), device_map=(device_map or "").strip(),
        offload=offload, local=local, compile=compile_, vae_tiling=vae_tiling, vae_slicing=vae_slicing,
        token=(token or "").strip(), lora_adapter_path=(lora_path or "").strip(),
        adapter_name=(adapter_name or "").strip(), weight_name=(weight_name or "").strip())
    # Skip a reload when nothing that affects the pipeline changed (it's already cached).
    sig = pipe_signature(a)
    if STATE["pipe"] is not None and STATE.get("cfg_sig") == sig:
        STATE["max_sequence_length"] = int(max_seq)
        return f"✓ Already loaded (cached): {model_path}", gr.update(open=False)
    STATE["pipe"] = load_pipe(a)
    STATE["cfg_sig"] = sig
    STATE["device"] = a.device if not a.device_map else "cuda"
    STATE["max_sequence_length"] = int(max_seq)
    msg = f"✓ Model loaded: {model_path}" + (f" · LoRA {adapter_name}" if lora_path else "")
    return msg, gr.update(open=False)   # collapse the Model panel once loaded


def grab_show(status, src_u, tgt_u):
    """Show source/target box images in the panels + a transient toast (no persistent text)."""
    msg = (status or "").strip()
    if msg.startswith("⚠") or not (src_u and tgt_u):
        gr.Warning(msg.lstrip("⚠ ").strip() or "Set a source, edit the boxes, then grab.")
    else:
        gr.Info("✓ Grabbed image + source/target boxes from the editor.")
    return dataurl_to_pil(src_u), dataurl_to_pil(tgt_u)


def generate(image_u, src_u, tgt_u, steps, guidance, true_cfg, lora_scale,
             randomize, height, width, num_images, progress=gr.Progress(track_tqdm=True)):
    if STATE["pipe"] is None:
        raise gr.Error("Load a model first (Model Settings → Load Model & Pipeline).")
    src = dataurl_to_pil(src_u); tgt = dataurl_to_pil(tgt_u)
    if src is None or tgt is None:
        raise gr.Error("Grab the source + target boxes from the editor first.")
    image = dataurl_to_pil(image_u)
    seed = random.randint(0, 2**31 - 1) if randomize else STATE.get("last_seed", 0)
    STATE["last_seed"] = seed
    generator = torch.Generator(device=STATE["device"]).manual_seed(seed)
    images = call_pipe(
        STATE["pipe"],
        image=image, source_primitive=src, target_primitive=tgt,
        prompt="", negative_prompt=None,
        guidance_scale=float(guidance), true_cfg_scale=float(true_cfg),
        num_inference_steps=int(steps), num_images_per_prompt=int(num_images),
        height=int(height), width=int(width), max_area= int(height) * int(width),
        max_sequence_length=STATE["max_sequence_length"], generator=generator,
        joint_attention_kwargs={"scale": float(lora_scale)},
    )
    return list(images), f"seed {seed}"


# ── website theme: cream bg, terracotta accent, Cormorant serif + Outfit sans ──
TERRACOTTA = gr.themes.colors.Color(
    name="terracotta", c50="#faf1ee", c100="#f3d9d0", c200="#e8b5a6", c300="#d98f78",
    c400="#cf7458", c500="#c0604a", c600="#a5503d", c700="#7b4f3a", c800="#5f3b2c",
    c900="#432a20", c950="#2a1a14")

THEME = gr.themes.Soft(
    primary_hue=TERRACOTTA, secondary_hue=TERRACOTTA, neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Outfit"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    # light + dark both use the website's cream / terracotta palette
    body_background_fill="#F6F3EE", body_background_fill_dark="#F6F3EE",
    background_fill_primary="#FFFFFF", background_fill_primary_dark="#FFFFFF",
    background_fill_secondary="#EDEAE3", background_fill_secondary_dark="#EDEAE3",
    block_background_fill="#FFFFFF", block_background_fill_dark="#FFFFFF",
    block_border_color="#D8D4CB", block_border_color_dark="#D8D4CB", block_radius="12px",
    block_label_text_color="#6B6560", block_label_text_color_dark="#6B6560",
    block_title_text_color="#1A1714", block_title_text_color_dark="#1A1714",
    border_color_primary="#D8D4CB", border_color_primary_dark="#D8D4CB",
    button_primary_background_fill="#C0604A", button_primary_background_fill_dark="#C0604A",
    button_primary_background_fill_hover="#7B4F3A", button_primary_background_fill_hover_dark="#7B4F3A",
    button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#FFFFFF",
    button_primary_border_color="#C0604A", button_primary_border_color_dark="#C0604A",
    button_secondary_background_fill="#FFFFFF", button_secondary_background_fill_dark="#FFFFFF",
    button_secondary_background_fill_hover="#EDEAE3", button_secondary_background_fill_hover_dark="#EDEAE3",
    button_secondary_border_color="#D8D4CB", button_secondary_border_color_dark="#D8D4CB",
    button_secondary_text_color="#1A1714", button_secondary_text_color_dark="#1A1714",
    body_text_color="#1A1714", body_text_color_dark="#1A1714",
    body_text_color_subdued="#6B6560", body_text_color_subdued_dark="#6B6560",
    input_background_fill="#FFFFFF", input_background_fill_dark="#FFFFFF",
    input_border_color="#D8D4CB", input_border_color_dark="#D8D4CB",
)

FORCE_LIGHT = None   # (theme is forced via CSS variables below, works in light + dark)

# Override Gradio's CSS variables directly so the cream/terracotta website theme
# applies regardless of the browser's dark-mode preference.
_VARS = """
  --body-background-fill:#F6F3EE; --background-fill-primary:#FFFFFF; --background-fill-secondary:#EDEAE3;
  --block-background-fill:#FFFFFF; --block-border-color:#D8D4CB; --border-color-primary:#D8D4CB;
  --border-color-accent-subdued:#e6c3ba; --panel-background-fill:#FFFFFF;
  --body-text-color:#1A1714; --body-text-color-subdued:#6B6560;
  --block-label-text-color:#C0604A; --block-label-background-fill:#F1EBE3; --block-title-text-color:#C0604A;
  --button-primary-background-fill:#C0604A; --button-primary-background-fill-hover:#7B4F3A;
  --button-primary-text-color:#FFFFFF; --button-primary-border-color:#C0604A;
  --button-secondary-background-fill:#FFFFFF; --button-secondary-background-fill-hover:#EDEAE3;
  --button-secondary-border-color:#D8D4CB; --button-secondary-text-color:#1A1714;
  --input-background-fill:#FFFFFF; --input-border-color:#D8D4CB; --neutral-800:#3a352f;
  --color-accent:#C0604A; --color-accent-soft:#f3d9d0; --link-text-color:#C0604A;
  --checkbox-background-color:#FFFFFF; --checkbox-border-color:#C0604A;
  --checkbox-background-color-selected:#C0604A; --checkbox-border-color-selected:#C0604A;
  --checkbox-background-color-focus:#C0604A; --checkbox-border-color-focus:#C0604A;
"""
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500;1,600&display=swap');
:root, .gradio-container, gradio-app, .dark { __VARS__ }
body, .gradio-container, gradio-app, .dark { background:#F6F3EE !important; color:#1A1714 !important; }
/* full-bleed: use the whole width of the screen */
.gradio-container{max-width:100%!important;width:100%!important;margin:0!important;padding:0 28px 28px!important}
/* Gradio 6 caps the inner .main.fillable at 1536px — lift it for true full-bleed */
.gradio-container .main, .main.fillable, .fillable.app, .wrap.svelte-zxu34v, main.contain{max-width:100%!important}
.dark input, .dark textarea{background:#FFFFFF!important;color:#1A1714!important}
/* every field label (textbox / dropdown / slider / number) uses the SAME scheme as the
   Source/Target box labels — terracotta text on a cream pill — in light AND dark mode */
.gradio-container .block > label > span:first-child,
.gradio-container .block .head > label > span:first-child,
.gradio-container .block > .container > span:first-child,
.gradio-container .block .wrap > .head span{color:#C0604A!important;background:#F1EBE3!important}
.gradio-container label > span{color:#C0604A}
/* keep any accent/terracotta surface text off pure black */
button.primary, .primary{color:#FFF6F1!important}
.gr-button-primary, button[variant="primary"]{color:#FFF6F1!important}
/* ── header with floating 3D boxes ── */
#tib-header{position:relative;text-align:center;margin:2px 0 14px;padding:22px 0 18px;overflow:hidden}
#tib-header .v{position:relative;z-index:2;display:inline-block;font-family:'Outfit',system-ui,sans-serif;font-weight:600;font-size:.78rem;letter-spacing:.1em;text-transform:uppercase;color:#C0604A;background:#F1EBE3;border:1px solid #e6c3ba;border-radius:999px;padding:4px 15px;margin-bottom:12px}
#tib-header .t{position:relative;z-index:2;font-family:'Cormorant Garamond',Georgia,serif;font-weight:600;font-size:3.1rem;line-height:1;letter-spacing:-.01em;color:#1A1714}
#tib-header .t em{font-style:italic;color:#C0604A}
#tib-header .s{position:relative;z-index:2;font-family:'Outfit',sans-serif;font-weight:400;color:#6B6560;font-size:1.06rem;margin-top:6px;letter-spacing:.01em}
#tib-header .s b{color:#C0604A;font-weight:600}
.tib-cube{position:absolute;z-index:1;opacity:.5;animation:tibfloat 9s ease-in-out infinite}
.tib-cube.c1{left:6%;top:24%;width:54px;animation-delay:0s}
.tib-cube.c2{left:16%;top:58%;width:38px;opacity:.35;animation-delay:1.4s}
.tib-cube.c3{right:8%;top:20%;width:60px;animation-delay:.6s}
.tib-cube.c4{right:19%;top:60%;width:40px;opacity:.35;animation-delay:2.1s}
.tib-cube.c5{left:38%;top:8%;width:30px;opacity:.28;animation-delay:1.1s}
.tib-cube.c6{right:36%;top:70%;width:32px;opacity:.3;animation-delay:.3s}
@keyframes tibfloat{0%,100%{transform:translateY(0) rotate(-6deg)}50%{transform:translateY(-12px) rotate(6deg)}}
h2,h3{font-family:'Cormorant Garamond',Georgia,serif!important;color:#1A1714}
/* editor fills the row height so it matches the (usually taller) controls column.
   The row stretches both columns to equal height; make the editor column's children
   flex-fill that stretched height (height:100% would resolve against an indefinite
   row height and collapse to content). */
.tib-edcol{display:flex!important;flex-direction:column;min-height:0}
.tib-edcol > *, .tib-edcol .block, .tib-edcol .html-container, .tib-edcol .prose{
  display:flex!important;flex-direction:column;flex:1 1 auto!important;min-height:0;height:auto!important}
#tib-editor{flex:1 1 auto;height:100%!important;min-height:620px;display:block}
.tib-ctrl{align-self:stretch;flex-wrap:nowrap!important;overflow-y:auto;padding-right:8px;gap:12px}
/* collapse empty status/markdown blocks so they don't add dead space */
.tib-ctrl .prose:empty, .tib-ctrl .md:empty, .tib-ctrl .block.padded:empty{display:none!important;min-height:0!important;padding:0!important;margin:0!important}
.tib-ctrl::-webkit-scrollbar{width:8px}
.tib-ctrl::-webkit-scrollbar-thumb{background:#D8D4CB;border-radius:99px}
/* square source / target preview panels */
.tib-sq .image-container{aspect-ratio:1/1;min-height:190px}
.tib-sq .image-container img{object-fit:contain}
/* generated-edit gallery: square panel, image fits tightly (contain, no crop) */
.tib-out .grid-wrap, .tib-out .fixed-height{min-height:0!important;height:auto!important}
.tib-out .grid-container{aspect-ratio:1/1;height:auto!important;grid-auto-rows:1fr}
.tib-out .thumbnail-item{aspect-ratio:1/1;height:auto!important}
.tib-out .preview .image-container{aspect-ratio:1/1;min-height:0!important}
.tib-out .preview{height:auto!important}
.tib-out img{width:100%!important;height:100%!important;object-fit:contain!important}
.tib-out .empty, .tib-out .wrap{aspect-ratio:1/1;min-height:0!important}
/* small orange numbered step circles on the Grab / Generate buttons (demo-video style) */
.tib-step button, button.tib-step{position:relative;padding-left:46px!important;font-weight:600!important}
.tib-step button::before, button.tib-step::before{
  position:absolute;left:12px;top:50%;transform:translateY(-50%);
  width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-family:'Outfit',system-ui,sans-serif;font-weight:700;font-size:.88rem;line-height:1;border:1.5px solid #C0604A}
.tib-step1 button::before, button.tib-step1::before{content:"1";background:#C0604A;color:#FFF6F1}
.tib-step2 button::before, button.tib-step2::before{content:"2";background:#FFF6F1;color:#C0604A;border-color:#FFF6F1}
/* checkboxes: visible terracotta fill + white check when ticked */
.gradio-container input[type=checkbox]{accent-color:#C0604A;width:18px;height:18px;border:1.5px solid #C0604A!important;background:#FFFFFF!important;appearance:auto!important;-webkit-appearance:auto!important;opacity:1!important}
.gradio-container input[type=checkbox]:checked{background:#C0604A!important;border-color:#C0604A!important}
.tib-hidden{display:none!important}
footer{display:none!important}
""".replace("__VARS__", _VARS)


# Gradio 6 moved theme/css/js off Blocks and onto mount_gradio_app()/launch();
# older Gradio takes them on Blocks. Detect which and route accordingly.
UI_KWARGS = {"theme": THEME, "css": CSS}
if FORCE_LIGHT is not None:
    UI_KWARGS["js"] = FORCE_LIGHT
_MOUNT_HAS_THEME = "theme" in inspect.signature(gr.mount_gradio_app).parameters


def build_demo():
    iframe = ('<iframe id="tib-editor" src="box-editor.html" '
              'style="width:100%;height:100%;min-height:620px;border:1px solid #D8D4CB;'
              'border-radius:12px;background:#F6F3EE"></iframe>')
    blocks_kwargs = {} if _MOUNT_HAS_THEME else dict(UI_KWARGS)
    with gr.Blocks(title="Thinking in Boxes", **blocks_kwargs) as demo:
        cube = ('<svg class="tib-cube {c}" viewBox="0 0 100 100" fill="none" stroke="#C0604A" '
                'stroke-width="4" stroke-linejoin="round">'
                '<path d="M20 35 L50 20 L80 35 L50 50 Z"/>'
                '<path d="M20 35 L20 70 L50 85 L50 50 Z"/>'
                '<path d="M80 35 L80 70 L50 85 L50 50 Z"/></svg>')
        cubes = "".join(cube.format(c=f"c{i}") for i in range(1, 7))
        gr.HTML(f'<div id="tib-header">{cubes}'
                '<div class="v">NeurIPS 2026</div>'
                '<div class="t">Thinking in <em>Boxes</em></div>'
                '<div class="s">3D Editing in Real Images Made Easy</div></div>')

        with gr.Row(equal_height=True):
            with gr.Column(scale=3, elem_classes="tib-edcol"):
                gr.HTML(iframe)
            with gr.Column(scale=2, elem_classes="tib-ctrl"):
                grab = gr.Button("Grab image + boxes from editor", elem_classes="tib-step tib-step1")
                with gr.Row(equal_height=True):
                    src_prev = gr.Image(label="Source boxes", elem_classes="tib-sq", show_label=True)
                    tgt_prev = gr.Image(label="Target boxes", elem_classes="tib-sq", show_label=True)
                image_u = gr.Textbox(visible=False); src_u = gr.Textbox(visible=False)
                tgt_u = gr.Textbox(visible=False); status_u = gr.Textbox(visible=False)
                randomize = gr.Checkbox(value=True, label="🎲 Randomize seed", elem_classes="tib-rand")
                gen = gr.Button("Generate", variant="primary", elem_classes="tib-step tib-step2 tib-gen")
                # generated edit sits directly under the Generate button (right column, beside the editor)
                out = gr.Gallery(label="Generated edit", columns=1, object_fit="contain",
                                 show_label=True, elem_classes="tib-out")
                seed_used = gr.Markdown()

        # WildDet3D bridge: editor postMessage → trigger → detect_objects → editor.
        # Rendered (not visible=False, which drops them from the DOM) but hidden via CSS.
        detect_payload = gr.Textbox(elem_id="tib-detect-payload", elem_classes="tib-hidden")
        detect_result  = gr.Textbox(elem_id="tib-detect-result", elem_classes="tib-hidden")
        detect_trigger = gr.Button("detect", elem_id="tib-detect-trigger", elem_classes="tib-hidden")

        # ── Model / pipeline settings (config; collapsed at the bottom) ──
        with gr.Accordion("Model Settings", open=False) as model_acc:
            with gr.Row():
                model_path = gr.Textbox(value=DEFAULT_MODEL, label="pretrained_model_name_or_path", scale=2,
                                        placeholder="black-forest-labs/FLUX.1-dev")
                pipeline_path = gr.Textbox(value=DEFAULT_PIPELINE, label="pipeline_path (custom pipeline)", scale=2,
                                           placeholder="path/to/pipeline.py or module")
            with gr.Row():
                dtype = gr.Dropdown(list(DTYPE_MAP), value=DEFAULT_DTYPE, label="dtype")
                device = gr.Textbox(value=DEFAULT_DEVICE, label="device")
                device_map = gr.Textbox(value=DEFAULT_DEVICE_MAP, label="device_map (optional)", placeholder="balanced")
                max_seq = gr.Number(value=DEFAULT_MAX_SEQ, label="max_sequence_length", precision=0)
            with gr.Row():
                offload = gr.Checkbox(value=False, label="offload")
                local = gr.Checkbox(value=DEFAULT_LOCAL, label="local files only")
                compile_ = gr.Checkbox(value=DEFAULT_COMPILE, label="compile")
                vae_tiling = gr.Checkbox(value=False, label="vae_tiling")
                vae_slicing = gr.Checkbox(value=False, label="vae_slicing")
            with gr.Row():
                lora_path = gr.Textbox(value=DEFAULT_LORA, label="lora_adapter_path", scale=2,
                                       placeholder="path/to/lora (optional)")
                adapter_name = gr.Textbox(value=DEFAULT_ADAPTER, label="adapter_name")
                weight_name = gr.Textbox(value=DEFAULT_WEIGHT, label="weight_name")
            token = gr.Textbox(label="HF token (optional)", type="password")
            load_btn = gr.Button("Load Model & Pipeline", variant="primary")
            load_status = gr.Markdown()

            gr.Markdown("#### Generation settings")
            with gr.Row():
                steps = gr.Slider(1, 100, value=DEFAULT_STEPS, step=1, label="Inference steps")
                num_images = gr.Slider(1, 4, value=1, step=1, label="Images per run")
            with gr.Row():
                height = gr.Slider(256, 1536, value=DEFAULT_HEIGHT, step=16, label="Height")
                width = gr.Slider(256, 1536, value=DEFAULT_WIDTH, step=16, label="Width")
            # fixed defaults (no UI): guidance / true-CFG / LoRA scale — mirror inference.sh
            guidance = gr.State(DEFAULT_GUIDANCE); true_cfg = gr.State(DEFAULT_TRUE_CFG); lora_scale = gr.State(DEFAULT_SCALE)

        load_btn.click(load_model,
                       [model_path, pipeline_path, dtype, device, device_map, offload, local, compile_,
                        vae_tiling, vae_slicing, token, lora_path, adapter_name, weight_name, max_seq],
                       [load_status, model_acc])
        grab.click(None, None, [image_u, src_u, tgt_u, status_u], js=GRAB_JS).then(
            grab_show, [status_u, src_u, tgt_u], [src_prev, tgt_prev])
        # WildDet3D: editor → hidden trigger → in-process model → post detections back
        detect_trigger.click(detect_objects, [detect_payload], [detect_result]).then(
            None, [detect_result], None, js=DETECT_RESULT_JS)
        demo.load(None, None, None, js=DETECT_BRIDGE_JS)   # install the postMessage listener
        gen.click(generate,
                  [image_u, src_u, tgt_u, steps, guidance, true_cfg, lora_scale,
                   randomize, height, width, num_images],
                  [out, seed_used])
    return demo


def autoload_default_model():
    """Load + cache the default model/pipeline once at startup (mirrors inference.sh)."""
    if not DEFAULT_MODEL:
        return
    a = SimpleNamespace(
        pretrained_model_name_or_path=DEFAULT_MODEL, pipeline_path=DEFAULT_PIPELINE,
        dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE, device_map=DEFAULT_DEVICE_MAP,
        offload=False, local=DEFAULT_LOCAL, compile=DEFAULT_COMPILE,
        vae_tiling=False, vae_slicing=False, token="",
        lora_adapter_path=DEFAULT_LORA, adapter_name=DEFAULT_ADAPTER, weight_name=DEFAULT_WEIGHT)
    print(f"[tib] loading default pipeline: {DEFAULT_MODEL}  pipeline={DEFAULT_PIPELINE}  "
          f"LoRA={DEFAULT_LORA} ({DEFAULT_ADAPTER})  device_map={DEFAULT_DEVICE_MAP}  dtype={DEFAULT_DTYPE} ...", flush=True)
    try:
        STATE["pipe"] = load_pipe(a)
        STATE["cfg_sig"] = pipe_signature(a)
        STATE["device"] = DEFAULT_DEVICE if not DEFAULT_DEVICE_MAP else "cuda"
        STATE["max_sequence_length"] = DEFAULT_MAX_SEQ
        print(f"[tib] ✓ pipeline loaded + cached on {STATE['device']}  (LoRA {DEFAULT_ADAPTER})", flush=True)
    except Exception:
        import traceback
        print("[tib] default pipeline autoload FAILED — full traceback below; "
              "fix the paths/deps or load from the UI:", flush=True)
        traceback.print_exc()


DETECT_STATE = {"model": None, "mock": None, "tried": False, "wd3d": None}

def _np_any(x):
    """Any torch tensor / list / ndarray -> numpy (CPU or CUDA)."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _import_wilddet3d():
    """Make `import wilddet3d` resolve from the code checkout, then import it (cached).
    `wilddet3d/__init__.py` self-registers its sam3/lingbot_depth submodule paths, so
    only the repo root needs to be on sys.path. Returns the module."""
    if DETECT_STATE["wd3d"] is not None:
        return DETECT_STATE["wd3d"]
    if WILDDET3D_REPO and WILDDET3D_REPO not in sys.path:
        sys.path.insert(0, WILDDET3D_REPO)
        # belt-and-braces: also add the vendored dirs in case __init__ expects them on-path
        for sub in ("third_party/sam3", "third_party"):
            p = os.path.join(WILDDET3D_REPO, *sub.split("/"))
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
    if WILDDET3D_SAM3 and WILDDET3D_SAM3 not in sys.path:
        sys.path.insert(0, WILDDET3D_SAM3)
    import wilddet3d  # noqa: resolved via WILDDET3D_REPO (repo has no setup.py)
    DETECT_STATE["wd3d"] = wilddet3d
    return wilddet3d


def _wd3d_checkpoint():
    """Local ckpt override if set + present, else auto-download from the HF Hub."""
    if WILDDET3D_CKPT and os.path.isfile(WILDDET3D_CKPT):
        return WILDDET3D_CKPT
    from huggingface_hub import hf_hub_download
    print(f"[tib] fetching WildDet3D weights {WILDDET3D_HF_REPO}/{WILDDET3D_WEIGHT} "
          f"from the HF Hub (cached after first run) ...", flush=True)
    return hf_hub_download(repo_id=WILDDET3D_HF_REPO, filename=WILDDET3D_WEIGHT,
                           token=None)


def _detector():
    """Load + cache the real WildDet3D model on the device once. Sets DETECT_STATE['mock']
    True (and returns None) when the package/weights can't be loaded so Detect still works."""
    if DETECT_STATE["tried"]:
        return DETECT_STATE["model"]
    DETECT_STATE["tried"] = True
    if FORCE_DETECT_MOCK:
        DETECT_STATE["mock"] = True
        print("[tib] WildDet3D: TIB_DETECT_MOCK set — using mock boxes.")
        return None
    try:
        wd3d = _import_wilddet3d()
        ckpt = _wd3d_checkpoint()
        bm_kwargs = dict(checkpoint=ckpt, score_threshold=WILDDET3D_SCORE, skip_pretrained=True)
        # If this build_model exposes an hf_endpoint / endpoint / token arg, pass it through
        # (some checkouts require it to fetch the LingBot-Depth / SAM3 sub-weights).
        bm_params = inspect.signature(wd3d.build_model).parameters
        if "hf_endpoint" in bm_params:
            bm_kwargs["hf_endpoint"] = HF_ENDPOINT
        elif "endpoint" in bm_params:
            bm_kwargs["endpoint"] = HF_ENDPOINT
        model = wd3d.build_model(**bm_kwargs)
        dev = STATE.get("device", "cuda")
        if hasattr(model, "to"):
            try:
                model = model.to(dev)
            except Exception as e:
                print(f"[tib] WildDet3D .to({dev}) failed ({e}); using as-loaded.")
        DETECT_STATE["model"] = model
        DETECT_STATE["mock"] = False
        print(f"[tib] WildDet3D loaded on {dev}: {ckpt}", flush=True)
    except Exception:
        import traceback
        DETECT_STATE["mock"] = True
        print("[tib] WildDet3D unavailable — falling back to mock boxes. "
              "For the real detector, set WILDDET3D_REPO to a "
              "`git clone --recurse-submodules https://github.com/allenai/WildDet3D` checkout "
              "(the weights auto-download from HF). Traceback:", flush=True)
        traceback.print_exc()
    return DETECT_STATE["model"]


def run_wilddet3d(image_np, prompts, points=None, boxes=None, mode="geometric"):
    """Real model. image_np: HxWx3 float32. points/boxes in ORIGINAL-image pixels:
      points : [[x,y,label], ...] label 1=pos 0=neg -> one box each ("geometric")
      boxes  : [[x1,y1,x2,y2], ...] "geometric"=one-to-one, "visual"=find similar
      else   : text prompts (one-to-many)."""
    wd3d = _import_wilddet3d()
    model = DETECT_STATE["model"]
    if model is None:
        raise RuntimeError("WildDet3D model is not loaded")
    data = wd3d.preprocess(image_np, None)     # intrinsics=None -> model predicts them
    dev = STATE.get("device", "cuda")
    kw = {}
    if points:
        kw["input_points"] = [[(float(p[0]), float(p[1]), int(p[2])) for p in points]]
        kw["prompt_text"] = mode
    elif boxes:
        kw["input_boxes"] = [[float(v) for v in b] for b in boxes]
        kw["prompt_text"] = mode
    else:
        kw["input_texts"] = prompts
    out = model(
        images=data["images"].to(dev),
        intrinsics=data["intrinsics"].to(dev)[None],
        input_hw=[data["input_hw"]],
        original_hw=[data["original_hw"]],
        padding=[data["padding"]],
        **kw,
    )
    # out = (boxes, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps), per-image lists
    b2d, b3d, sc, cid = _np_any(out[0][0]), _np_any(out[1][0]), _np_any(out[2][0]), _np_any(out[5][0])
    K = _np_any(data["intrinsics"])
    if K.ndim == 3:
        K = K[0]
    H, W = image_np.shape[:2]
    labels = list(prompts) if prompts else []
    dets = []
    for i in range(b3d.shape[0]):
        ci = int(cid[i]) if i < len(cid) else i
        dets.append({
            "label": labels[ci] if 0 <= ci < len(labels) else f"object {i+1}",
            "class_id": ci,
            "score": float(sc[i]) if i < len(sc) else 1.0,
            "box2d": [float(v) for v in b2d[i]],
            "center": [float(v) for v in b3d[i, 0:3]],
            "dims":   [float(v) for v in b3d[i, 3:6]],
            "quat":   [float(v) for v in b3d[i, 6:10]],
        })
    return {"image_hw": [H, W], "intrinsics": K.tolist(), "detections": dets}


def run_mock(image_np, prompts, points=None, boxes=None):
    """Plausible metric detections without a model, so Detect works with zero setup."""
    H, W = image_np.shape[:2]
    fx = 1.2 * W                                   # ~55° hfov, principal point centred
    K = [[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]]
    prompt_xy = []
    if points:
        prompt_xy = [(p[0], p[1]) for p in points if int(p[2]) == 1]
    elif boxes:
        prompt_xy = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    if prompt_xy:
        dets = []
        for i, (u, v) in enumerate(prompt_xy):
            Z = 6.0
            X = (u - W / 2) * Z / fx
            Y = (v - H / 2) * Z / fx
            dets.append({"label": prompts[i] if i < len(prompts) else f"object {i+1}",
                         "class_id": i, "score": 0.88,
                         "box2d": [u - 40, v - 40, u + 40, v + 40],
                         "center": [X, Y, Z], "dims": [0.8, 1.0, 0.8], "quat": [1, 0, 0, 0]})
        return {"image_hw": [H, W], "intrinsics": K, "detections": dets}
    catalog = {                                    # metric priors, camera frame (y-down)
        "car":     {"center": [0.4, 0.9, 8.0],  "dims": [1.8, 1.5, 4.3], "quat": [0.966, 0, 0.259, 0]},
        "person":  {"center": [-2.2, 0.5, 6.0], "dims": [0.6, 1.7, 0.6], "quat": [1, 0, 0, 0]},
        "bicycle": {"center": [2.4, 0.8, 5.0],  "dims": [0.6, 1.1, 1.7], "quat": [0.985, 0, 0.174, 0]},
        "chair":   {"center": [0.0, 0.6, 3.2],  "dims": [0.6, 0.9, 0.6], "quat": [1, 0, 0, 0]},
        "cup":     {"center": [0.1, 0.4, 1.4],  "dims": [0.09, 0.11, 0.09], "quat": [1, 0, 0, 0]},
    }
    dets = []
    for i, p in enumerate(prompts):
        pr = catalog.get(p.strip().lower(), {"center": [0, 0.6, 4 + i], "dims": [0.8, 0.8, 0.8], "quat": [1, 0, 0, 0]})
        dets.append({"label": p.strip(), "class_id": i, "score": 0.9 - 0.05 * i,
                     "box2d": [W * 0.3, H * 0.4, W * 0.7, H * 0.8], **pr})
    return {"image_hw": [H, W], "intrinsics": K, "detections": dets}


def detect_objects(payload_json):
    """Run WildDet3D in-process on the editor's image + point/text prompts.
    Returns a JSON string: the detections dict, or {"error": ...}."""
    try:
        req = json.loads(payload_json) if payload_json else {}
    except Exception:
        return json.dumps({"error": "bad request payload"})
    img = dataurl_to_pil(req.get("image"))
    if img is None:
        return json.dumps({"error": "no image — load one in the editor first"})
    arr = np.array(img).astype(np.float32)
    prompts = [p.strip() for p in (req.get("prompts") or "").split(",") if p.strip()]
    points = req.get("points") or None
    boxes = req.get("boxes") or None
    mode = req.get("mode", "geometric")
    try:
        _detector()
        if DETECT_STATE["mock"]:
            return json.dumps(run_mock(arr, prompts, points, boxes))
        return json.dumps(run_wilddet3d(arr, prompts, points, boxes, mode))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"error": str(e)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server_name", type=str, default="0.0.0.0")
    p.add_argument("--server_port", type=int, default=7880)
    p.add_argument("--share", action="store_true")
    cfg = p.parse_args()

    autoload_default_model()          # load + cache the default model/pipeline if configured
    demo = build_demo()
    app = FastAPI()

    @app.get("/box-editor.html")
    def _editor():
        return FileResponse(str(EDITOR_HTML))

    mount_kwargs = dict(UI_KWARGS) if _MOUNT_HAS_THEME else {}
    _fav = Path(GRADIO_TEMP_DIR) / "favicon.svg"          # ← add
    _fav.write_text(FAVICON_SVG, encoding="utf-8")        # ← add
    mount_kwargs["favicon_path"] = str(_fav)              # ← add
    app = gr.mount_gradio_app(app, demo, path="/", **mount_kwargs)
    import uvicorn
    uvicorn.run(app, host=cfg.server_name, port=cfg.server_port)


if __name__ == "__main__":
    main()
