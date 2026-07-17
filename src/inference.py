"""
inference.py
============
Generate radiology reports from chest X-rays using the LLaVA-Med base model
with merged SFT + GRPO LoRA adapters stored under src/CKPT/.

Two ways to use this file:

1. Programmatic (used by BQ_run_module.py for one image at a time under the
   BisQue PythonScriptWrapper):
       from src.inference import load_pipeline, generate_report
       pipe = load_pipeline(cfg)
       text = generate_report(pipe, image_path)

2. Standalone CLI — batch over the PNGs in ../samples and dump JSON:
       python -m src.inference                     # uses src/config.json
       python -m src.inference --config other.json

All checkpoint paths in config.json are resolved relative to this src/
directory, so the module is self-contained (no HuggingFace download needed
once CKPT/BASE holds the base model).
"""

import argparse
import glob
import json
import os
import random
import sys

import torch
from PIL import Image
from tqdm import tqdm


HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve(p: str) -> str:
    """Resolve a config path: absolute stays absolute; relative is anchored to src/."""
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.join(HERE, p)


def _resolve_model(p: str) -> str:
    """Resolve a base_model spec. A relative path that exists under src/ is
    anchored there (self-contained local weights); otherwise it is returned
    unchanged so it can still be used as a HuggingFace hub id."""
    if not p:
        return p
    if os.path.isabs(p):
        return p
    local = os.path.join(HERE, p)
    return local if os.path.exists(local) else p


# =============================================================================
# Model loading + per-image generation (reusable from BQ_run_module)
# =============================================================================

def load_model(base_model: str, sft_adapter: str, lora_adapter: str, device: str):
    """Load base -> merge SFT LoRA -> merge GRPO LoRA. Mirrors the parent
    project's model.py loading logic so behavior matches the original eval.
    """
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from peft import PeftModel

    base_model = _resolve_model(base_model)
    model_name = get_model_name_from_path(base_model)
    print(f"[load] base = {base_model}  (name={model_name})")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=base_model,
        model_base=None,
        model_name=model_name,
        device_map="auto",
        device=device,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _apply(model, adapter_path: str, tag: str):
        adapter_path = _resolve(adapter_path)
        non_lora = os.path.join(adapter_path, "non_lora_trainables.bin")
        if os.path.exists(non_lora):
            print(f"[load] {tag} non-LoRA trainables: {non_lora}")
            nl = torch.load(non_lora, map_location="cpu")
            nl = {(k[11:] if k.startswith("base_model.") else k): v for k, v in nl.items()}
            if any(k.startswith("model.model.") for k in nl):
                nl = {(k[6:] if k.startswith("model.") else k): v for k, v in nl.items()}
            model.load_state_dict(nl, strict=False)
        print(f"[load] {tag} LoRA weights: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"[load] {tag} merge_and_unload")
        return model.merge_and_unload()

    if sft_adapter:
        model = _apply(model, sft_adapter, tag="SFT")
    if lora_adapter:
        model = _apply(model, lora_adapter, tag="GRPO")

    model.eval()
    return model, tokenizer, image_processor


def build_input(prompt_text: str, image: Image.Image, model, tokenizer,
                image_processor, device: str):
    from llava.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
                                 DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    image_tensor = image_processor(images=image, return_tensors="pt")["pixel_values"][0]
    image_tensor = image_tensor.unsqueeze(0).to(device, dtype=torch.float16)

    if getattr(model.config, "mm_use_im_start_end", False):
        inp_text = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN + "\n" + prompt_text)
    else:
        inp_text = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

    conv = conv_templates["mistral_instruct"].copy()
    conv.append_message(conv.roles[0], inp_text)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    return input_ids, image_tensor


def load_pipeline(cfg: dict, device: str = None):
    """Load model + tokenizer + image processor once. Returns a dict that
    generate_report() consumes. Relative paths in cfg are resolved against src/.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(cfg.get("seed", 42))
    torch.manual_seed(cfg.get("seed", 42))

    model, tokenizer, image_processor = load_model(
        base_model=cfg["base_model"],
        sft_adapter=cfg.get("sft_adapter", ""),
        lora_adapter=cfg.get("lora_adapter", ""),
        device=device,
    )
    return {
        "model": model,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "device": device,
        "prompt": cfg["prompt"],
        "max_new_tokens": int(cfg.get("max_new_tokens", 1024)),
        "temperature":    float(cfg.get("temperature", 0.2)),
        "top_p":          float(cfg.get("top_p", 0.9)),
        "do_sample":      bool(cfg.get("do_sample", True)),
    }


def generate_report(pipeline: dict, image_path: str) -> str:
    """Run a single CXR image through the loaded pipeline and return the
    generated report text.
    """
    image = Image.open(image_path).convert("RGB")
    input_ids, image_tensor = build_input(
        pipeline["prompt"], image,
        pipeline["model"], pipeline["tokenizer"],
        pipeline["image_processor"], pipeline["device"])

    with torch.inference_mode():
        out = pipeline["model"].generate(
            input_ids, images=image_tensor,
            attention_mask=torch.ones_like(input_ids),
            pad_token_id=pipeline["tokenizer"].pad_token_id,
            max_new_tokens=pipeline["max_new_tokens"],
            do_sample=pipeline["do_sample"],
            temperature=pipeline["temperature"],
            top_p=pipeline["top_p"],
            num_beams=1,
            use_cache=True,
        )
    return pipeline["tokenizer"].decode(out[0], skip_special_tokens=True).strip()


# =============================================================================
# Standalone CLI — batch over ../samples and dump JSON.
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(HERE, "config.json"))
    args = parser.parse_args()

    cfg_path = os.path.abspath(args.config)
    print(f"[main] config = {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)

    samples_dir = _resolve(cfg.get("samples_dir", "../samples"))
    output_path = _resolve(cfg.get("output_path", "../samples/inference_results.json"))

    image_paths = sorted(glob.glob(os.path.join(samples_dir, "*.png")))
    if not image_paths:
        print(f"[main] No .png files found in {samples_dir}")
        sys.exit(1)
    print(f"[main] Found {len(image_paths)} images in {samples_dir}")

    pipeline = load_pipeline(cfg)

    results = {}
    for img_path in tqdm(image_paths, desc="Inference"):
        cid = os.path.splitext(os.path.basename(img_path))[0]
        try:
            text = generate_report(pipeline, img_path)
        except Exception as e:
            print(f"[skip] failed on {img_path}: {e}")
            continue

        tqdm.write(f"\n=== {cid} ===\n{text}\n")
        results[cid] = {
            "generated": text,
            "image_path": os.path.relpath(img_path, HERE),
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n[main] Wrote {len(results)} inference results to {output_path}")


if __name__ == "__main__":
    main()
