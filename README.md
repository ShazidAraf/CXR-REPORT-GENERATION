# Chest X-ray Report Generation

## Model weights

The model weights (~16 GB) are **not** included in this repository (they exceed
GitHub's file size limit). Download them here and place them under `src/CKPT/`:

**Model weights (Google Drive):** https://drive.google.com/file/d/1S5Cdc14Uldnc4rjWNbf9f1AO39n0nDtv/view?usp=sharing

This archive contains **all** required weights — the base model plus the SFT and
GRPO LoRA adapters. No other downloads are needed (no HuggingFace fetch at
runtime); the module is fully self-contained once these are in place.

For Docker build/run instructions, see [`DOCKER.md`](DOCKER.md).

## Contents

```
BISQUE/
├── README.md             # this file
├── config.json           # paths + generation params
├── inference.py          # entry point (inference + visualization)
├── environment.yml       # full conda env spec
├── requirements.txt      # pip fallback
├── CKPT/                 # LoRA adapters (SFT + GRPO)
│   ├── SFT/                 ← merged first
│   └── GRPO/                ← merged on top of SFT
└── samples/
    ├── *.png             # 10 chest X-ray inputs
    ├── ground_truth.json # reference reports per case
    ├── inference_results.json   # written by inference.py
    └── viz/              # written by inference.py — side-by-side PNGs
```

## Step-by-step setup (for someone who received `BISQUE.zip`)

### 1. Unzip

```bash
unzip BISQUE.zip
cd BISQUE
```

### 2. Create the conda environment

You need [Anaconda or Miniconda](https://docs.conda.io/en/latest/miniconda.html)
installed. Then:

```bash
conda env create -f environment.yml
conda activate cxr_report
```

If `conda env create` is too slow or hits package conflicts on a
different OS, use a fresh env plus pip:

```bash
conda create -n cxr_report python=3.10
conda activate cxr_report
pip install -r requirements.txt
```

### 3. Install LLaVA (third-party dependency)

The inference script imports from `llava.model.builder`,
`llava.constants`, `llava.conversation`, and `llava.mm_utils`. These
come from the LLaVA repository (the LLaVA-Med fork is compatible).
Install it once into the env:

```bash
git clone https://github.com/microsoft/LLaVA-Med.git
cd LLaVA-Med
pip install -e .
cd ..
```

(or use the upstream `git clone https://github.com/haotian-liu/LLaVA.git`
if you already have it set up.)

### 4. Verify GPU availability

You need an NVIDIA GPU with ≥ 16 GB VRAM (the base model is 7B params
in fp16, plus the merged adapters):

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### 5. Check the checkpoint paths

Make sure `config.json` points to the adapter directories that are
actually shipped in `CKPT/`:

```bash
ls CKPT/SFT/         # should show adapter_config.json + adapter_model.bin
ls CKPT/GRPO/        # same shape
```

If the names don't match, edit `config.json`:

```json
{
    "sft_adapter":  "CKPT/SFT",
    "lora_adapter": "CKPT/GRPO"
}
```

### 6. Run inference

```bash
python inference.py
```

What happens:

1. Downloads `microsoft/llava-med-v1.5-mistral-7b` from Hugging Face
   (cached to `~/.cache/huggingface/hub/` — first run only, ~14 GB).
2. Loads the base model → merges the SFT LoRA → merges the GRPO LoRA.
3. Generates a report for each PNG in `samples/`.
4. Writes `samples/inference_results.json` after every image (safe to
   resume; incremental save).
5. Renders one side-by-side PNG per case into `samples/viz/`.

### 7. Inspect the outputs

```bash
ls samples/viz/                          # one PNG per case
head -30 samples/inference_results.json
```

Each viz PNG has:
- **Left**: the input chest X-ray
- **Right**: the reference report (blue header, percentage = recall)
  and the generated report (red header, percentage = precision); every
  word colored **green** if it matched the other side, **red** otherwise.

## Customizing the run

Edit `config.json`:

- `prompt`         — the instruction sent with each image
- `temperature`    — 0.0 = greedy, higher = more diverse
- `top_p`          — nucleus sampling threshold
- `max_new_tokens` — generation length cap
- `do_sample`      — set to `false` for deterministic greedy output
- `seed`           — randomness control (only used when `do_sample=true`)

Override the config path at runtime:

```bash
python inference.py --config other_config.json
```

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'llava'` | Install LLaVA (step 3) |
| `load_pretrained_model() got an unexpected keyword argument 'torch_dtype'` | You're on an older LLaVA fork. This script already handles that signature. |
| `CUDA out of memory` | Use a GPU with more VRAM; or reduce `max_new_tokens` |
| `adapter_config.json not found` | The `CKPT/` paths in `config.json` don't match what's in the zip — fix step 5 |
