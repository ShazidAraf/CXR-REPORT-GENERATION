# Running the CXR Report module with Docker

This module generates a radiology report from a chest X-ray using
**LLaVA-Med v1.5 (Mistral-7B)** with merged **SFT + GRPO LoRA** adapters.

There are two Docker images:

| File | Image | Purpose |
|------|-------|---------|
| `Dockerfile` | CUDA / GPU | **Real inference.** Loads the 7B model and generates actual reports. Linux + NVIDIA GPU. |
| `Dockerfile.cpu` | CPU / arm64 | **Integration stub.** No model, returns a canned report. For testing the BisQue round-trip on a GPU-less host (e.g. a Mac). |

---

## 0. Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- For **real inference**: a Linux x86_64 host with an NVIDIA GPU and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (`--gpus all` must work). The CUDA wheels in `requirements.txt` do **not**
  install on macOS / Apple Silicon — use `Dockerfile.cpu` there instead.

## 1. Get the model weights (required for real inference)

The weights (~16 GB) are **not** in this repository (they exceed GitHub's file
size limit). Download them here:

**Model weights (Google Drive):** https://drive.google.com/file/d/1S5Cdc14Uldnc4rjWNbf9f1AO39n0nDtv/view?usp=sharing

This archive contains **all** required weights — the base model plus the SFT and
GRPO LoRA adapters. No other downloads are needed. After downloading, unpack them
under `src/CKPT/` so the layout matches `src/config.json`:

```
src/CKPT/
├── BASE/llava-med-v1.5-mistral-7b/     # base model
├── SFT/best/                           # SFT LoRA adapter
└── GRPO/checkpoint-best/               # GRPO LoRA adapter
```

The stub image (`Dockerfile.cpu`) does **not** need the weights.

---

## 2. Real inference (GPU)

### Build

```bash
docker build -t cxr_report:v1.0.0 .
```

> For BisQue deployment, tag with the registry prefix BisQue prepends when it
> pulls, and keep it in sync with `runtime-module.cfg`:
> `docker build -t biodev.ece.ucsb.edu:5000/cxr_report:v1.0.0 .`

### Run a standalone smoke test (no BisQue server)

```bash
docker run --rm --gpus all cxr_report:v1.0.0 \
    python -m src.BQ_run_module samples/CXR1701_IM-0462.png
```

Expected: the model loads from `src/CKPT/...`, then prints the generated report
and `[saved to <stem>_report.txt]`.

---

## 3. Integration stub (CPU / Mac)

Use this to verify the module runs without a GPU. It sets `BISQUE_STUB=1` and
returns a canned report.

### Build

```bash
docker build -f Dockerfile.cpu -t cxr_report:v1.0.0-stub .
```

### Run

```bash
docker run --rm cxr_report:v1.0.0-stub \
    python -c "import bqapi.comm; print('bqapi import OK')"
```

`bqapi import OK` confirms the image and BisQue client are healthy.

---

## 4. Inputs and outputs

- **Input:** one chest X-ray image (`CXR Image`). Sample images are in `samples/`.
- **Output:** a text report `<image_stem>_report.txt` (`Generated Report`),
  written to the output folder and, under BisQue, uploaded as a file resource.

Generation settings (in `src/config.json`): fp16, `max_new_tokens=1024`,
`temperature=0.2`, `top_p=0.9`.

---

## 5. Running inside BisQue

To register and run this as a BisQue module on a local BisQue dev server
(Module Manager, engine service, staging logs), see **`LOCAL_BISQUE_TESTING.md`**.

Key points:
- `runtime-module.cfg` sets `docker.image`; the built image tag must match it.
- BisQue launches the module container and calls `python PythonScriptWrapper.py`,
  which fetches the input, calls `src/BQ_run_module.run_module(...)`, and uploads
  the resulting report.
- For the module container to reach the BisQue server, run the dev container with
  host networking (`--net=host --ipc=host`; on Docker Desktop enable host
  networking in Settings).
```
