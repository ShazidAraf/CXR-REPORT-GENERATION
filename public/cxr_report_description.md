# CXR Report Generator

**Author:** Md Islam · **Group:** Medical Imaging · **Input:** one chest X-ray image · **Output:** a text radiology report (`.txt`)

## Overview

CXR Report Generator produces a draft radiology report from a single chest X-ray. Upload a CXR image, run the module, and it returns a written report describing both abnormal and normal findings across the visible anatomical structures, in the familiar **FINDINGS / IMPRESSION** style.

The module is powered by **LLaVA-Med v1.5 (Mistral-7B)**, a medical vision-language model, specialized for chest-X-ray reporting through two stages of fine-tuning: **supervised fine-tuning (SFT)** followed by **reinforcement learning with GRPO**, each delivered as a LoRA adapter merged into the base model.

## How it works

The chest X-ray is encoded by the model's vision tower and passed, together with a report-generation prompt, to the language model. The base weights are first specialized with the SFT LoRA adapter and then with the GRPO LoRA adapter, so the deployed model reflects both supervised and reinforcement-learning training. The model generates the report text, which BisQue saves as a downloadable file resource attached to the run.

Generation is deterministic-leaning for clinical consistency: half-precision (fp16) inference, up to 1024 new tokens, temperature 0.2, top-p 0.9.

## Inputs and outputs

| | |
|---|---|
| **Input** | **CXR Image** — a single chest X-ray image resource |
| **Output** | **Generated Report** — a `.txt` file containing the generated radiology report |

The module processes one image per run and can be iterated over a dataset of chest X-rays.

## Model

- **Base model:** LLaVA-Med v1.5 (Mistral-7B)
- **Adapter 1 — SFT LoRA:** supervised fine-tuning on chest-X-ray reports
- **Adapter 2 — GRPO LoRA:** reinforcement-learning refinement (GRPO), merged on top of SFT
- **Self-contained:** all weights ship inside the module image, so no external model download is required at runtime

## Requirements

An NVIDIA GPU (CUDA) is recommended — the 7B model runs in fp16 and is slow or memory-limited on CPU.

## Example output (format)

```
FINDINGS: The lungs are clear without focal consolidation. No pleural
effusion or pneumothorax. Heart size is normal.
IMPRESSION: No acute cardiopulmonary abnormality.
```

---

*Research/education tool. Generated reports are model output and are not a substitute for interpretation by a qualified radiologist.*
