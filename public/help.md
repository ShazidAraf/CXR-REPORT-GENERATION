# BISQUE — CXR Report Generator

Generate a radiology report from a chest X-ray using **LLaVA-Med v1.5 (Mistral-7B)**
fine-tuned with **SFT + GRPO** LoRA adapters.

## Inputs

* **CXR Image** — a single chest X-ray image resource.

## Outputs

* **Generated Report** — a `.txt` file containing the generated radiology
  report, describing both abnormal and normal findings across the visible
  anatomical structures.

## How it works

1. The base model `llava-med-v1.5-mistral-7b` is loaded from the
   self-contained weights shipped inside the module (`src/CKPT/BASE`).
2. The **SFT** LoRA adapter is merged, then the **GRPO** LoRA adapter on top.
3. The X-ray is passed through the model with a report-generation prompt and
   the resulting report is uploaded back to BisQue as a file resource.

## Notes

* Generation parameters (prompt, temperature, top_p, max_new_tokens) are set
  in `src/config.json`.
* A GPU is recommended; the 7B model runs in fp16.
