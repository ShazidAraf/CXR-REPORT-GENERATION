# Testing the CXR_report module on a local Bisque server

This module follows the [BQ_module_generator](https://github.com/ivanfarevalo/BQ_module_generator)
convention. The folder is **self-contained**: the base model + LoRA adapters
live under `src/CKPT/` (≈16 GB), so no HuggingFace download is needed at runtime.

```
CXR_report/
├── CXR_report.xml              # module manifest (input "CXR Image" -> output "Generated Report")
├── PythonScriptWrapper.py  # BisQue <-> module bridge (standard)
├── bqapi/                  # BisQue python3 client (standard)
├── runtime-module.cfg      # docker.image = cxr_report:v1.0.0
├── Dockerfile
├── public/                 # help.html, help.md, thumbnail.jpg
└── src/
    ├── BQ_run_module.py     # run_module(input_path_dict, output_folder_path)
    ├── inference.py         # load_pipeline() + generate_report()
    ├── config.json          # checkpoint paths (relative to src/)
    └── CKPT/{BASE,SFT,GRPO}  # self-contained weights
```

---

## Step 0 — (on the GPU cluster) verify the logic works, no Bisque

```bash
# GPU node, cxr_report env
cd <...>/CXR_report
python -m src.BQ_run_module samples/CXR1701_IM-0462.png
```
Expect: `[load] base = .../src/CKPT/BASE/...` then a printed report and
`[saved to <stem>_report.txt]`.

---

## Step 1 — copy the module onto the Docker machine

The Docker machine should ideally have an NVIDIA GPU + nvidia-container-toolkit
(the 7B model is slow/OOM on CPU). Place the module under a `Modules/` parent
that contains ONLY modules you want Bisque to see.

```bash
mkdir -p ~/Bisque/Modules
# 16 GB transfer (rsync resumes if interrupted):
rsync -avP <cluster-host>:<...>/CXR_report ~/Bisque/Modules/
```

## Step 2 — build the module image

Bisque prepends `biodev.ece.ucsb.edu:5000/` when it pulls, so build with that
prefix. Image names must be lowercase.

```bash
cd ~/Bisque/Modules/CXR_report
docker build -t biodev.ece.ucsb.edu:5000/cxr_report:v1.0.0 .
```
`runtime-module.cfg` already points at `docker.image = cxr_report:v1.0.0`.

(Optional) sanity-check the module in the image directly:
```bash
docker run --rm --gpus all biodev.ece.ucsb.edu:5000/cxr_report:v1.0.0 \
    python -m src.BQ_run_module samples/CXR1701_IM-0462.png
```

## Step 3 — run the Bisque dev server

```bash
cd ~/Bisque/Modules
docker pull amilworks/bisque-module-dev:git
docker run --name bisque --rm -p 8080:8080 \
    -v $(pwd):/source/modules \
    -v /var/run/docker.sock:/var/run/docker.sock \
    amilworks/bisque-module-dev:git
```
`-v $(pwd):/source/modules` mounts your modules; the docker.sock mount lets
Bisque launch module containers.

## Step 4 — log in and register the module

1. Browse to `http://localhost:8080` (or `http://<private-ip>:8080`).
   Login: `admin` / `admin`.
2. `Upload` -> choose a chest X-ray -> `Upload`.
3. Top-right `Bisque admin -> Module Manager`. In `Engine Modules`, set the
   Engine URL to `http://<private-ip>:8080/engine_service` and click `Load`.
4. Drag `CXR_report` from the right panel to the left to register it.

## Step 5 — run it

`Analyse` -> `CXR_report` -> select the uploaded CXR -> `Run`. The generated report
(.txt) appears in the results with a download link.

---

## Debugging

* Logs per run live inside the Bisque container at `/source/staging/<mex_id>/`
  (`docker_run.log`, `PythonScript.log`).
* Module not listed: confirm the Engine URL, that `~/Bisque/Modules` is mounted,
  and that `CXR_report.xml` and the `CXR_report/` folder share the same name.
* GPU: if the module container has no GPU, loading the 7B model is very slow and
  may OOM. Ensure nvidia-container-toolkit is installed on the Docker host.
