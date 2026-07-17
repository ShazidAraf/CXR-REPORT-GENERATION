# ==================================================================
# BisQue analysis module: BISQUE (CXR report generation)
# ------------------------------------------------------------------
# Base model + LoRA adapters ship INSIDE the image (src/CKPT/*), so the
# container is self-contained: no HuggingFace download at runtime.
#
# Build (note the registry prefix BisQue prepends when pulling locally):
#   docker build -t biodev.ece.ucsb.edu:5000/cxr_report:v1.0.0 .
# ==================================================================

# CUDA 12.4 runtime matches torch==2.5.1+cu124 in requirements.txt.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124

# python3.10 + git (for the LLaVA-Med editable install in requirements.txt)
# + libgl1/libglib2.0-0 for OpenCV / PIL backends pulled in by the deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip python3.10-venv \
        git build-essential \
        libgl1 libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && python -m pip install --upgrade pip

RUN mkdir /module
WORKDIR /module

# ===================Module Dependencies============================
# Install the model's pip deps first so this layer caches across code edits.
# requirements.txt carries one host-only entry (`packaging @ file:///...`)
# that won't resolve in a clean container -> strip it and let pip choose.
COPY requirements.txt /module/requirements.txt
RUN sed -E '/^packaging\s*@/d' /module/requirements.txt > /module/requirements.docker.txt \
    && pip install -r /module/requirements.docker.txt

# ===============bqapi for python3 dependencies=====================
RUN pip install six lxml requests-toolbelt

# ===================Copy Source Code===============================
# User code + LoRA adapters + base model weights (self-contained).
COPY src /module/src
# Sample CXRs for in-container smoke testing.
COPY samples /module/samples

# ===================BisQue plumbing================================
COPY PythonScriptWrapper.py /module/
COPY bqapi/ /module/bqapi
COPY public /module/public
COPY CXR_report.xml /module/CXR_report.xml

ENV PATH=/module:$PATH:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHONPATH=$PYTHONPATH:/module:/module/src

# BisQue invokes:
#   docker run ... <image> python PythonScriptWrapper.py --mex_url <...> --bisque_token <...>
CMD ["python", "PythonScriptWrapper.py"]
