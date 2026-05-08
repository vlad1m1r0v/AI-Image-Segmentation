# Backend setup

## Prerequisites

- Python 3.13
- [Poetry](https://python-poetry.org/docs/#installation) ≥ 2.0
- Git (for the SAM-2 dependency)

## Install

```bash
cd backend
poetry install
```

`poetry install` resolves everything from the committed `poetry.lock` — no extra flags
needed. Torch is installed from the CPU-only PyTorch wheel index (~190 MB, no CUDA
packages).

## Download model weights

Model weights are gitignored (SAM-2 checkpoint is ~180 MB). Download once into
`backend/models/`:

```bash
mkdir -p models

wget -O models/sam2_hiera_small.pt \
  https://huggingface.co/facebook/sam2-hiera-small/resolve/main/sam2_hiera_small.pt

wget -O models/sam2_hiera_s.yaml \
  https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2/sam2_hiera_s.yaml
```

## Run the POC

```bash
# place image.png in backend/ first
poetry run python poc.py
# → output.png with transparent background
```

---

## Regenerating poetry.lock

Only needed when you add, remove, or update a dependency. The lock step requires
`PIP_INDEX_URL` to be set so that the pip environment poetry creates while building the
`sam-2` git dependency picks up the CPU-only torch wheel instead of the CUDA one from
PyPI (the CUDA build deps total ~5 GB and overflow the system's 1.6 GB `/tmp` tmpfs).

```bash
mkdir -p /home/$(whoami)/tmp_pip

TMPDIR=/home/$(whoami)/tmp_pip \
PIP_INDEX_URL=https://download.pytorch.org/whl/cpu \
PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
poetry lock

# After locking, a plain `poetry install` is enough again:
poetry install
```

Commit the updated `poetry.lock` so teammates can just run `poetry install`.
