# Nexels: Neurally-Textured Surfels for Real-Time Novel View Synthesis with Sparse Geometries

[Project Page](https://lessvrong.com/cs/nexels) | [Paper](https://arxiv.org/pdf/????.?????) | [CUDA Rasterizer](https://github.com/victor-rong/nexels_cuda) | [Custom Dataset (???GB)](TODO) <br>

This is the official repository for "Nexels: Neurally-Textured Surfels for Real-Time Novel View Synthesis with Sparse Geometries".

## Installation

First, create a conda environment.

```
    conda create -n nexels python=3.8 -y
    conda activate nexels
```

Install a suitable version of CUDA toolkit and [PyTorch](https://pytorch.org/get-started/locally/). This codebase has been tested with CUDA 11.8.

```
    conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
    pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

Our method uses the Instant-NGP architecture for the neural texture, which can be installed with

```
    pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

The other requirements, including the CUDA nexel rasterizer, can be installed via pip from `requirements.txt`.

```
    pip install --no-build-isolation -r requirements.txt
```

## Training

Training scripts have been prepared to reproduce the paper's results on the datasets. Run

```
    python scripts/eval_m360.py --start 0 --end 9 --data_dir ... --cap_max ...
```

to train nexel models on all nine scenes of the MipNeRF-360 dataset with a set amount of primitives.