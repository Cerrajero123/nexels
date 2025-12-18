# Nexels: Neurally-Textured Surfels for Real-Time Novel View Synthesis with Sparse Geometries

[Project Page](https://lessvrong.com/cs/nexels) | [Paper](https://arxiv.org/pdf/2512.13796) | [CUDA Rasterizer](https://github.com/victor-rong/nexels_cuda) | [Custom Dataset (15.7GB)](https://www.dropbox.com/scl/fi/oqwi15avd80e0tt3cvy98/data.zip?rlkey=xcqi1klskg9petiwxs219ylov&st=a6gtrthj&dl=0) <br>

<img src="./assets/teaser.jpg" alt="Teaser figure" width="100%"/>

This is the official repository for "Nexels: Neurally-Textured Surfels for Real-Time Novel View Synthesis with Sparse Geometries".

## Setup

First, clone the repository. Note that we have a number of submodules, so you should include the recursive flag.

```
git clone git@github.com:victor-rong/nexels.git --recursive
```

To install, create a conda environment.

```
conda create -n nexels python=3.8 -y
conda activate nexels
```

A suitable version of CUDA toolkit and [PyTorch](https://pytorch.org/get-started/locally/) is needed. This codebase has been tested with CUDA 11.8.

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

## Running the Method

Our method works on scenes preprocessed with COLMAP (see [these instructions](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes) on setting up custom datasets). To train a nexels model with the recommended settings, run

```
python train.py -s $DATA_DIR -m $OUTPUT_DIR
```

To set the maximum number of primitives, you can set the `--cap_max_init` and `--cap_max_final` flags. The former determines the initial number of primitives and the latter is the maximum number of primitives by the end of training. Setting the initial amount to be half the final amount works consistently well. It may also be helpful to adjust `--log_hash_table_size`, which determines the amount of parameters used for the neural field. Increasing this generally improves results if memory is available.

The model can then be rendered across the train and test sets, as well as an estimated ellipse trajectory with

```
python render.py -s $DATA_DIR -m $OUTPUT_DIR
```

A browser-based viewer will be released soon.

## Paper Results

Training scripts have been prepared to reproduce the paper's results on the datasets. Run

```
python scripts/eval_m360.py --start 0 --end 9 --data_dir $DATA_DIR --cap_max $CAP_MAX
python scripts/eval_custom.py --start 0 --end 4 --data_dir $DATA_DIR --cap_max $CAP_MAX
python scripts/eval_tntdb.py --start 0 --end 2 --data_dir $DATA_DIR --cap_max $CAP_MAX
```

to train nexel models on the MipNeRF-360, custom, and Tanks & Temples datasets with a maximum amount of primitives, `$CAP_MAX`.

## Acknowledgements

This method was inspired by several great works. The codebase is built off of the [original 3DGS code](https://github.com/graphdeco-inria/gaussian-splatting) from Kerbl et al. We also used portions of [gsplat](https://docs.gsplat.studio/main/) from the Nerfstudio team and [2DGS](https://surfsplatting.github.io/) by Huang et al.

## Citation

If you find this repository useful in your projects or papers, please consider citing our paper:
```
@article{rong2025nexels,
    title={Nexels: Neurally-textured surfels for real-time novel view synthesis with sparse geometries},
    author={Rong, Victor and Held, Jan and Chu, Victor and Rebain, Daniel and
        Van Droogenbroeck, Marc and Kutulakos, Kiriakos N and Tagliasacchi, Andrea and Lindell, David B},
    journal={arXiv preprint arXiv:2512.13796},
    year={2025}
}
```