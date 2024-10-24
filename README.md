# <img width="60" alt="image" src="https://github.com/OpenGVLab/InternVL/assets/8529570/5aa4cda8-b453-40a0-9336-17012b430ae8"> Train InternViT-6B in MMSegmentation and MMDetection with DeepSpeed

<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/930e6814-8a9f-43e1-a284-118a5732daa4">
  <br>
</div>

This repository contains our customized mmcv/mmsegmentation/mmdetection code, integrated with DeepSpeed, which can be used for training large-scale object detection and semantic segmentation models.

## What is InternVL?

\[[Paper](https://arxiv.org/abs/2312.14238)\]  \[[Chat Demo](https://internvl.opengvlab.com/)\] \[[Quick Start](#Installation)\]

InternVL scales up the ViT to _**6B parameters**_ and aligns it with LLM.

It is _**the largest open-source vision/vision-language foundation model (14B)**_ to date, achieving _**32 state-of-the-art**_ performances on a wide range of tasks such as visual perception, cross-modal retrieval, multimodal dialogue, etc.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internvl-scaling-up-vision-foundation-models/zero-shot-cross-modal-retrieval-on-coco-2014)](https://paperswithcode.com/sota/zero-shot-cross-modal-retrieval-on-coco-2014?p=internvl-scaling-up-vision-foundation-models)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internvl-scaling-up-vision-foundation-models/zero-shot-image-retrieval-on-coco-cn)](https://paperswithcode.com/sota/zero-shot-image-retrieval-on-coco-cn?p=internvl-scaling-up-vision-foundation-models)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internvl-scaling-up-vision-foundation-models/zero-shot-cross-modal-retrieval-on-flickr30k)](https://paperswithcode.com/sota/zero-shot-cross-modal-retrieval-on-flickr30k?p=internvl-scaling-up-vision-foundation-models)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internvl-scaling-up-vision-foundation-models/image-to-text-retrieval-on-flickr30k)](https://paperswithcode.com/sota/image-to-text-retrieval-on-flickr30k?p=internvl-scaling-up-vision-foundation-models)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internvl-scaling-up-vision-foundation-models/zero-shot-image-retrieval-on-flickr30k-cn)](https://paperswithcode.com/sota/zero-shot-image-retrieval-on-flickr30k-cn?p=internvl-scaling-up-vision-foundation-models)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internvl-scaling-up-vision-foundation-models/image-retrieval-on-flickr30k-cn)](https://paperswithcode.com/sota/image-retrieval-on-flickr30k-cn?p=internvl-scaling-up-vision-foundation-models)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internvl-scaling-up-vision-foundation-models/zero-shot-image-retrieval-on-xtd10)](https://paperswithcode.com/sota/zero-shot-image-retrieval-on-xtd10?p=internvl-scaling-up-vision-foundation-models)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internvl-scaling-up-vision-foundation-models/zero-shot-transfer-image-classification-on-cn)](https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-cn?p=internvl-scaling-up-vision-foundation-models)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internvl-scaling-up-vision-foundation-models/zero-shot-transfer-image-classification-on-8)](https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-8?p=internvl-scaling-up-vision-foundation-models)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internvl-scaling-up-vision-foundation-models/zero-shot-transfer-image-classification-on-6)](https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-6?p=internvl-scaling-up-vision-foundation-models)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internvl-scaling-up-vision-foundation-models/zero-shot-transfer-image-classification-on-5)](https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-5?p=internvl-scaling-up-vision-foundation-models)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internvl-scaling-up-vision-foundation-models/zero-shot-transfer-image-classification-on-3)](https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-3?p=internvl-scaling-up-vision-foundation-models)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internvl-scaling-up-vision-foundation-models/zero-shot-transfer-image-classification-on-1)](https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-1?p=internvl-scaling-up-vision-foundation-models)

## Performance

- Semantic Segmentation [\[see details\]](./mmsegmentation#-evaluation)

  | method                | decoder | #param (train/total) | crop size | mIoU         |
  | --------------------- | :-----: | :------------------: | :-------: | ------------ |
  | OpenCLIP-G (frozen)   | Linear  |     0.3M / 1.8B      |    512    | 39.3         |
  | ViT-22B (frozen)      | Linear  |     0.9M / 21.7B     |    504    | 34.6         |
  | InternViT-6B (frozen) | Linear  |     0.5M / 5.9B      |    504    | 47.2 (+12.6) |
  | ViT-22B (frozen)      | UperNet |     0.8B / 22.5B     |    504    | 52.7         |
  | InternViT-6B (frozen) | UperNet |     0.4B / 6.3B      |    504    | 54.9 (+2.2)  |
  | ViT-22B               | UperNet |    22.5B / 22.5B     |    504    | 55.3         |
  | InternViT-6B          | UperNet |     6.3B / 6.3B      |    504    | 58.9 (+3.6)  |

## Installation

> [!Warning]
> <div align="left">
> <b>
> 🚨 This codebase requires you to install a lower version of the environment (i.e., torch==1.12.0), which is different from our main repository's environment.
> </b>
> </div>

> [!Note]
> <div align="left">
> <b>
> 📝 On 2024/10/24, the environment was successfully installed and verified by following the installation instructions below.
> </b>
> </div>

- Clone this repo:

  ```bash
  git clone https://github.com/OpenGVLab/InternVL-MMDetSeg
  cd InternVL-MMDetSeg/
  ```

- Create a conda virtual environment and activate it:

  ```bash
  conda create -n internvl-mmdetseg python=3.9 -y
  conda activate internvl-mmdetseg
  ```

- Install `PyTorch>=1.11<2.0` and `torchvision>=0.13.0` with `CUDA>=10.2`:

  For example, to install torch==1.12.0 with CUDA==11.3:

  ```bash
  conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
  # or
  pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
  ```

- Install `flash-attn==0.2.8` :

  If you want to fully replicate my results, please install `v0.2.8`, otherwise install the latest version.

  This is because different versions of flash attention yield slight differences in results.

  ```bash
  git clone https://github.com/Dao-AILab/flash-attention.git
  cd flash-attention
  git checkout v0.2.8
  pip install ninja
  python setup.py install # I use gcc-7.3 to compile this package
  ```

- Install other requirements:

  ```bash
  conda install -c conda-forge termcolor yacs pyyaml scipy pip -y
  pip install opencv-python
  pip install timm==0.6.11
  pip install yapf==0.40.1
  pip install addict
  pip install numpy==1.26.3 # please install this old version
  pip install deepspeed==0.8.0 # please install this old version
  pip install pydantic==1.10.2 # later versions may have compatibility issues
  ```

- Install `tensorboard`:

  ```bash
  pip install future tensorboard
  ```

- Install our customized `mmcv-full==1.7.0`:

  ```bash
  cd mmcv/
  export MMCV_WITH_OPS=1
  python setup.py develop
  cd ../
  ```

- Install our customized mmsegmentation & mmdetection:

  ```bash
  cd mmsegmentation/
  python setup.py develop
  cd ../
  cd mmdetection/
  python setup.py develop
  cd ../
  ```

- Compile the deformable attention (optional):

  ```bash
  # if you want to use ViT-Adapter, please compile the deformable attention operator
  cd ops
  sh compile.sh
  # Soft link the `ops` folder to the mmsegmentation directory
  cd ../mmsegmentation/
  ln -s ../ops ./
  ```

## How to use?

The usage is basically consistent with that of common mmsegmentation and mmdetection. 

Please enter the corresponding folder to check README.

- [mmsegmentation](./mmsegmentation)

- [mmdetection](./mmdetection)

## Schedule

- [x] Release customized MMDetection
- [x] Release customized MMSegmentation
- [x] Release customized MMCV

## Citation

If you find this project useful in your research, please consider citing:

```BibTeX
@article{chen2023internvl,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and Li, Bin and Luo, Ping and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2312.14238},
  year={2023}
}
@article{chen2024far,
  title={How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites},
  author={Chen, Zhe and Wang, Weiyun and Tian, Hao and Ye, Shenglong and Gao, Zhangwei and Cui, Erfei and Tong, Wenwen and Hu, Kongzhi and Luo, Jiapeng and Ma, Zheng and others},
  journal={arXiv preprint arXiv:2404.16821},
  year={2024}
}
@article{gao2024mini,
  title={Mini-InternVL: A Flexible-Transfer Pocket Multimodal Model with 5\% Parameters and 90\% Performance},
  author={Gao, Zhangwei and Chen, Zhe and Cui, Erfei and Ren, Yiming and Wang, Weiyun and Zhu, Jinguo and Tian, Hao and Ye, Shenglong and He, Junjun and Zhu, Xizhou and others},
  journal={arXiv preprint arXiv:2410.16261},
  year={2024}
}
```
