# InternViT-6B for Semantic Segmentation

This folder contains the implementation of the InternViT-6B for semantic segmentation.

Our segmentation code is developed on top of [MMSegmentation v0.27.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.27.0).

## 📦 Data Preparation

Prepare the ADE20K dataset according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.

## 📦 Model Preparation

| model name              | date       | type    | download                                                                                       | size  |
| ----------------------- | ---------- | ------- | ---------------------------------------------------------------------------------------------- |:-----:|
| InternViT-6B-224px      | 2023.12.22 | pytorch | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL/blob/main/intern_vit_6b_224px.pth)      | 12 GB |
| InternViT-6B-448px-V1-0 | 2024.01.30 | pytorch | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL/blob/main/intern_vit_6b_448px_v1_0.pth) | 12 GB |
| InternViT-6B-448px-V1-2 | 2024.02.11 | pytorch | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL/blob/main/intern_vit_6b_448px_v1_2.pth) | 11 GB |
| InternViT-6B-448px-V1-5 | 2024.04.20 | pytorch | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL/blob/main/intern_vit_6b_448px_v1_5.pth) | 11 GB |
| InternViT-6B-448px-V2-5 | 2024.10.24 | pytorch | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL/blob/main/intern_vit_6b_448px_v2_5.pth) | 11 GB |

Please download the above model weight and place it in the `pretrained/` folder.

```sh
mkdir pretrained & cd pretrained
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/intern_vit_6b_224px.pth
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/intern_vit_6b_448px_v1_0.pth
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/intern_vit_6b_448px_v1_2.pth
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/intern_vit_6b_448px_v1_5.pth
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/intern_vit_6b_448px_v2_5.pth
```

The directory structure is:

```sh
pretrained
├── intern_vit_6b_224px.pth
├── intern_vit_6b_448px_v1_0.pth
├── intern_vit_6b_448px_v1_2.pth
├── intern_vit_6b_448px_v1_5.pth
└── intern_vit_6b_448px_v2_5.pth
```

## 🔥 Training

To train a linear classifier for `InternViT-6B-224px` with 8 GPU on 1 node (total batch size 16), run:

```bash
sh tools/dist_train.sh configs/intern_vit_6b/linear_probing/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.py 8
# or manage jobs with slurm
GPUS=8 sh tools/slurm_train.sh <partition> <job-name> configs/intern_vit_6b/linear_probing/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.py
```

To train a UperNet with `InternViT-6B-224px` with 8 GPU on 1 node (total batch size 16), run:

```bash
sh tools/dist_train.sh configs/intern_vit_6b/full_tuning/upernet_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5.py 8
# or manage jobs with slurm
GPUS=8 sh tools/slurm_train.sh <partition> <job-name> configs/intern_vit_6b/full_tuning/upernet_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5.py
```

## 📊 Evaluation

| type            | backbone                    | head    | mIoU | config                                                                                                     | download                                                                                                                                                                                                                                            |
| --------------- | --------------------------- |:-------:|:----:|:----------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| few-shot (1/16) | InternViT-6B-224px          | Linear  | 46.5 | [config](./configs/intern_vit_6b/few_shot/linear_intern_vit_6b_504_5k_ade20k_bs16_lr4e-5_1of16.py)         | [ckpt](https://huggingface.co/OpenGVLab/InternVL/resolve/main/linear_intern_vit_6b_504_5k_ade20k_bs16_lr4e-5_1of16.pth) \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/linear_intern_vit_6b_504_5k_ade20k_bs16_lr4e-5_1of16.log)       |
| few-shot (1/8)  | InternViT-6B-224px          | Linear  | 50.0 | [config](./configs/intern_vit_6b/few_shot/linear_intern_vit_6b_504_10k_ade20k_bs16_lr4e-5_1of8.py)         | [ckpt](https://huggingface.co/OpenGVLab/InternVL/resolve/main/linear_intern_vit_6b_504_10k_ade20k_bs16_lr4e-5_1of8.pth) \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/linear_intern_vit_6b_504_10k_ade20k_bs16_lr4e-5_1of8.log)       |
| few-shot (1/4)  | InternViT-6B-224px          | Linear  | 53.3 | [config](./configs/intern_vit_6b/few_shot/linear_intern_vit_6b_504_20k_ade20k_bs16_lr4e-5_1of4.py)         | [ckpt](https://huggingface.co/OpenGVLab/InternVL/resolve/main/linear_intern_vit_6b_504_20k_ade20k_bs16_lr4e-5_1of4.pth) \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/linear_intern_vit_6b_504_20k_ade20k_bs16_lr4e-5_1of4.log)       |
| few-shot (1/2)  | InternViT-6B-224px          | Linear  | 55.8 | [config](./configs/intern_vit_6b/few_shot/linear_intern_vit_6b_504_40k_ade20k_bs16_lr4e-5_1of2.py)         | [ckpt](https://huggingface.co/OpenGVLab/InternVL/resolve/main/linear_intern_vit_6b_504_40k_ade20k_bs16_lr4e-5_1of2.pth) \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/linear_intern_vit_6b_504_40k_ade20k_bs16_lr4e-5_1of2.log)       |
| few-shot (1/1)  | InternViT-6B-224px          | Linear  | 57.2 | [config](./configs/intern_vit_6b/few_shot/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_1of1.py)         | [ckpt](https://huggingface.co/OpenGVLab/InternVL/resolve/main/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_1of1.pth) \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_1of1.log)       |
| linear probing  | InternViT-6B-224px (frozen) | Linear  | 47.2 | [config](./configs/intern_vit_6b/linear_probing/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.py) | [ckpt](https://huggingface.co/OpenGVLab/InternVL/resolve/main/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.pth) \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.log)   |
| head tuning     | InternViT-6B-224px (frozen) | UperNet | 54.9 | [config](./configs/intern_vit_6b/head_tuning/upernet_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.py)   | [ckpt](https://huggingface.co/OpenGVLab/InternVL/resolve/main/upernet_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.pth) \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/upernet_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.log) |
| full tuning     | InternViT-6B-224px          | UperNet | 58.9 | [config](./configs/intern_vit_6b/full_tuning/upernet_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5.py)          | [ckpt](https://huggingface.co/OpenGVLab/InternVL/resolve/main/upernet_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5.pth) \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/upernet_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5.log)               |
| adapter tuning  | InternViT-6B-224px-Adapter  | UperNet | 58.1 | [config](./configs/intern_vit_6b/vit_adapter/upernet_intern_vit_adapter_6b_512_80k_ade20k_bs16_lr4e-5.py)  | ckpt \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/upernet_intern_vit_adapter_6b_512_80k_ade20k_bs16_lr4e-5.log)                                                                                                                      |

You can download checkpoints from [here](https://huggingface.co/OpenGVLab/InternVL/tree/main) or from the table above. Then place them to `checkpoints/`.

For example, to evaluate the `InternViT-6B-224px` with a single GPU:

```bash
python tools/test.py configs/intern_vit_6b/linear_probing/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.py checkpoints/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.pth --eval mIoU
```

For example, to evaluate the `InternViT-6B-224px` with a single node with 8 GPUs:

```bash
sh tools/dist_test.sh configs/intern_vit_6b/linear_probing/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.py checkpoints/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.pth 8 --eval mIoU
```

---------------------

<div align="center">
  <img src="resources/mmseg-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

<br />

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmsegmentation)](https://pypi.org/project/mmsegmentation/)
[![PyPI](https://img.shields.io/pypi/v/mmsegmentation)](https://pypi.org/project/mmsegmentation)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmsegmentation.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmsegmentation/workflows/build/badge.svg)](https://github.com/open-mmlab/mmsegmentation/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmsegmentation/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmsegmentation)
[![license](https://img.shields.io/github/license/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/blob/master/LICENSE)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)

[📘Documentation](https://mmsegmentation.readthedocs.io/en/latest/) |
[🛠️Installation](https://mmsegmentation.readthedocs.io/en/latest/get_started.html) |
[👀Model Zoo](https://mmsegmentation.readthedocs.io/en/latest/model_zoo.html) |
[🆕Update News](https://mmsegmentation.readthedocs.io/en/latest/changelog.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmsegmentation/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

## Introduction

MMSegmentation is an open source semantic segmentation toolbox based on PyTorch.
It is a part of the [OpenMMLab](https://openmmlab.com/) project.

The master branch works with **PyTorch 1.5+**.

![demo image](resources/seg_demo.gif)

<details open>
<summary>Major features</summary>

- **Unified Benchmark**
  
  We provide a unified benchmark toolbox for various semantic segmentation methods.

- **Modular Design**
  
  We decompose the semantic segmentation framework into different components and one can easily construct a customized semantic segmentation framework by combining different modules.

- **Support of multiple methods out of box**
  
  The toolbox directly supports popular and contemporary semantic segmentation frameworks, *e.g.* PSPNet, DeepLabV3, PSANet, DeepLabV3+, etc.

- **High efficiency**
  
  The training speed is faster than or comparable to other codebases.

</details>

## What's New

v0.27.0 was released in 7/28/2022:

- Add Swin-L Transformer models
- Update ERFNet result on Cityscapes

Please refer to [changelog.md](docs/en/changelog.md) for details and release history.

## Installation

Please refer to [get_started.md](docs/en/get_started.md#installation) for installation and [dataset_prepare.md](docs/en/dataset_prepare.md#prepare-datasets) for dataset preparation.

## Get Started

Please see [train.md](docs/en/train.md) and [inference.md](docs/en/inference.md) for the basic usage of MMSegmentation.
There are also tutorials for:

- [customizing dataset](docs/en/tutorials/customize_datasets.md)
- [designing data pipeline](docs/en/tutorials/data_pipeline.md)
- [customizing modules](docs/en/tutorials/customize_models.md)
- [customizing runtime](docs/en/tutorials/customize_runtime.md)
- [training tricks](docs/en/tutorials/training_tricks.md)
- [useful tools](docs/en/useful_tools.md)

A Colab tutorial is also provided. You may preview the notebook [here](demo/MMSegmentation_Tutorial.ipynb) or directly [run](https://colab.research.google.com/github/open-mmlab/mmsegmentation/blob/master/demo/MMSegmentation_Tutorial.ipynb) on Colab.

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/en/model_zoo.md).

Supported backbones:

- [x] ResNet (CVPR'2016)
- [x] ResNeXt (CVPR'2017)
- [x] [HRNet (CVPR'2019)](configs/hrnet)
- [x] [ResNeSt (ArXiv'2020)](configs/resnest)
- [x] [MobileNetV2 (CVPR'2018)](configs/mobilenet_v2)
- [x] [MobileNetV3 (ICCV'2019)](configs/mobilenet_v3)
- [x] [Vision Transformer (ICLR'2021)](configs/vit)
- [x] [Swin Transformer (ICCV'2021)](configs/swin)
- [x] [Twins (NeurIPS'2021)](configs/twins)
- [x] [BEiT (ICLR'2022)](configs/beit)
- [x] [ConvNeXt (CVPR'2022)](configs/convnext)
- [x] [MAE (CVPR'2022)](configs/mae)

Supported methods:

- [x] [FCN (CVPR'2015/TPAMI'2017)](configs/fcn)
- [x] [ERFNet (T-ITS'2017)](configs/erfnet)
- [x] [UNet (MICCAI'2016/Nat. Methods'2019)](configs/unet)
- [x] [PSPNet (CVPR'2017)](configs/pspnet)
- [x] [DeepLabV3 (ArXiv'2017)](configs/deeplabv3)
- [x] [BiSeNetV1 (ECCV'2018)](configs/bisenetv1)
- [x] [PSANet (ECCV'2018)](configs/psanet)
- [x] [DeepLabV3+ (CVPR'2018)](configs/deeplabv3plus)
- [x] [UPerNet (ECCV'2018)](configs/upernet)
- [x] [ICNet (ECCV'2018)](configs/icnet)
- [x] [NonLocal Net (CVPR'2018)](configs/nonlocal_net)
- [x] [EncNet (CVPR'2018)](configs/encnet)
- [x] [Semantic FPN (CVPR'2019)](configs/sem_fpn)
- [x] [DANet (CVPR'2019)](configs/danet)
- [x] [APCNet (CVPR'2019)](configs/apcnet)
- [x] [EMANet (ICCV'2019)](configs/emanet)
- [x] [CCNet (ICCV'2019)](configs/ccnet)
- [x] [DMNet (ICCV'2019)](configs/dmnet)
- [x] [ANN (ICCV'2019)](configs/ann)
- [x] [GCNet (ICCVW'2019/TPAMI'2020)](configs/gcnet)
- [x] [FastFCN (ArXiv'2019)](configs/fastfcn)
- [x] [Fast-SCNN (ArXiv'2019)](configs/fastscnn)
- [x] [ISANet (ArXiv'2019/IJCV'2021)](configs/isanet)
- [x] [OCRNet (ECCV'2020)](configs/ocrnet)
- [x] [DNLNet (ECCV'2020)](configs/dnlnet)
- [x] [PointRend (CVPR'2020)](configs/point_rend)
- [x] [CGNet (TIP'2020)](configs/cgnet)
- [x] [BiSeNetV2 (IJCV'2021)](configs/bisenetv2)
- [x] [STDC (CVPR'2021)](configs/stdc)
- [x] [SETR (CVPR'2021)](configs/setr)
- [x] [DPT (ArXiv'2021)](configs/dpt)
- [x] [Segmenter (ICCV'2021)](configs/segmenter)
- [x] [SegFormer (NeurIPS'2021)](configs/segformer)
- [x] [K-Net (NeurIPS'2021)](configs/knet)

Supported datasets:

- [x] [Cityscapes](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#cityscapes)
- [x] [PASCAL VOC](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-voc)
- [x] [ADE20K](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#ade20k)
- [x] [Pascal Context](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-context)
- [x] [COCO-Stuff 10k](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#coco-stuff-10k)
- [x] [COCO-Stuff 164k](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#coco-stuff-164k)
- [x] [CHASE_DB1](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#chase-db1)
- [x] [DRIVE](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#drive)
- [x] [HRF](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#hrf)
- [x] [STARE](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#stare)
- [x] [Dark Zurich](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#dark-zurich)
- [x] [Nighttime Driving](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#nighttime-driving)
- [x] [LoveDA](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#loveda)
- [x] [Potsdam](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#isprs-potsdam)
- [x] [Vaihingen](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#isprs-vaihingen)
- [x] [iSAID](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#isaid)

## FAQ

Please refer to [FAQ](docs/en/faq.md) for frequently asked questions.

## Contributing

We appreciate all contributions to improve MMSegmentation. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMSegmentation is an open source project that welcome any contribution and feedback.
We wish that the toolbox and benchmark could serve the growing research
community by providing a flexible as well as standardized toolkit to reimplement existing methods
and develop their own new semantic segmentation methods.

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```

## License

MMSegmentation is released under the Apache 2.0 license, while some specific features in this library are with other licenses. Please refer to [LICENSES.md](LICENSES.md) for the careful check, if you are using our code for commercial matters.

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab Model Deployment Framework.
