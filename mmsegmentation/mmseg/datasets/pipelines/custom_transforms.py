import numpy as np
from PIL import Image, ImageOps, ImageFilter
from torch.utils.data.dataset import Dataset
import random
from mmcv.utils import deprecated_api_warning
from ..builder import PIPELINES
from torchvision import transforms


@PIPELINES.register_module()
class IRSatVideoLEOTransforms(object):
    """https://github.com/XinyiYing/RFR/blob/main/codes/dataset.py

    Args:
        object (_type_): _description_
    """

    def __init__(
        self,
        patch_size=128,
        pos_prob=0.5,
        test_mode=False,
        img_norm_cfg={'mean': 72.1040267944336, 'std': 12.302865028381348},
    ):
        self.patch_size = patch_size
        self.pos_prob = pos_prob
        self.test_mode = test_mode
        self.img_norm_cfg = img_norm_cfg

    def __call__(self, results):

        if not self.test_mode:
            img_seq = results["img"]  # # shape: (T, H, W)
            mask_seq = results["gt_semantic_seg"]  # shape: (T, H, W)
            img_seq = img_seq.astype(np.float32)

            # for idx, img in enumerate(img_seq):
            #     img_seq[idx] = self.normalized(img, self.img_norm_cfg)

            img_seq = self.normalized(img_seq, self.img_norm_cfg)

            img_seq, mask_seq = self.random_crop_seq(
                img_seq, mask_seq, self.patch_size, self.pos_prob
            )
            img_seq, mask_seq = self.augumentation(img_seq, mask_seq)
            # T H W -> T 1 H W
            img_seq, mask_seq = (
                img_seq[:, np.newaxis, :, :],
                mask_seq[:, np.newaxis, :, :],
            )
            results["img"] = img_seq.astype(np.float32)
            results["gt_semantic_seg"] = mask_seq
        else:
            img = results["img"]  # # shape: (H, W)
            img = self.normalized(img, self.img_norm_cfg)

            results["img"] = img.astype(np.float32)

        return results

    def normalized(self, img, img_norm_cfg):
        return (img - img_norm_cfg['mean']) / img_norm_cfg['std']

    def random_crop_seq(self, img_seq, mask_seq, patch_size=128, pos_prob=False):
        _, h, w = img_seq.shape
        if min(h, w) < patch_size:
            for i in range(len(img_seq)):
                img_seq[i, :, :] = np.pad(
                    img_seq[i, :, :],
                    ((0, 0), (0, max(h, patch_size) - h), (0, max(w, patch_size) - w)),
                    mode='constant',
                )
                mask_seq[i, :, :] = np.pad(
                    mask_seq[i, :, :],
                    ((0, 0), (0, max(h, patch_size) - h), (0, max(w, patch_size) - w)),
                    mode='constant',
                )
                _, h, w = img_seq.shape

        cur_prob = random.random()

        if pos_prob == None or cur_prob > pos_prob or mask_seq.max() == 0:
            h_start = random.randint(0, h - patch_size)
            w_start = random.randint(0, w - patch_size)
        else:
            loc = np.where(mask_seq > 0)
            if len(loc[0]) <= 1:
                idx = 0
            else:
                idx = random.randint(0, len(loc[0]) - 1)
            h_start = random.randint(
                max(0, loc[1][idx] - patch_size), min(loc[1][idx], h - patch_size)
            )
            w_start = random.randint(
                max(0, loc[2][idx] - patch_size), min(loc[2][idx], w - patch_size)
            )

        h_end = h_start + patch_size
        w_end = w_start + patch_size
        img_patch_seq = img_seq[:, h_start:h_end, w_start:w_end]
        mask_patch_seq = mask_seq[:, h_start:h_end, w_start:w_end]

        return img_patch_seq, mask_patch_seq

    def augumentation(self, input, target):
        if random.random() < 0.5:  # A. 垂直翻转（上下翻转）
            input = input[:, ::-1, :]
            target = target[:, ::-1, :]
        if random.random() < 0.5:  # B. 水平翻转（左右翻转）
            input = input[:, :, ::-1]
            target = target[:, :, ::-1]
        if random.random() < 0.5:  # C. 通道翻转（可能是 RGB → BGR）
            input = input[::-1, :, :]
            target = target[::-1, :, :]
        if random.random() < 0.5:  # D. 沿主对角线转置（H <=> W）
            input = input.transpose(0, 2, 1)
            target = target.transpose(0, 2, 1)

        return input, target


@PIPELINES.register_module()
class DNANetTransforms(object):
    """GitHub link: https://github.com/YeRen123455/Infrared-Small-Target-Detection/blob/master/model/utils.py
        DNANet paper link: https://arxiv.org/pdf/2106.00487

    Added key is "img_noise_cfg".

    Args:
        L (int): The shape parameter for the gamma distribution.
        scale (float): The scaling factor for the noise, default is 1.0.
    """

    def __init__(
        self,
        test_mode: bool = False,
        base_size: int = 256,
        crop_size: int = 256,
        transform: bool = True,
    ):
        self.test_mode = test_mode
        self.base_size = base_size
        self.crop_size = crop_size

        # Preprocess and load data
        # self.transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )
        self.transform = transform
        # ImageNet 的 mean 和 std (R, G, B)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, results):
        # img = results["img"].astype(
        #     np.float32
        # )  # Ensure the image is float32 for processing
        img = results["img"]
        mask = results["gt_semantic_seg"]
        if not self.test_mode:
            img, mask = self._sync_transform(img, mask)
        else:
            img, mask = self._testval_sync_transform(img, mask)

        if self.transform:
            # array H, W, C -> tensor C, H, W
            img = self.transforms(img)

        results["img"] = img
        results["gt_semantic_seg"] = mask

        return results

    def transforms(self, img):
        """
        对 numpy 格式的 RGB 图像 (H, W, C) 进行归一化，保持 shape 不变。
        mean/std: 3 个通道的均值和标准差，列表形式。
        """
        img_norm = img.astype(np.float32) / 255.0  # 转为 0~1 区间
        mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.std, dtype=np.float32).reshape(1, 1, 3)
        img_norm = (img_norm - mean) / std

        return img_norm

    def _sync_transform(self, img, mask):

        # 假设 img 是 (H, W, 3)，mask 可能是 (H, W) 或 (H, W, 3)
        img = img.astype(np.uint8)
        mask = mask.astype(np.uint8)
        
        # 如果mask是多通道，转换为单通道
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]  # 取第一个通道
        
        # background 0, target 255 -> 1
        # mask[mask == 255] = 1
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        img, mask = np.array(img), np.array(mask, dtype=np.float32)

        return img, mask

    def _testval_sync_transform(self, img, mask):

        # 假设 img 是 (H, W, 3)，mask 可能是 (H, W) 或 (H, W, 3) 或 (H, W, 4)
        
        # 如果mask是多通道，转换为单通道
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]  # 取第一个通道
            
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        # final transform
        img, mask = np.array(img), np.array(
            mask, dtype=np.float32
        )  # img: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)

        return img, mask
