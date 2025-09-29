"""DNANet decoder head."""
import torch
import torch.nn as nn

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_planes = max(in_planes // ratio, 1)
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class VGGCBAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.relu(out)
        return out


class ResCBAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


def _build_block(block_type):
    block_type = block_type.lower()
    if 'res' in block_type:
        return ResCBAMBlock
    if 'vgg' in block_type:
        return VGGCBAMBlock
    raise ValueError(f'Unsupported block type: {block_type}')


@HEADS.register_module()
class DNANetDecodeHead(BaseDecodeHead):
    """Decode head that implements the decoder of DNANet."""

    def __init__(
        self,
        decoder_channels=(16, 32, 64, 128),
        num_blocks=(2, 2, 2),
        block_type='res_cbam',
        **kwargs,
    ):
        if 'input_transform' not in kwargs:
            kwargs['input_transform'] = 'multiple_select'
        if 'channels' not in kwargs:
            kwargs['channels'] = decoder_channels[0]
        super().__init__(**kwargs)

        if len(decoder_channels) != 4:
            raise AssertionError('DNANet decoder expects four feature levels.')
        if len(self.in_channels) != len(decoder_channels):
            raise AssertionError(
                'The length of in_channels and decoder_channels must match.'
            )
        # if len(num_blocks) != len(decoder_channels) - 1:
        #     raise AssertionError(
        #         'num_blocks should have len(decoder_channels) - 1 elements.'
        #     )

        self.decoder_channels = decoder_channels
        block = _build_block(block_type)

        self.stem_convs = nn.ModuleList()
        self.stem_convs.append(self._make_layer(block, self.in_channels[0], decoder_channels[0]))
        for idx in range(1, len(decoder_channels)):
            num_block = num_blocks[idx - 1]
            self.stem_convs.append(
                self._make_layer(block, self.in_channels[idx], decoder_channels[idx], num_block)
            )

        self.conv0_1 = self._make_layer(
            block, decoder_channels[0] + decoder_channels[1], decoder_channels[0]
        )
        self.conv1_1 = self._make_layer(
            block,
            decoder_channels[1] + decoder_channels[2] + decoder_channels[0],
            decoder_channels[1],
            num_blocks[0],
        )
        self.conv2_1 = self._make_layer(
            block,
            decoder_channels[2] + decoder_channels[3] + decoder_channels[1],
            decoder_channels[2],
            num_blocks[1],
        )
        self.conv3_1 = self._make_layer(
            block,
            decoder_channels[3] + decoder_channels[2],
            decoder_channels[3],
            num_blocks[2],
        )

        self.conv0_2 = self._make_layer(
            block, decoder_channels[0] * 2 + decoder_channels[1], decoder_channels[0]
        )
        self.conv1_2 = self._make_layer(
            block,
            decoder_channels[1] * 2 + decoder_channels[2] + decoder_channels[0],
            decoder_channels[1],
            num_blocks[0],
        )
        self.conv2_2 = self._make_layer(
            block,
            decoder_channels[2] * 2 + decoder_channels[3] + decoder_channels[1],
            decoder_channels[2],
            num_blocks[1],
        )

        self.conv0_3 = self._make_layer(
            block, decoder_channels[0] * 3 + decoder_channels[1], decoder_channels[0]
        )
        self.conv1_3 = self._make_layer(
            block,
            decoder_channels[1] * 3 + decoder_channels[2] + decoder_channels[0],
            decoder_channels[1],
            num_blocks[0],
        )

        self.conv0_4 = self._make_layer(
            block, decoder_channels[0] * 4 + decoder_channels[1], decoder_channels[0]
        )
        self.conv0_4_final = self._make_layer(
            block, decoder_channels[0] * 4, decoder_channels[0]
        )

        self.conv0_3_1x1 = nn.Conv2d(decoder_channels[3], decoder_channels[0], kernel_size=1)
        self.conv0_2_1x1 = nn.Conv2d(decoder_channels[2], decoder_channels[0], kernel_size=1)
        self.conv0_1_1x1 = nn.Conv2d(decoder_channels[1], decoder_channels[0], kernel_size=1)

    @staticmethod
    def _make_layer(block, in_channels, out_channels, num_blocks=1):
        layers = [block(in_channels, out_channels)]
        for _ in range(num_blocks - 1):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _resize_as(self, x, ref):
        return resize(
            x,
            size=ref.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
        )

    def _forward_feature(self, inputs):
        feats = self._transform_inputs(inputs)
        assert len(feats) == len(self.decoder_channels)
        dtype = feats[0].dtype
        feats = [feat.to(dtype) if feat.dtype != dtype else feat for feat in feats]

        x0_0 = self.stem_convs[0](feats[0])
        x1_0 = self.stem_convs[1](feats[1])
        x2_0 = self.stem_convs[2](feats[2])
        x3_0 = self.stem_convs[3](feats[3])

        x0_1 = self.conv0_1(torch.cat([x0_0, self._resize_as(x1_0, x0_0)], dim=1))

        x1_1 = self.conv1_1(
            torch.cat([
                x1_0,
                self._resize_as(x2_0, x1_0),
                self._resize_as(x0_1, x1_0),
            ], dim=1)
        )
        x0_2 = self.conv0_2(
            torch.cat([x0_0, x0_1, self._resize_as(x1_1, x0_0)], dim=1)
        )

        x2_1 = self.conv2_1(
            torch.cat([
                x2_0,
                self._resize_as(x3_0, x2_0),
                self._resize_as(x1_1, x2_0),
            ], dim=1)
        )
        x1_2 = self.conv1_2(
            torch.cat([
                x1_0,
                x1_1,
                self._resize_as(x2_1, x1_0),
                self._resize_as(x0_2, x1_0),
            ], dim=1)
        )
        x0_3 = self.conv0_3(
            torch.cat([
                x0_0,
                x0_1,
                x0_2,
                self._resize_as(x1_2, x0_0),
            ], dim=1)
        )

        x3_1 = self.conv3_1(
            torch.cat([
                x3_0,
                self._resize_as(x2_1, x3_0),
            ], dim=1)
        )
        x2_2 = self.conv2_2(
            torch.cat([
                x2_0,
                x2_1,
                self._resize_as(x3_1, x2_0),
                self._resize_as(x1_2, x2_0),
            ], dim=1)
        )
        x1_3 = self.conv1_3(
            torch.cat([
                x1_0,
                x1_1,
                x1_2,
                self._resize_as(x2_2, x1_0),
                self._resize_as(x0_3, x1_0),
            ], dim=1)
        )
        x0_4 = self.conv0_4(
            torch.cat([
                x0_0,
                x0_1,
                x0_2,
                x0_3,
                self._resize_as(x1_3, x0_0),
            ], dim=1)
        )

        final = self.conv0_4_final(
            torch.cat([
                self._resize_as(self.conv0_3_1x1(x3_1), x0_0),
                self._resize_as(self.conv0_2_1x1(x2_2), x0_0),
                self._resize_as(self.conv0_1_1x1(x1_3), x0_0),
                x0_4,
            ], dim=1)
        )

        return final

    def forward(self, inputs):
        feats = self._forward_feature(inputs)
        output = self.cls_seg(feats)
        return output