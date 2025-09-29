# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
import warnings

import mmcv
import numpy as np
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg import digit_version
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes

def save_prediction_images(results, dataset, pred_dir, pred_format='png'):
    """Save prediction masks, original images, GT masks, and overlay images.
    
    Args:
        results (list): List of prediction results
        dataset: Dataset object containing image information
        pred_dir (str): Directory to save prediction images
        pred_format (str): Image format ('png', 'jpg', 'jpeg')
    """
    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Ensure pred_dir exists
    mmcv.mkdir_or_exist(pred_dir)
    
    # Get palette for visualization
    if hasattr(dataset, 'PALETTE') and dataset.PALETTE is not None:
        palette = np.array(dataset.PALETTE)
    else:
        # Generate random palette if not available
        num_classes = len(dataset.CLASSES) if hasattr(dataset, 'CLASSES') else 256
        palette = np.random.randint(0, 255, size=(num_classes, 3))
    
    print(f'Saving {len(results)} prediction images...')
    
    for idx, result in enumerate(results):
        try:
            # Get image info from dataset
            if hasattr(dataset, 'img_infos'):
                img_info = dataset.img_infos[idx]
            elif hasattr(dataset, 'data_infos'):
                img_info = dataset.data_infos[idx]
            else:
                # For Subset datasets, get from the underlying dataset
                if hasattr(dataset, 'dataset'):
                    if hasattr(dataset.dataset, 'img_infos'):
                        img_info = dataset.dataset.img_infos[dataset.indices[idx]]
                    elif hasattr(dataset.dataset, 'data_infos'):
                        img_info = dataset.dataset.data_infos[dataset.indices[idx]]
                    else:
                        print(f'Warning: Cannot get image info for index {idx}')
                        continue
                else:
                    print(f'Warning: Cannot get image info for index {idx}')
                    continue
            
            # Get filename without extension for naming
            if 'filename' in img_info:
                img_filename = img_info['filename']
            elif 'img_path' in img_info:
                img_filename = osp.basename(img_info['img_path'])
            else:
                img_filename = f'{idx:05d}.png'
            
            base_name = osp.splitext(osp.basename(img_filename))[0]
            
            # Load original image
            if 'filename' in img_info:
                img_path = osp.join(dataset.img_dir, img_info['filename'])
            elif 'img_path' in img_info:
                img_path = img_info['img_path']
            else:
                print(f'Warning: Cannot find image path for index {idx}')
                continue
                
            if not osp.exists(img_path):
                print(f'Warning: Image file not found: {img_path}')
                continue
                
            original_img = mmcv.imread(img_path)
            
            # Get prediction result (assuming result is a list with segmentation mask)
            if isinstance(result, list) and len(result) > 0:
                pred_mask = result[0]  # First element is usually the segmentation mask
            else:
                pred_mask = result
            
            # Ensure pred_mask is numpy array
            if torch.is_tensor(pred_mask):
                pred_mask = pred_mask.cpu().numpy()
            
            # Squeeze extra dimensions if present
            if pred_mask.ndim > 2:
                pred_mask = pred_mask.squeeze()
            
            # Save prediction mask as grayscale
            # pred_path = osp.join(pred_dir, f'{base_name}_pred.{pred_format}')
            # cv2.imwrite(pred_path, pred_mask.astype(np.uint8))
            
            # # Save original image
            # orig_path = osp.join(pred_dir, f'{base_name}_original.{pred_format}')
            # cv2.imwrite(orig_path, original_img)
            
            # Try to load and save ground truth if available
            try:
                if 'ann' in img_info and 'seg_map' in img_info['ann']:
                    gt_path_rel = img_info['ann']['seg_map']
                    gt_path = osp.join(dataset.ann_dir, gt_path_rel)
                elif hasattr(dataset, 'get_ann_info'):
                    ann_info = dataset.get_ann_info(idx)
                    if 'seg_map' in ann_info:
                        gt_path = osp.join(dataset.ann_dir, ann_info['seg_map'])
                    else:
                        gt_path = None
                else:
                    gt_path = None
                
                if gt_path and osp.exists(gt_path):
                    gt_mask = mmcv.imread(gt_path, flag='grayscale')
                    # Squeeze extra dimensions if present
                    if gt_mask.ndim > 2:
                        gt_mask = gt_mask.squeeze()
                    gt_save_path = osp.join(pred_dir, f'{base_name}_gt.{pred_format}')
                    # cv2.imwrite(gt_save_path, gt_mask)
                else:
                    print(f'Warning: Ground truth not found for {base_name}')
                    gt_mask = None
            except Exception as e:
                print(f'Warning: Could not load ground truth for {base_name}: {e}')
                gt_mask = None
            
            # Create overlay image (prediction on original)
            try:
                overlay_img = original_img.copy()
                
                # Create colored segmentation mask
                color_seg = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
                for label in range(len(palette)):
                    if label < len(palette):
                        color_seg[pred_mask == label, :] = palette[label]
                
                # Convert to BGR for OpenCV
                color_seg = color_seg[..., ::-1]
                
                # Blend with original image
                opacity = 0.5
                overlay_img = overlay_img * (1 - opacity) + color_seg * opacity
                overlay_img = overlay_img.astype(np.uint8)
                
                # overlay_path = osp.join(pred_dir, f'{base_name}_overlay.{pred_format}')
                # cv2.imwrite(overlay_path, overlay_img)
            except Exception as e:
                print(f'Warning: Could not create overlay for {base_name}: {e}')
            
            # Create comparison image if GT is available
            if gt_mask is not None:
                try:
                    # Create a side-by-side comparison
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                    
                    # Original image
                    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
                    axes[0].set_title('Original')
                    axes[0].axis('off')
                    
                    # Ground truth with colored visualization
                    gt_colored = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
                    # Use green color for GT (RGB: 0, 255, 0)
                    gt_colored[gt_mask > 0] = [0, 255, 0]
                    axes[1].imshow(gt_colored)
                    axes[1].set_title('Ground Truth (Green)')
                    axes[1].axis('off')
                    
                    # Prediction with colored visualization
                    pred_colored = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
                    # Use red color for prediction (RGB: 255, 0, 0)
                    pred_colored[pred_mask > 0] = [255, 0, 0]
                    axes[2].imshow(pred_colored)
                    axes[2].set_title('Prediction (Red)')
                    axes[2].axis('off')
                    
                    # Overlay comparison (GT in green, Prediction in red)
                    overlay_comparison = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
                    overlay_comparison[gt_mask > 0] = [0, 255, 0]  # Green for GT
                    overlay_comparison[pred_mask > 0] = [255, 0, 0]  # Red for prediction
                    # Where both GT and prediction are present, use yellow (mix of green and red)
                    both_mask = (gt_mask > 0) & (pred_mask > 0)
                    overlay_comparison[both_mask] = [255, 255, 0]  # Yellow for overlap
                    axes[3].imshow(overlay_comparison)
                    axes[3].set_title('Overlay (Green=GT, Red=Pred, Yellow=Both)')
                    axes[3].axis('off')
                    
                    plt.tight_layout()
                    comp_path = osp.join(pred_dir, f'{base_name}_comparison.{pred_format}')
                    plt.savefig(comp_path, dpi=150, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f'Warning: Could not create comparison for {base_name}: {e}')
                    plt.close()  # Ensure figure is closed even on error
            
        except Exception as e:
            print(f'Error processing image {idx}: {e}')
            continue
        
        # Progress logging
        if (idx + 1) % 100 == 0:
            print(f'Saved {idx + 1}/{len(results)} prediction images...')
    
def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--save-pred',
        action='store_true',
        default=True,
        help='Save prediction masks as images')
    parser.add_argument(
        '--pred-dir',
        help='Directory to save prediction images')
    parser.add_argument(
        '--pred-format',
        type=str,
        default='png',
        choices=['png', 'jpg', 'jpeg'],
        help='Format for saving prediction images')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--max-test-samples',
        type=int,
        default=None,
        help='Maximum number of test samples to process (for debugging)')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir or args.save_pred, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show", "--show-dir" or "--save-pred"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(f'The gpu-ids is reset from {cfg.gpu_ids} to '
                          f'{cfg.gpu_ids[0:1]} to avoid potential error in '
                          'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(args.work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(args.work_dir,
                                 f'eval_single_scale_{timestamp}.json')
    elif rank == 0:
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(work_dir,
                                 f'eval_single_scale_{timestamp}.json')
    
    # Set up prediction directory
    if args.save_pred and rank == 0:
        if args.pred_dir is not None:
            pred_dir = osp.abspath(args.pred_dir)
        else:
            if args.work_dir is not None:
                pred_dir = osp.join(args.work_dir, 'predictions')
            else:
                pred_dir = osp.join(work_dir, 'predictions')
        mmcv.mkdir_or_exist(pred_dir)
        print(f'Prediction images will be saved to: {pred_dir}')

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    
    # Limit dataset size for debugging if specified
    if args.max_test_samples is not None and args.max_test_samples > 0:
        print(f'Limiting test dataset to {args.max_test_samples} samples for debugging')
        if hasattr(dataset, 'img_infos'):
            # For NUDTSIRSTDataset, use img_infos
            dataset.img_infos = dataset.img_infos[:args.max_test_samples]
        elif hasattr(dataset, 'data_infos'):
            # For other datasets, use data_infos
            dataset.data_infos = dataset.data_infos[:args.max_test_samples]
        else:
            # Fallback: create a subset by slicing the dataset
            original_len = len(dataset)
            subset_indices = range(min(args.max_test_samples, original_len))
            dataset = torch.utils.data.Subset(dataset, subset_indices)
            # Copy important attributes from the original dataset
            if hasattr(dataset.dataset, 'CLASSES'):
                dataset.CLASSES = dataset.dataset.CLASSES
            if hasattr(dataset.dataset, 'PALETTE'):
                dataset.PALETTE = dataset.dataset.PALETTE
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options

    # Deprecated
    efficient_test = eval_kwargs.get('efficient_test', False)
    if efficient_test:
        warnings.warn(
            '``efficient_test=True`` does not have effect in tools/test.py, '
            'the evaluation and format results are CPU memory efficient by '
            'default')

    eval_on_format_results = (
        args.eval is not None and 'cityscapes' in args.eval)
    if eval_on_format_results:
        assert len(args.eval) == 1, 'eval on format results is not ' \
                                    'applicable for metrics other than ' \
                                    'cityscapes'
    if args.format_only or eval_on_format_results:
        if 'imgfile_prefix' in eval_kwargs:
            tmpdir = eval_kwargs['imgfile_prefix']
        else:
            tmpdir = '.format_cityscapes'
            eval_kwargs.setdefault('imgfile_prefix', tmpdir)
        mmcv.mkdir_or_exist(tmpdir)
    else:
        tmpdir = None

    cfg.device = get_device()
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        if not torch.cuda.is_available():
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        model = revert_sync_batchnorm(model)
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        results = single_gpu_test(
            model,
            data_loader,
            args.show,
            args.show_dir,
            False,
            args.opacity,
            pre_eval=args.eval is not None and not eval_on_format_results,
            format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        results = multi_gpu_test(
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
            False,
            pre_eval=args.eval is not None and not eval_on_format_results,
            format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs)

    rank, _ = get_dist_info()
    if rank == 0:
        # if args.out:
        #     warnings.warn(
        #         'The behavior of ``args.out`` has been changed since MMSeg '
        #         'v0.16, the pickled outputs could be seg map as type of '
        #         'np.array, pre-eval results or file paths for '
        #         '``dataset.format_results()``.')
        #     print(f'\nwriting results to {args.out}')
        #     mmcv.dump(results, args.out)
        if args.eval:
            eval_kwargs.update(metric=args.eval)
            metric = dataset.evaluate(results, **eval_kwargs)
            metric_dict = dict(config=args.config, metric=metric)
            mmcv.dump(metric_dict, json_file, indent=4)
            if tmpdir is not None and eval_on_format_results:
                # remove tmp dir when cityscapes evaluation
                shutil.rmtree(tmpdir)
        
        # Save prediction images if requested
        if args.save_pred:
            save_prediction_images(results, dataset, pred_dir, args.pred_format)


if __name__ == '__main__':
    main()
