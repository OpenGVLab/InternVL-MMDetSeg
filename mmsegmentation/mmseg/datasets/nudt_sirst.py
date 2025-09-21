import os.path as osp
import random
from functools import reduce
import os
import numpy as np
from terminaltables import AsciiTable
from torch.utils.data import Dataset
import torch
import scipy.io as sio
import mmcv
from mmcv.utils import print_log
from mmseg.core import eval_metrics, eval_pixel_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose
from ..core.evaluation import ROCMetric, PD_FA, mIoU


@DATASETS.register_module()
class NUDTSIRSTDataset(Dataset):
    """
    Paper name: "Dense Nested Attention Network for Infrared Small Target Detection"
    Paper link: https://ieeexplore.ieee.org/document/9864119
    Code link: https://github.com/YeRen123455/Infrared-Small-Target-Detection

    """

    CLASSES = (
        # "background", # 0
        "target",  # 255
    )
    PALETTE = [
        # [0, 0, 0],
        [255, 255, 255],
    ]

    def __init__(
        self,
        pipeline,
        data_root="data/nudt_sirst",
        img_dir="images",
        img_suffix=".png",
        ann_dir="masks",
        seg_map_suffix=".png",
        split_dir="split_dataset",
        split_suffix=".txt",
        test_mode=False,  # train or test/val
        ignore_index=255,
        reduce_zero_label=False,
        label_map={255: 1},  # 将 target 的像素值 255 映射为 1
        classes=None,
        palette=None,
    ):
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split_dir = split_dir
        self.split_suffix = split_suffix
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = label_map
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(classes, palette)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split_dir is None or osp.isabs(self.split_dir)):
                self.split_dir = osp.join(self.data_root, self.split_dir)

        # load annotations for training or test
        self.img_infos = self.load_annotations(
            self.img_dir,
            self.img_suffix,
            self.ann_dir,
            self.seg_map_suffix,
            self.split_dir,
            self.split_suffix,
        )
        self.pipeline = Compose(pipeline)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(
        self, img_dir, img_suffix, ann_dir, seg_map_suffix, split_dir, split_suffix
    ):
        """Load annotation from directory. Note that we do not use split for slovenia dataset!

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.

        Returns:
            list[dict]: All image info of dataset.
        """
        img_names = self.load_img_names()
        # add suffix for img_names, e.g., 000001 -> 000001.png
        img_names_with_suffix = [f'{idx + self.img_suffix}' for idx in img_names]
        
        # 严格按照 train.txt 的顺序构建 img_infos
        img_infos = []
        # 先扫描所有文件，建立文件存在性检查的集合
        existing_files = set(mmcv.scandir(img_dir, img_suffix, recursive=True))
        
        # 严格按照 train.txt 的顺序添加图片信息
        for img_name in img_names_with_suffix:
            if img_name in existing_files:
                img_info = dict(filename=img_name)
                if ann_dir is not None:
                    seg_map = img_name.replace(img_suffix, seg_map_suffix)
                    img_info["ann"] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            else:
                print_log(f"Warning: Image {img_name} listed in split file but not found in {img_dir}", 
                         logger=get_root_logger())

        print_log(f"Loaded {len(img_infos)} images in strict order from split file", logger=get_root_logger())
        return img_infos

    def load_img_names(self):
        """Load image names for the training or testing set.

        Returns:
            _type_: list[str]: image names for the training or testing set.
        """
        split_filename = 'test' if self.test_mode else 'train'
        split_txt = osp.join(self.split_dir, split_filename + self.split_suffix)
        img_names = []
        with open(split_txt, "r") as f:
            line = f.readline()
            while line:
                img_names.append(line.split('\n')[0])
                line = f.readline()
            f.close()

        return img_names

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]["ann"]

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results["seg_fields"] = []
        results["img_prefix"] = self.img_dir
        results["seg_prefix"] = self.ann_dir
        # if self.custom_classes:
        if self.label_map:
            results["label_map"] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """
        img_info = self.img_infos[idx]
        # results = dict(img_info=img_info)
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.
        
        Args:
            preds (list[torch.Tensor]): Prediction results.
            indices (list[int]): Indices of predictions.
            
        Returns:
            list[torch.Tensor]: Collected predictions.
        """
        # Convert predictions to numpy arrays if they are tensors
        collected_preds = []
        for pred, idx in zip(preds, indices):
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            collected_preds.append(pred)
        return collected_preds

    def get_gt_seg_maps(self, idx=None):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        if idx is None:
            for img_info in self.img_infos:
                seg_map = osp.join(self.ann_dir, img_info["ann"]["seg_map"])
                # gt_seg_map.shape (H, W) or (H, W, C)
                gt_seg_map = mmcv.imread(seg_map, flag='unchanged', backend='pillow')
                # modify if custom classes
                if self.label_map is not None:
                    for old_id, new_id in self.label_map.items():
                        gt_seg_map[gt_seg_map == old_id] = new_id

                if self.reduce_zero_label:
                    # avoid using underflow conversion
                    gt_seg_map[gt_seg_map == 0] = 255
                    gt_seg_map = gt_seg_map - 1
                    gt_seg_map[gt_seg_map == 254] = 255

                # print_log(
                #     f"gt_seg_map.min() = {gt_seg_map.min()}, gt_seg_map.max() = {gt_seg_map.max()}",
                #     logger=get_root_logger(),
                # )
                gt_seg_maps.append(gt_seg_map)

            return gt_seg_maps
        else:
            img_info = self.img_infos[idx]
            seg_map = osp.join(self.ann_dir, img_info["ann"]["seg_map"])
            print_log(
                f"\nseg_map = {seg_map}",
                logger=get_root_logger(),
            )
            # gt_seg_map.shape (H, W) or (H, W, C)
            gt_seg_map = mmcv.imread(seg_map, flag='unchanged', backend='pillow')
            # modify if custom classes
            if self.label_map is not None:
                for old_id, new_id in self.label_map.items():
                    gt_seg_map[gt_seg_map == old_id] = new_id

            if self.reduce_zero_label:
                # avoid using underflow conversion
                gt_seg_map[gt_seg_map == 0] = 255
                gt_seg_map = gt_seg_map - 1
                gt_seg_map[gt_seg_map == 254] = 255

            return [gt_seg_map]

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f"Unsupported type {type(classes)} of classes.")

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError("classes is not a subset of CLASSES.")

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            # !!!!!!!!!!!!!!!!!!!!!!!HERE!!!!!!!!!!!!!!!!!!!
            # !WE DO NOT USE self.label_map dict.!
            # !BECAUSE OUR SEG MAP IS RIGHT! DO NOT NEED MAP!!!!
            # self.label_map = {}
            # for i, c in enumerate(self.CLASSES):
            #     if c not in class_names:
            #         self.label_map[i] = -1
            #     else:
            #         self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def real_time_evaluate(
        self, results, idx=-1, metric=["mIoU"], ROC_thr=10, logger=None, **kwargs
    ):
        """Evaluate the dataset.

        Args:
            result (list): One Testing result of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        allowed_metrics = [
            'PdFa',
            'ROC',
            'mIoU',  # 有两种实现，此处使用 pixel_metrics 中的 mIoU
            # 'mDice',
            # 'mF1score',
        ]
        if isinstance(metric, str):
            metric = [metric]
            if metric[0] not in allowed_metrics:
                raise KeyError(f'Metric {metric} is not supported')
        elif isinstance(metric, (list, tuple)):
            unsupported = [m for m in metric if m not in allowed_metrics]
            if unsupported:
                raise KeyError(f'Metrics {unsupported} are not supported')

        self.metric = metric
        self.ROC_thr = ROC_thr
        gt_seg_maps = self.get_gt_seg_maps(idx)

        # if self.CLASSES is None:
        #     num_classes = len(reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        # else:
        #     num_classes = len(self.CLASSES)

        eval_result = {}
        # ['PdFa', 'ROC', 'mIoU'] 可以一起打包计算，节省计算时间
        # 如果 self.metric_objs 不存在，则初始化它
        if not hasattr(self, "metric_objs") or not self.metric_objs:
            self.metric_objs = {}
            if 'PdFa' in metric:
                self.metric_objs['PdFa'] = PD_FA(num_classes=1, bins=ROC_thr)
            if 'ROC' in metric:
                self.metric_objs['ROC'] = ROCMetric(num_classes=1, bins=ROC_thr)
            if 'mIoU' in metric:
                self.metric_objs['mIoU'] = mIoU(num_classes=1, threshod=None)

        # 逐样本更新 —— 只需一次 zip(results, gt_seg_maps) 循环
        # print(f"\n metric_objs.keys()={metric_objs.keys()}")
        # 单个样本计算指标
        for pred, label in zip(results, gt_seg_maps):
            # pred.shape = [1 H W] label.shape = [H, W]
            pred = pred.reshape(pred.shape[-2], pred.shape[-1])
            assert pred.shape == label.shape, "Predict and Label Shape Don't Match"
            for key, metric_obj in self.metric_objs.items():  # 统一 update
                # metric.update() needs pred.shape = label.shape = [1, 1, H, W]
                metric_obj.update(pred, label)

        print_log(f"eval idx={idx}/{len(self)-1}", logger=get_root_logger())
        if idx < len(self) - 1:
            # 获取当前结果
            # Calculate shapes for variable sized images
            shapes = [gt_seg_map.shape[:2] for gt_seg_map in gt_seg_maps]
            
            # Debug: Print shape information
            print_log(f"DEBUG - idx={idx}, img_num={idx + 1}", logger=get_root_logger())
            print_log(f"DEBUG - gt_seg_maps length: {len(gt_seg_maps)}", logger=get_root_logger())
            print_log(f"DEBUG - Individual shapes: {[gt_seg_map.shape for gt_seg_map in gt_seg_maps]}", logger=get_root_logger())
            print_log(f"DEBUG - Extracted shapes (H,W): {shapes}", logger=get_root_logger())
            
            # Check if all shapes are the same
            if len(set(shapes)) == 1:
                print_log(f"DEBUG - All images have same shape: {shapes[0]}", logger=get_root_logger())
            else:
                print_log(f"DEBUG - Images have different shapes: {len(set(shapes))} unique shapes", logger=get_root_logger())
                print_log(f"DEBUG - Shape distribution: {dict((shape, shapes.count(shape)) for shape in set(shapes))}", logger=get_root_logger())
            
            # Use accumulated pixel counts instead of fixed shapes
            Fa, Pd = self.metric_objs['PdFa'].get()
            # 长度为 ROC_thr+1 的 list
            ture_positive_rate, false_positive_rate, recall, precision = (
                self.metric_objs['ROC'].get()
            )
            # 两个值
            overall_acc, mean_iou = self.metric_objs['mIoU'].get()
            # 汇总到 cur_eval_results
            cur_eval_result = {}
            cur_eval_result.update(
                {
                    'FA': Fa,
                    'PD': Pd,
                    'TPR': ture_positive_rate,
                    'FPR': false_positive_rate,
                    'Recall': recall,
                    'Precision': precision,
                    'mIoU': mean_iou,
                    'OA': overall_acc,
                }
            )  # 构建表格数据

            return cur_eval_result
        else:
            # 统一取结果
            # 长度为 ROC_thr+1 的 list
            # Calculate shapes for variable sized images
            shapes = [gt_seg_map.shape[:2] for gt_seg_map in gt_seg_maps]
            
            # Debug: Print final evaluation shape information
            print_log(f"DEBUG - Final evaluation, total images: {len(self)}", logger=get_root_logger())
            print_log(f"DEBUG - gt_seg_maps length: {len(gt_seg_maps)}", logger=get_root_logger())
            print_log(f"DEBUG - Individual shapes: {[gt_seg_map.shape for gt_seg_map in gt_seg_maps]}", logger=get_root_logger())
            print_log(f"DEBUG - Extracted shapes (H,W): {shapes}", logger=get_root_logger())
            
            # Check if all shapes are the same
            if len(set(shapes)) == 1:
                print_log(f"DEBUG - All images have same shape: {shapes[0]}", logger=get_root_logger())
            else:
                print_log(f"DEBUG - Images have different shapes: {len(set(shapes))} unique shapes", logger=get_root_logger())
                print_log(f"DEBUG - Shape distribution: {dict((shape, shapes.count(shape)) for shape in set(shapes))}", logger=get_root_logger())
            
            # Use accumulated pixel counts instead of fixed shapes
            Fa, Pd = self.metric_objs['PdFa'].get()
            # 长度为 ROC_thr+1 的 list
            ture_positive_rate, false_positive_rate, recall, precision = (
                self.metric_objs['ROC'].get()
            )
            # 两个值
            overall_acc, mean_iou = self.metric_objs['mIoU'].get()
            # 汇总到 eval_results
            eval_result.update(
                {
                    'FA': Fa,
                    'PD': Pd,
                    'TPR': ture_positive_rate,
                    'FPR': false_positive_rate,
                    'Recall': recall,
                    'Precision': precision,
                    'mIoU': mean_iou,
                    'OA': overall_acc,
                }
            )  # 构建表格数据

            # 构建 AsciiTable 表格并打印
            self.show_eval_results(eval_result, logger)

            return eval_result

    def evaluate(self, results, metric=["mIoU"], ROC_thr=10, logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        allowed_metrics = [
            'PdFa',
            'ROC',
            'mIoU',  # 有两种实现，此处使用 pixel_metrics 中的 mIoU
            # 'mDice',
            # 'mF1score',
        ]
        if isinstance(metric, str):
            metric = [metric]
            if metric[0] not in allowed_metrics:
                raise KeyError(f'Metric {metric} is not supported')
        elif isinstance(metric, (list, tuple)):
            unsupported = [m for m in metric if m not in allowed_metrics]
            if unsupported:
                raise KeyError(f'Metrics {unsupported} are not supported')

        self.metric = metric
        self.ROC_thr = ROC_thr
        gt_seg_maps = self.get_gt_seg_maps()
        assert len(results) == len(
            gt_seg_maps
        ), f"Results and gt_seg_maps must be the same length, got {len(results)} and {len(gt_seg_maps)}"

        if self.CLASSES is None:
            num_classes = len(reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)

        eval_results = {}
        eval_results = eval_pixel_metrics(
            results,
            gt_seg_maps,
            num_classes,
            ignore_index=self.ignore_index,
            metrics=metric,
            ROC_thr=ROC_thr,
        )
        # 汇总到 eval_results
        # eval_results.update(
        #     {
        #         'FA': Fa,
        #         'PD': Pd,
        #         'TPR': ture_positive_rate,
        #         'FPR': false_positive_rate,
        #         'Recall': recall,
        #         'Precision': precision,
        #         'mIoU': mean_iou,
        #         'OA': overall_acc,
        #     }
        # )  # 构建表格数据

        # 构建 AsciiTable 表格并打印
        # self.show_eval_results(eval_results, logger)

        return eval_results

    def show_eval_results(
        self,
        eval_results,
        logger=None,
    ):
        """
        展示评估结果表格，并返回 eval_results

        参数:
            FA, PD, ture_positive_rate, false_positive_rate, recall, precision: list, 长度一致
            overall_acc, mean_iou: float
            eval_results: dict 或其他
            logger: 日志对象，可为 None
            thr_list: 阈值列表，可为 None，若为 None 则用索引
        """
        # ------------- 1) 构建表头 -------------
        header = ['Threshold']
        metric = self.metric
        if 'PdFa' in metric:
            header += ['Pd (%)', 'Fa']
        if 'ROC' in metric:
            header += ['TPR (%)', 'FPR (%)', 'Recall (%)', "Precision (%)"]
        if 'mIoU' in metric:
            header += ['mIoU (%)', 'OA (%)']
        # if 'mAP' in metric:
        #     header += ['mAP']
        print_log(f"header={header}", logger=get_root_logger())
        table_data = [header]

        # ------------- 2) 每个阈值一行 -------------
        # thr_list = np.linspace(0, 1, self.ROC_thr + 1).tolist()
        thr_list = np.round(np.linspace(0, 1, self.ROC_thr + 1), 2).tolist()
        n_thr = len(thr_list)

        FA = eval_results['FA']
        PD = eval_results['PD']
        TPR = eval_results['TPR']
        FPR = eval_results['FPR']
        Recall = eval_results['Recall']
        Precision = eval_results['Precision']
        OA = eval_results['OA']
        for i in range(n_thr):
            thr_val = thr_list[i]
            cur_row = [f'{thr_val}']
            if 'PdFa' in metric:
                cur_row += [f'{PD[i] * 100:.4f}', f'{FA[i]:.6e}']
            if 'ROC' in metric:
                cur_row += [
                    f'{TPR[i] * 100:.4f}',
                    f'{FPR[i] * 100:.4f}',
                    f'{Recall[i] * 100:.4f}',
                    f'{Precision[i] * 100:.4f}',
                ]
            if 'mIoU' in metric:
                cur_row += ['', '']
            # if 'mAP' in metric:
            #     cur_row += ['']
            table_data.append(cur_row)

        # ------------- 3) 末行均值 & 标量 -------------
        mean_FA = sum(FA) / n_thr
        mean_PD = sum(PD) / n_thr
        mean_TPR = sum(TPR) / n_thr
        mean_FPR = sum(FPR) / n_thr
        mean_Recall = sum(Recall) / n_thr
        mean_Precision = sum(Precision) / n_thr

        last_row = [
            'Mean',
        ]
        if 'PdFa' in metric:
            last_row += [f'{mean_PD * 100:.4f}', f'{mean_FA:.6e}']
        if 'ROC' in metric:
            last_row += [
                f'{mean_TPR * 100:.4f}',
                f'{mean_FPR * 100:.4f}',
                f'{mean_Recall * 100:.4f}',
                f'{mean_Precision * 100:.4f}',
            ]
        if 'mIoU' in metric:
            last_row += [f"{eval_results['mIoU'] * 100:.4f}", f'{OA * 100:.4f}']
        # if 'mAP' in metric:
        #     last_row += [f"{eval_results['mAP'] * 100:.4f}"]
        table_data.append(last_row)

        # ------------- 4) 打印 -------------
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True  # 末行横线
        print_log('\n' + table.table, logger=logger)

        return eval_results
