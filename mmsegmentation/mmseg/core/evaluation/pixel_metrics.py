import numpy as np
from skimage import measure

# from .metrics import *
from scipy.special import expit  # 稳定版 sigmoid


def cal_tp_pos_fp_neg(output, target, num_classes, thr, already_sigmoid=False):
    assert output.shape == target.shape, "Predict and Label Shape Don't Match"

    if not already_sigmoid:
        from scipy.special import expit

        predict = (expit(output) > thr).astype(np.float32)
    else:
        predict = (output > thr).astype(np.float32)

    intersection = predict * (predict == target)
    tp = intersection.sum()
    fp = (predict * (predict != target)).sum()
    tn = ((1 - predict) * (predict == target)).sum()
    fn = ((predict != target) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos = tp + fp

    return tp, pos, fp, neg, class_pos


def batch_pix_accuracy(output, target):

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).astype(np.float32)
    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target) * (target > 0)).sum()

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, num_classes):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).astype(np.float32)
    intersection = predict * (predict == target)

    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter

    assert (
        area_inter <= area_union
    ).all(), "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union


class ROCMetric:
    """Computes pixAcc and mIoU metric scores"""

    def __init__(
        self, num_classes, bins, thr_mode='logits'
    ):  # bin 的意义实际上是确定 ROC 曲线上的 threshold 取多少个离散值
        super(ROCMetric, self).__init__()
        self.num_classes = num_classes
        self.bins = bins
        assert thr_mode in ['logits', 'sigmoid']
        if thr_mode == 'logits':
            self.already_sigmoid = False
        elif thr_mode == "sigmoid":
            self.already_sigmoid = True
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins + 1):
            thr = (iBin + 0.0) / self.bins
            # print(iBin, "-th, thr:", thr)
            i_tp, i_pos, i_fp, i_neg, i_class_pos = cal_tp_pos_fp_neg(
                preds, labels, self.num_classes, thr, self.already_sigmoid
            )
            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg
            self.class_pos[iBin] += i_class_pos

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)

        recall = self.tp_arr / (self.pos_arr + 0.001)
        precision = self.tp_arr / (self.class_pos + 0.001)

        return tp_rates, fp_rates, recall, precision

    def reset(self):

        self.tp_arr = np.zeros([11])
        self.pos_arr = np.zeros([11])
        self.fp_arr = np.zeros([11])
        self.neg_arr = np.zeros([11])
        self.class_pos = np.zeros([11])


class PD_FA:
    def __init__(self, num_classes, bins=10, thr_mode='logits'):
        """
        参数说明：
        num_classes (int): 类别数，当前仅用于占位，保留多类扩展可能。
        bins (int): 阈值划分的数量，将 [0,1] 区间均分用于计算 PD/FA 曲线。
        thr_mode (str): 输入预测值的取值范围，可选：
            - 'logits'：原始 logits，对应 thr_range 取值 255
            - 'sigmoid'：经过 sigmoid 后的输出，对应 thr_range 取值 1
        threshold (float): 默认阈值，仅在特定模式下用于单一阈值评估。
        """
        super(PD_FA, self).__init__()
        assert thr_mode in ['logits', 'sigmoid']
        self.num_classes = num_classes
        self.thr_range = thr_mode
        if thr_mode == 'logits':
            self.thr_range = 255
        elif thr_mode == "sigmoid":
            self.thr_range = 1.0
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins + 1)
        self.PD = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)

    def update(self, preds, labels):
        for iBin in range(self.bins + 1):
            # todo ========
            thr = iBin * (self.thr_range / self.bins)
            # todo ========
            predits = np.array((preds > thr)).astype('int64')
            predits = np.reshape(predits, predits.shape[-2:])
            labelss = np.array((labels)).astype('int64')  # P
            labelss = np.reshape(labelss, predits.shape[-2:])

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss, connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin] += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match = []
            self.dismatch = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]
                        break
            # todo 这里不严谨！！！会导致虚警率变低！！！
            self.dismatch = [
                x for x in self.image_area_total if x not in self.image_area_match
            ]
            self.FA[iBin] += np.sum(self.dismatch)
            self.PD[iBin] += len(self.distance_match)

    def get(self, img_num, shape=(256, 256)):
        H, W = shape
        Final_FA = self.FA / ((H * W) * img_num)
        Final_PD = self.PD / self.target

        return Final_FA, Final_PD

    def reset(self):
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])


class mIoU:
    def __init__(self, num_classes, threshod=None):
        super(mIoU, self).__init__()
        self.num_classes = num_classes
        self.threshod = threshod
        self.reset()

    def update(self, preds, labels):
        if self.threshod is not None:
            preds = preds > self.threshod
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.num_classes)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


def eval_pixel_metrics(
    results,
    gt_seg_maps,
    num_classes,
    ignore_index,
    metrics=['mIoU'],
    ROC_thr=10,
    nan_to_num=0.0,
):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
    Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes,)
        ndarray: Per category evalution metrics, shape (num_classes,)
    """

    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = [
        'PdFa',
        'ROC',
        'mIoU',  # 有两种实现，此处使用 pixel_metrics 中的 mIoU
        # 'mDice',
        # 'mF1score',
    ]  # ['mIoU', 'mDice', 'mF1score']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    # ['PdFa', 'ROC', 'mIoU'] 可以一起打包计算，节省计算时间
    metric_objs = {}
    if 'PdFa' in metrics:
        metric_objs['PdFa'] = PD_FA(num_classes=1, bins=ROC_thr)
    if 'ROC' in metrics:
        metric_objs['ROC'] = ROCMetric(num_classes=1, bins=ROC_thr)
    if 'mIoU' in metrics:
        metric_objs['mIoU'] = mIoU(num_classes=1)
    # 逐样本更新 —— 只需一次 zip(results, gt_seg_maps) 循环
    # print(f"\n metric_objs.keys()={metric_objs.keys()}")
    for pred, label in zip(results, gt_seg_maps):
        # pred.shape = [1 H W] label.shape = [H, W]
        # print(f"Debug - pred.shape: {pred.shape}, label.shape: {label.shape}")
        pred = pred.reshape(pred.shape[-2], pred.shape[-1])
        
        # Fix label shape if it has extra dimensions
        if len(label.shape) == 3:
            if label.shape[-1] in [3, 4]:
                # If label is multi-channel (H, W, C), take first channel
                label = label[:, :, 0]
            elif label.shape[0] == 1:
                # If label is (1, H, W), squeeze first dimension
                label = label.squeeze(0)
        
        # print(f"Debug - after reshape - pred.shape: {pred.shape}, label.shape: {label.shape}")
        assert pred.shape == label.shape, f"Predict and Label Shape Don't Match: pred={pred.shape}, label={label.shape}"
        for key, metric_obj in metric_objs.items():  # 统一 update
            # metric.update() needs pred.shape = label.shape = [1, 1, H, W]
            metric_obj.update(pred, label)
            # if key == 'PdFa':
            #     metric.update(pred, label)  # maybe binary mask only
            #     pass
            # elif key == 'ROC':
            #     # metric.update(pred, label)
            #     ture_positive_rate, false_positive_rate, recall, precision = metric_objs['ROC'].get()
            # elif key == 'mIoU':
            #     # metric.update(pred, label)
            #     _, mean_iou = metric_objs['mIoU'].get()

    # 统一取结果
    ret_metrics = {}
    
    # 长度为 ROC_thr+1 的 list
    if 'PdFa' in metric_objs:
        Fa, Pd = metric_objs['PdFa'].get(len(gt_seg_maps), gt_seg_maps[0].shape)
        ret_metrics.update({'FA': Fa, 'PD': Pd})
    
    # 长度为 ROC_thr+1 的 list
    if 'ROC' in metric_objs:
        ture_positive_rate, false_positive_rate, recall, precision = metric_objs['ROC'].get()
        ret_metrics.update({
            'TPR': ture_positive_rate,
            'FPR': false_positive_rate,
            'Recall': recall,
            'Precision': precision,
        })
    
    # 两个值
    if 'mIoU' in metric_objs:
        overall_acc, mean_iou = metric_objs['mIoU'].get()
        ret_metrics.update({'mIoU': mean_iou, 'OA': overall_acc})

    return ret_metrics


class PD_FA_1bin:
    def __init__(
        self,
    ):
        super(PD_FA_1bin, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0

    def update(self, preds, labels, size):
        predits = preds.astype('int64')
        labelss = labels.astype('int64')

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match = []
        self.dismatch = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)

                    del coord_image[m]
                    break

        self.dismatch = [
            x for x in self.image_area_total if x not in self.image_area_match
        ]
        self.dismatch_pixel += np.sum(self.dismatch)
        self.all_pixel += size[0] * size[1]
        self.PD += len(self.distance_match)

    def get(self):
        Final_FA = self.dismatch_pixel / self.all_pixel
        Final_PD = self.PD / self.target
        return Final_PD, float(Final_FA)
