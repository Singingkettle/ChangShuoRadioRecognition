import warnings
from typing import List, Optional, Tuple
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.task_modules import AnchorGenerator, YOLOBBoxCoder
from mmdet.models.task_modules import GridAssigner, PseudoSampler
from mmdet.models.utils import images_to_levels, multi_apply
from mmdet.structures.bbox import BaseBoxes, HorizontalBoxes, get_box_tensor
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn.modules.utils import _pair

from .base_head import BaseHead
from ..builder import HEADS, build_loss


class SignalYOLOAnchorGenerator(AnchorGenerator):
    """Anchor generator for YOLO.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels.
        base_sizes (list[list[tuple[int, int]]]): The basic sizes
            of anchors in multiple levels.
    """

    def __init__(self,
                 strides: Union[List[int], List[Tuple[int, int]]],
                 base_sizes: List[List[Tuple[int, int]]],
                 use_box_type: bool = False) -> None:
        self.strides = [_pair(stride) for stride in strides]
        self.centers = [(72.0, 0.5)]
        self.base_sizes = []
        num_anchor_per_level = len(base_sizes[0])
        for base_sizes_per_level in base_sizes:
            assert num_anchor_per_level == len(base_sizes_per_level)
            self.base_sizes.append(
                [_pair(base_size) for base_size in base_sizes_per_level])
        self.base_anchors = self.gen_base_anchors()
        self.use_box_type = use_box_type

    @property
    def num_levels(self) -> int:
        """int: number of feature levels that the generator will be applied"""
        return len(self.base_sizes)

    def gen_base_anchors(self) -> List[Tensor]:
        """Generate base anchors.

        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_sizes_per_level in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(base_sizes_per_level,
                                                   center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_sizes_per_level: List[Tuple[int]],
                                      center: Optional[Tuple[float]] = None) \
            -> Tensor:
        """Generate base anchors of a single level.

        Args:
            base_sizes_per_level (list[tuple[int]]): Basic sizes of
                anchors.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """
        x_center, y_center = center
        base_anchors = []
        for base_size in base_sizes_per_level:
            w, h = base_size

            # use float anchor and the anchor's center is aligned with the
            # pixel center
            base_anchor = torch.Tensor([
                x_center - 0.5 * w, y_center - 0.5 * h, x_center + 0.5 * w,
                y_center + 0.5 * h
            ])
            base_anchors.append(base_anchor)
        base_anchors = torch.stack(base_anchors, dim=0)

        return base_anchors


class SignalYOLOBBoxCoder(YOLOBBoxCoder):

    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(eps=eps, **kwargs)

    def encode(self, bboxes: Union[Tensor, BaseBoxes],
               gt_bboxes: Union[Tensor, BaseBoxes],
               stride: Union[Tensor, int]) -> Tensor:
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor or :obj:`BaseBoxes`): Source boxes,
                e.g., anchors.
            gt_bboxes (torch.Tensor or :obj:`BaseBoxes`): Target of the
                transformation, e.g., ground-truth boxes.
            stride (torch.Tensor | int): Stride of bboxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """
        bboxes = get_box_tensor(bboxes)
        gt_bboxes = get_box_tensor(gt_bboxes)
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        x_center_gt = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) * 0.5
        # y_center_gt = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) * 0.5
        w_gt = gt_bboxes[..., 2] - gt_bboxes[..., 0]
        h_gt = gt_bboxes[..., 3] - gt_bboxes[..., 1]
        x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        # y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        w = bboxes[..., 2] - bboxes[..., 0]
        # h = bboxes[..., 3] - bboxes[..., 1]
        w_target = torch.log((w_gt / w).clamp(min=self.eps))
        # h_target = torch.log((h_gt / h).clamp(min=self.eps))
        x_center_target = ((x_center_gt - x_center) / stride + 0.5).clamp(
            self.eps, 1 - self.eps)
        # y_center_target = ((y_center_gt - y_center) / 1 + 0.5).clamp(
        #     self.eps, 1 - self.eps)
        # encoded_bboxes = torch.stack(
        #     [x_center_target, y_center_target, w_target, h_target], dim=-1)
        encoded_bboxes = torch.stack(
            [x_center_target, w_target], dim=-1)
        return encoded_bboxes

    def decode(self, bboxes: Union[Tensor, BaseBoxes], pred_bboxes: Tensor,
               stride: Union[Tensor, int]) -> Union[Tensor, BaseBoxes]:
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor or :obj:`BaseBoxes`): Basic boxes,
                e.g. anchors.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            stride (torch.Tensor | int): Strides of bboxes.

        Returns:
            Union[torch.Tensor, :obj:`BaseBoxes`]: Decoded boxes.
        """
        bboxes = get_box_tensor(bboxes)
        assert pred_bboxes.size(-1) == bboxes.size(-1) == 4
        xy_centers = (bboxes[..., :2] + bboxes[..., 2:]) * 0.5 + (
                pred_bboxes[..., :2] - 0.5) * stride
        whs = (bboxes[..., 2:] -
               bboxes[..., :2]) * 0.5 * pred_bboxes[..., 2:].exp()
        decoded_bboxes = torch.stack(
            (xy_centers[..., 0] - whs[..., 0], xy_centers[..., 1] -
             whs[..., 1], xy_centers[..., 0] + whs[..., 0],
             xy_centers[..., 1] + whs[..., 1]),
            dim=-1)

        if self.use_box_type:
            decoded_bboxes = HorizontalBoxes(decoded_bboxes)
        return decoded_bboxes


@HEADS.register_module()
class SignalDetectionHead(BaseHead):
    def __init__(self, num_anchors=2, in_size=256, featmap_strides=[16],
                 loss_bw=None, loss_cf=None, loss_conf=None, init_cfg=None):
        super(SignalDetectionHead, self).__init__(init_cfg)
        if loss_bw is None:
            loss_bw = dict(
                type='MSELoss',
                loss_weight=2.0,
                reduction='sum',
            )
        if loss_cf is None:
            loss_cf = dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0,
                reduction='sum'
            )
        if loss_conf is None:
            loss_conf = dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0,
                reduction='sum'
            )

        self.loss_bw = build_loss(loss_bw)
        self.loss_cf = build_loss(loss_cf)
        self.loss_conf = build_loss(loss_conf)

        self.featmap_strides = featmap_strides
        self.head = nn.Conv2d(in_size, num_anchors * 3, kernel_size=1)
        self.bbox_coder = SignalYOLOBBoxCoder()
        self.prior_generator = SignalYOLOAnchorGenerator(base_sizes=[[(121, 1), (93, 1)]], strides=[(16, 1)])
        self.assigner = GridAssigner(pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0)
        self.sampler = PseudoSampler()

    @property
    def num_levels(self) -> int:
        """int: number of feature map levels"""
        return len(self.featmap_strides)

    @property
    def num_attrib(self):
        """int: number of attributes in pred_map, cf (1) +
        bw (1) + is_signal (1)"""

        return 3

    def loss(self, pred_maps, data_samples, input_metas=None, weight=None, **kwargs):
        num_signals = len(input_metas)
        device = pred_maps[0][0].device

        featmap_sizes = [
            pred_maps[i].shape[-2:] for i in range(self.num_levels)
        ]

        mlvl_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=pred_maps[0].device)
        anchor_list = [mlvl_anchors for _ in range(num_signals)]

        batch_gt_instances = []
        for data_sample in data_samples:
            batch_gt_instances.append(data_sample.gt_instances)

        responsible_flag_list = []
        for id in range(num_signals):
            responsible_flag_list.append(
                self.responsible_flags(featmap_sizes, batch_gt_instances[id].bboxes,
                                       device))

        target_maps_list, neg_maps_list = self.get_targets(
            anchor_list, responsible_flag_list, batch_gt_instances)

        losses_conf, losses_x, losses_w = multi_apply(
            self.loss_by_feat_single, pred_maps, target_maps_list,
            neg_maps_list)

        return dict(
            loss_conf=losses_conf,
            loss_cf=losses_x,
            loss_bw=losses_w)

    def loss_by_feat_single(self, pred_map: Tensor, target_map: Tensor,
                            neg_map: Tensor) -> tuple:
        num_imgs = len(pred_map)
        pred_map = pred_map.permute(0, 2, 3,
                                    1).reshape(num_imgs, -1, self.num_attrib)
        neg_mask = neg_map.float()
        pos_mask = target_map[..., 2]
        pos_and_neg_mask = neg_mask + pos_mask
        if torch.max(pos_and_neg_mask) > 1.:
            warnings.warn('There is overlap between pos and neg sample.')
            pos_and_neg_mask = pos_and_neg_mask.clamp(min=0., max=1.)

        pred_cf = pred_map[..., 0]
        pred_bw = pred_map[..., 1]
        pred_conf = pred_map[..., 2]

        target_cf = target_map[..., 0]
        target_bw = target_map[..., 1]
        target_conf = target_map[..., 2]

        loss_conf = self.loss_conf(pred_conf, target_conf, weight=pos_and_neg_mask)
        loss_x = self.loss_cf(pred_cf, target_cf, weight=pos_mask)
        loss_w = self.loss_bw(pred_bw, target_bw, weight=pos_mask)

        return loss_conf, loss_x, loss_w

    def forward(self, x, vis_fea=False, is_test=False):
        pred_maps = self.head(x)
        pred_maps = (pred_maps, )
        if is_test:
            pass
            # self.predict_by_feat(pred_maps)
        else:
            return pred_maps

    # def predict_by_feat(self,
    #                     pred_maps: Sequence[Tensor]) -> InstanceList:
    #     """Transform a batch of output features extracted from the head into
    #     bbox results. It has been accelerated since PR #5991.
    #
    #     Args:
    #         pred_maps (Sequence[Tensor]): Raw predictions for a batch of
    #             images.
    #         batch_img_metas (list[dict], Optional): Batch image meta info.
    #             Defaults to None.
    #         cfg (:obj:`ConfigDict` or dict, optional): Test / postprocessing
    #             configuration, if None, test_cfg would be used.
    #             Defaults to None.
    #         rescale (bool): If True, return boxes in original image space.
    #             Defaults to False.
    #         with_nms (bool): If True, do nms before return boxes.
    #             Defaults to True.
    #
    #     Returns:
    #         list[:obj:`InstanceData`]: Object detection results of each image
    #         after the post process. Each item usually contains following keys.
    #
    #         - scores (Tensor): Classification scores, has a shape
    #           (num_instance, )
    #         - labels (Tensor): Labels of bboxes, has a shape
    #           (num_instances, ).
    #         - bboxes (Tensor): Has a shape (num_instances, 4),
    #           the last dimension 4 arrange as (x1, y1, x2, y2).
    #     """
    #     assert len(pred_maps) == self.num_levels
    #     cfg = self.test_cfg if cfg is None else cfg
    #     cfg = copy.deepcopy(cfg)
    #
    #     num_imgs = len(batch_img_metas)
    #     featmap_sizes = [pred_map.shape[-2:] for pred_map in pred_maps]
    #
    #     mlvl_anchors = self.prior_generator.grid_priors(
    #         featmap_sizes, device=pred_maps[0].device)
    #     flatten_preds = []
    #     flatten_strides = []
    #     for pred, stride in zip(pred_maps, self.featmap_strides):
    #         pred = pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
    #                                                 self.num_attrib)
    #         pred[..., :2].sigmoid_()
    #         flatten_preds.append(pred)
    #         flatten_strides.append(
    #             pred.new_tensor(stride).expand(pred.size(1)))
    #
    #     flatten_preds = torch.cat(flatten_preds, dim=1)
    #     flatten_bbox_preds = flatten_preds[..., :4]
    #     flatten_objectness = flatten_preds[..., 4].sigmoid()
    #     flatten_cls_scores = flatten_preds[..., 5:].sigmoid()
    #     flatten_anchors = torch.cat(mlvl_anchors)
    #     flatten_strides = torch.cat(flatten_strides)
    #     flatten_bboxes = self.bbox_coder.decode(flatten_anchors,
    #                                             flatten_bbox_preds,
    #                                             flatten_strides.unsqueeze(-1))
    #     results_list = []
    #     for (bboxes, scores, objectness,
    #          img_meta) in zip(flatten_bboxes, flatten_cls_scores,
    #                           flatten_objectness, batch_img_metas):
    #         # Filtering out all predictions with conf < conf_thr
    #         conf_thr = cfg.get('conf_thr', -1)
    #         if conf_thr > 0:
    #             conf_inds = objectness >= conf_thr
    #             bboxes = bboxes[conf_inds, :]
    #             scores = scores[conf_inds, :]
    #             objectness = objectness[conf_inds]
    #
    #         score_thr = cfg.get('score_thr', 0)
    #         nms_pre = cfg.get('nms_pre', -1)
    #         scores, labels, keep_idxs, _ = filter_scores_and_topk(
    #             scores, score_thr, nms_pre)
    #
    #         results = InstanceData(
    #             scores=scores,
    #             labels=labels,
    #             bboxes=bboxes[keep_idxs],
    #             score_factors=objectness[keep_idxs],
    #         )
    #         results = self._bbox_post_process(
    #             results=results,
    #             cfg=cfg,
    #             rescale=rescale,
    #             with_nms=with_nms,
    #             img_meta=img_meta)
    #         results_list.append(results)
    #     return results_list

    def get_targets(self, anchor_list: List[List[Tensor]],
                    responsible_flag_list: List[List[Tensor]],
                    batch_gt_instances: List[InstanceData]) -> tuple:
        """Compute target maps for anchors in multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_total_anchors, 4).
            responsible_flag_list (list[list[Tensor]]): Multi level responsible
                flags of each image. Each element is a tensor of shape
                (num_total_anchors, )
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.

        Returns:
            tuple: Usually returns a tuple containing learning targets.
                - target_map_list (list[Tensor]): Target map of each level.
                - neg_map_list (list[Tensor]): Negative map of each level.
        """
        num_imgs = len(anchor_list)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        results = multi_apply(self._get_targets_single, anchor_list,
                              responsible_flag_list, batch_gt_instances)

        all_target_maps, all_neg_maps = results
        assert num_imgs == len(all_target_maps) == len(all_neg_maps)
        target_maps_list = images_to_levels(all_target_maps, num_level_anchors)
        neg_maps_list = images_to_levels(all_neg_maps, num_level_anchors)

        return target_maps_list, neg_maps_list

    def _get_targets_single(self, anchors: List[Tensor],
                            responsible_flags: List[Tensor],
                            gt_instances: InstanceData) -> tuple:
        """Generate matching bounding box prior and converted GT.

        Args:
            anchors (List[Tensor]): Multi-level anchors of the image.
            responsible_flags (List[Tensor]): Multi-level responsible flags of
                anchors
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.

        Returns:
            tuple:
                target_map (Tensor): Predication target map of each
                    scale level, shape (num_total_anchors,
                    5+num_classes)
                neg_map (Tensor): Negative map of each scale level,
                    shape (num_total_anchors,)
        """
        gt_bboxes = gt_instances.bboxes
        anchor_strides = []
        for i in range(len(anchors)):
            anchor_strides.append(
                torch.tensor(self.featmap_strides[i],
                             device=gt_bboxes.device).repeat(len(anchors[i])))
        concat_anchors = torch.cat(anchors)
        concat_responsible_flags = torch.cat(responsible_flags)

        anchor_strides = torch.cat(anchor_strides)
        assert len(anchor_strides) == len(concat_anchors) == \
               len(concat_responsible_flags)
        pred_instances = InstanceData(
            priors=concat_anchors, responsible_flags=concat_responsible_flags)

        assign_result = self.assigner.assign(pred_instances, gt_instances)
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        target_map = concat_anchors.new_zeros(
            concat_anchors.size(0), self.num_attrib)

        target_map[sampling_result.pos_inds, :2] = self.bbox_coder.encode(
            sampling_result.pos_priors, sampling_result.pos_gt_bboxes,
            anchor_strides[sampling_result.pos_inds])

        target_map[sampling_result.pos_inds, 2] = 1

        neg_map = concat_anchors.new_zeros(
            concat_anchors.size(0), dtype=torch.uint8)
        neg_map[sampling_result.neg_inds] = 1

        return target_map, neg_map

    def responsible_flags(self, featmap_sizes: List[tuple], gt_bboxes: Tensor,
                          device: str) -> List[Tensor]:
        """Generate responsible anchor flags of grid cells in multiple scales.

        Args:
            featmap_sizes (List[tuple]): List of feature map sizes in multiple
                feature levels.
            gt_bboxes (Tensor): Ground truth boxes, shape (n, 4).
            device (str): Device where the anchors will be put on.

        Return:
            List[Tensor]: responsible flags of anchors in multiple level
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_responsible_flags = []

        for i in range(self.num_levels):
            anchor_stride = self.prior_generator.strides[i]
            feat_h, feat_w = featmap_sizes[i]

            gt_cx = ((gt_bboxes[:, 0] + gt_bboxes[:, 2]) * 0.5 - 72).to(device)
            gt_cx = torch.clamp(gt_cx, min=0, max=1056)
            gt_cy = ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) * 0.5 - 0.5).to(device)
            gt_grid_x = torch.floor(gt_cx / anchor_stride[0]).long()
            gt_grid_y = torch.floor(gt_cy / anchor_stride[1]).long()

            # row major indexing
            gt_bboxes_grid_idx = gt_grid_y * feat_w + gt_grid_x
            responsible_grid = torch.zeros(
                feat_h * feat_w, dtype=torch.uint8, device=device)
            responsible_grid[gt_bboxes_grid_idx] = 1

            responsible_grid = responsible_grid[:, None].expand(
                responsible_grid.size(0),
                self.prior_generator.num_base_priors[i]).contiguous().view(-1)
            multi_level_responsible_flags.append(responsible_grid)

        return multi_level_responsible_flags
