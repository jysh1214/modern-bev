# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time


@DETECTORS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(BEVFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        # TODO: Allow users to specify the device via command line args
        self.device = 'cuda' # cpu

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }


    def extract_img_feat(self, img, len_queue=None):
        """Extract features of images."""

        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, len_queue=len_queue)
        
        return img_feats

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        rescale = kwargs['rescale']

        if self.device == 'cuda':
            img = kwargs['img'][0]
            img_metas = kwargs['img_metas'][0][0]
        else:
            img = kwargs['img'][0].data[0]
            img_metas = kwargs['img_metas'][0].data[0][0]

        ori_shape = img_metas['ori_shape']
        ori_shape = torch.tensor(ori_shape)

        img_shape = img_metas['img_shape']
        img_shape = torch.tensor(img_shape)

        lidar2img = img_metas['lidar2img']
        lidar2img = torch.tensor(lidar2img)

        lidar2cam = img_metas['lidar2cam']
        lidar2cam = torch.tensor(lidar2cam)

        pad_shape = img_metas['pad_shape']
        pad_shape = torch.tensor(pad_shape)

        scale_factor = img_metas['scale_factor']
        flip = img_metas['flip']
        pcd_horizontal_flip = img_metas['pcd_horizontal_flip']
        pcd_vertical_flip = img_metas['pcd_vertical_flip']

        box_mode_3d = img_metas['box_mode_3d']
        box_type_3d = img_metas['box_type_3d']
        img_norm_cfg = img_metas['img_norm_cfg']
        sample_idx = img_metas['sample_idx']
        prev_idx = img_metas['prev_idx']
        next_idx = img_metas['next_idx']      
        pts_filename = img_metas['pts_filename']
        scene_token = img_metas['scene_token']

        can_bus = img_metas['can_bus']
        can_bus = torch.tensor(can_bus)

        return self.forward_test(
            img=img,
            rescale=rescale,
            ori_shape=ori_shape,
            img_shape=img_shape,
            lidar2img=lidar2img,
            lidar2cam=lidar2cam,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip,
            pcd_horizontal_flip=pcd_horizontal_flip,
            pcd_vertical_flip=pcd_vertical_flip,
            can_bus=can_bus,
        )
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    def forward_test(self,
        img,
        rescale,
        ori_shape,
        img_shape,
        lidar2img,
        lidar2cam,
        pad_shape,
        scale_factor,
        flip,
        pcd_horizontal_flip,
        pcd_vertical_flip,
        can_bus,
    ):
        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(can_bus[:3])
        tmp_angle = copy.deepcopy(can_bus[-1])
        if self.prev_frame_info['prev_bev'] is not None:
            can_bus[:3] -= self.prev_frame_info['prev_pos']
            can_bus[-1] -= self.prev_frame_info['prev_angle']
        else:
            can_bus[-1] = 0
            can_bus[:3] = 0

        new_prev_bev, bbox_results =  self.simple_test(
            img=img,
            rescale=rescale,
            can_bus=can_bus,
            lidar2img=lidar2img,
            img_shape=img_shape,
            prev_bev=None,
        )
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        # boxes_3d, scores_3d, labels_3d = bbox_results
        return bbox_results

    def simple_test(
        self,
        img=None,
        rescale=False,
        can_bus=None,
        lidar2img=None,
        img_shape=None,
        prev_bev=None,
    ):
        img_feats = self.extract_feat(img=img,)
        outs = self.pts_bbox_head(
            img_feats,
            can_bus=can_bus,
            lidar2img=lidar2img,
            img_shape=img_shape,
            prev_bev=prev_bev,
            only_bev=False,
        )
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, rescale=rescale,
        )
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        # bbox_list = [dict() for i in range(len(img_metas))]
        bbox_list = [dict() for _ in range(1)]
        new_prev_bev = outs['bev_embed']
        for result_dict, pts_bbox in zip(bbox_list, bbox_results):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list
