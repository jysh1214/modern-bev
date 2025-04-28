BEVFORMER_CONFIG = {
    "dataset": {
        "type": "CustomNuScenesDataset",
        "data_root": "data/nuscenes/",
        "ann_file": "data/nuscenes/nuscenes_infos_temporal_val.pkl",
        "pipeline": [
            {
                "type": "LoadMultiViewImageFromFiles",
                "to_float32": True
            },
            {
                "type": "NormalizeMultiviewImage",
                "mean": [103.53, 116.28, 123.675],
                "std": [1.0, 1.0, 1.0],
                "to_rgb": False
            },
            {
                "type": "PadMultiViewImage",
                "size_divisor": 32
            },
            {
                "type": "MultiScaleFlipAug3D",
                "img_scale": (1600, 900),
                "pts_scale_ratio": 1,
                "flip": False,
                "transforms": [
                    {
                        "type": "DefaultFormatBundle3D",
                        "class_names": [
                            "car",
                            "truck",
                            "construction_vehicle",
                            "bus",
                            "trailer",
                            "barrier",
                            "motorcycle",
                            "bicycle",
                            "pedestrian",
                            "traffic_cone"
                        ],
                        "with_label": False
                    },
                    {
                        "type": "CustomCollect3D",
                        "keys": ["img"]
                    }
                ]
            }
        ],
        "classes": [
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone"
        ],
        "modality": {
            "use_lidar": False,
            "use_camera": True,
            "use_radar": False,
            "use_map": False,
            "use_external": True
        },
        "test_mode": True,
        "box_type_3d": "LiDAR",
        "bev_size": (200, 200)
    },
    "model": {
        "type": "BEVFormer",
        "use_grid_mask": True,
        "video_test_mode": True,
        "img_backbone": {
            "type": "ResNet",
            "depth": 101,
            "num_stages": 4,
            "out_indices": (1, 2, 3),
            "frozen_stages": 1,
            "norm_cfg": {
                "type": "BN2d",
                "requires_grad": False
            },
            "norm_eval": True,
            "style": "caffe",
            "dcn": {
                "type": "DCNv2",
                "deform_groups": 1,
                "fallback_on_stride": False
            },
            "stage_with_dcn": (False, False, True, True)
        },
        "img_neck": {
            "type": "FPN",
            "in_channels": [512, 1024, 2048],
            "out_channels": 256,
            "start_level": 0,
            "add_extra_convs": "on_output",
            "num_outs": 4,
            "relu_before_extra_convs": True
        },
        "pts_bbox_head": {
            "type": "BEVFormerHead",
            "bev_h": 200,
            "bev_w": 200,
            "num_query": 900,
            "num_classes": 10,
            "in_channels": 256,
            "sync_cls_avg_factor": True,
            "with_box_refine": True,
            "as_two_stage": False,
            "transformer": {
                "type": "PerceptionTransformer",
                "rotate_prev_bev": True,
                "use_shift": True,
                "use_can_bus": True,
                "embed_dims": 256,
                "encoder": {
                    "type": "BEVFormerEncoder",
                    "num_layers": 6,
                    "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                    "num_points_in_pillar": 4,
                    "return_intermediate": False,
                    "transformerlayers": {
                        "type": "BEVFormerLayer",
                        "attn_cfgs": [
                            {
                                "type": "TemporalSelfAttention",
                                "embed_dims": 256,
                                "num_levels": 1
                            },
                            {
                                "type": "SpatialCrossAttention",
                                "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                                "deformable_attention": {
                                    "type": "MSDeformableAttention3D",
                                    "embed_dims": 256,
                                    "num_points": 8,
                                    "num_levels": 4
                                },
                                "embed_dims": 256
                            }
                        ],
                        "feedforward_channels": 512,
                        "ffn_dropout": 0.1,
                        "operation_order": ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm")
                    }
                },
                "decoder": {
                    "type": "DetectionTransformerDecoder",
                    "num_layers": 6,
                    "return_intermediate": True,
                    "transformerlayers": {
                        "type": "DetrTransformerDecoderLayer",
                        "attn_cfgs": [
                            {
                                "type": "MultiheadAttention",
                                "embed_dims": 256,
                                "num_heads": 8,
                                "dropout": 0.1
                            },
                            {
                                "type": "CustomMSDeformableAttention",
                                "embed_dims": 256,
                                "num_levels": 1
                            }
                        ],
                        "feedforward_channels": 512,
                        "ffn_dropout": 0.1,
                        "operation_order": ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm")
                    }
                }
            },
            "bbox_coder": {
                "type": "NMSFreeCoder",
                "post_center_range": [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                "max_num": 300,
                "voxel_size": [0.2, 0.2, 8],
                "num_classes": 10
            },
            "positional_encoding": {
                "type": "LearnedPositionalEncoding",
                "num_feats": 128,
                "row_num_embed": 200,
                "col_num_embed": 200
            },
            "loss_cls": {
                "type": "FocalLoss",
                "use_sigmoid": True,
                "gamma": 2.0,
                "alpha": 0.25,
                "loss_weight": 2.0
            },
            "loss_bbox": {
                "type": "L1Loss",
                "loss_weight": 0.25
            },
            "loss_iou": {
                "type": "GIoULoss",
                "loss_weight": 0.0
            }
        },
        "train_cfg": {
            "pts": {
                "grid_size": [512, 512, 1],
                "voxel_size": [0.2, 0.2, 8],
                "point_cloud_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                "out_size_factor": 4,
                "assigner": {
                    "type": "HungarianAssigner3D",
                    "cls_cost": {
                        "type": "FocalLossCost",
                        "weight": 2.0
                    },
                    "reg_cost": {
                        "type": "BBox3DL1Cost",
                        "weight": 0.25
                    },
                    "iou_cost": {
                        "type": "IoUCost",
                        "weight": 0.0
                    },
                    "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
                }
            }
        },
        "pretrained": None
    }
}
