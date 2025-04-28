import argparse
import os
import torch
import numpy as np

from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from mmcv import Config, DictAction
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from configs.configs import BEVFORMER_CONFIG


def prepare_model(config):
    cfg = Config.fromfile(config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # set random seeds
    # if args.seed is not None:
    #     set_random_seed(args.seed, deterministic=args.deterministic)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    return model


def prepare_data_loader(config, shuffle=False):
    dataset = build_dataset(config)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        shuffle=False,
    )
    return data_loader


def verbose_info(args, message, newline=True):
    if args.verbose:
        print(message, flush=True, end="\n" if newline else "")


def compare(item, golden, test, args):
    if golden.dtype == "float32":
       eq = np.allclose(golden, test)
    elif golden.dtype == "int64":
       eq = np.array_equal(golden, test)
    else:
        print(f"Unknown dtype: {golden.dtype}")
        exit(-1)

    result = "PASSED" if eq else "FAILED"
    verbose_info(args, f"- Comparing the {item} with the golden data...{result}!")
    diff = np.abs(golden - test)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    verbose_info(args, f"    - max absolute difference: {max_diff}")
    verbose_info(args, f"    - mean absolute difference: {mean_diff}")


def run(model, data_loader, args):
    model.to(args.device)
    model.eval()

    dataset = data_loader.dataset
    for index, data in enumerate(data_loader):
        rescale = True
        img = data['img'][0].data[0].to(args.device)
        img_metas = data['img_metas'][0].data[0][0]

        ori_shape = torch.tensor(img_metas['ori_shape'])
        img_shape = torch.tensor(img_metas['img_shape'])
        lidar2img = torch.tensor(img_metas['lidar2img'])
        lidar2cam = torch.tensor(img_metas['lidar2cam'])
        pad_shape = torch.tensor(img_metas['pad_shape'])
        scale_factor = img_metas['scale_factor']
        flip = img_metas['flip']
        pcd_horizontal_flip = img_metas['pcd_horizontal_flip']
        pcd_vertical_flip = img_metas['pcd_vertical_flip']
        can_bus = torch.tensor(img_metas['can_bus'])

        boxes_3d, scores_3d, labels_3d = None, None, None
        with torch.no_grad():
            print(f"Runing {index} iter...", end="", flush=True)
            bbox_results = model(
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
            print(f"finished.", flush=True)
            boxes_3d = bbox_results[0]['pts_bbox']['boxes_3d'].tensor
            scores_3d = bbox_results[0]['pts_bbox']['scores_3d']
            labels_3d = bbox_results[0]['pts_bbox']['labels_3d']

        if not args.check:
            continue

        filename = img_metas['filename'][0].split('__')[2].split('.')[0]
        golden_dir = f"./assets/{filename}"
        golden_boxes_3d_path = os.path.join(golden_dir, f"{filename}_boxes_3d_300x9xf32.npy")
        golden_scores_3d_path = os.path.join(golden_dir, f"{filename}_scores_3d_300xf32.npy")
        golden_labels_3d_path = os.path.join(golden_dir, f"{filename}_labels_3d_300xsi64.npy")

        golden_boxes_3d = np.load(golden_boxes_3d_path)
        golden_scores_3d = np.load(golden_scores_3d_path)
        golden_labels_3d = np.load(golden_labels_3d_path)

        compare("boxes", golden_boxes_3d, boxes_3d.numpy(), args)
        compare("scores", golden_scores_3d, scores_3d.numpy(), args)
        compare("labels", golden_labels_3d, labels_3d.numpy(), args)


def check_args(args):
    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available"
        torch.cuda.set_device(0)


def create_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device to use")
    parser.add_argument("--checkpoint", default="./ckpts/bevformer_r101_dcn_24ep.pth", help="Checkpoint file")
    parser.add_argument("--check", default=False, action="store_true", help="Check the results")
    parser.add_argument("--verbose", default=False, action="store_true", help="Display verbose output")
    return parser


"""
PYTHONPATH="$(pwd)":$PYTHONPATH python3.9 tools/inference.py --check --verbose
"""
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    check_args(args)

    model = prepare_model("./projects/configs/bevformer/bevformer_base.py")
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    data_loader = prepare_data_loader(BEVFORMER_CONFIG["dataset"])

    run(model, data_loader, args)
