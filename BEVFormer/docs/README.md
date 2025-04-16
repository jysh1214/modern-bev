# BEVFormer

## Prerequisites

### Prepare Docker Image

Build the docker image with the Dockerfile.

```bash
# at BEVFormer root
docker build -t modern_bevformer_image .

docker run --name modern_bevformer_container --shm-size 64gb --gpus all --mount src=$PWD,target=/home/BEVFormer,type=bind -it modern_bevformer_image /bin/bash
```

### Prepare Pretrained Models

```bash
# at BEVFormer root
mkdir ckpts
cd ckpts
# for bevformer_base
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth
# for bevformer_tiny
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth
```

### Prepare nuScenes Dataset

Download the data from [nuScenes](https://www.nuscenes.org/download).

- v1.0-test_meta.tgz
- v1.0-test_blobs.tgz
- v1.0-mini.tgz

```bash
# at BEVFormer root
mkdir data
cd data

mkdir nuscenes
tar xf v1.0-test_meta.tgz -C nuscenes
tar xf v1.0-test_blobs.tgz -C nuscenes
tar xf v1.0-mini.tgz -C nuscenes
mv nuscenes/v1.0-mini nuscenes/v1.0-trainval

unzip can_bus.zip
```

Prepare the data related to nuScenes dataset.
```bash
python3.9 tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```

### Install Dependencies

Install mmdet3d from source code.
```bash
git clone -b v0.17.1 https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
# issue: https://github.com/open-mmlab/mmdetection/issues/9580
python3.9 setup.py build   # build without installing
pip3.9 install --no-deps . # install without dependencies
```

Install mmcv from source code.
```bash
git clone -b v1.4.0 https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip3.9 install -e . -v
```


## Demo

```bash
# for bevformer_base
./tools/dist_test.sh ./projects/configs/bevformer/bevformer_base.py ./ckpts/bevformer_r101_dcn_24ep.pth 1
# for bevformer_tiny
./tools/dist_test.sh ./projects/configs/bevformer/bevformer_tiny.py ./ckpts/bevformer_tiny_epoch_24.pth 1
```


## Visulization

```bash
python3.9 tools/analysis_tools/visual.py test/$MODEL/$DATE/pts_bbox/results_nusc.json
```


## Trouble Shooting

### Numba error

**Error**
```bash
    from numba.errors import NumbaPerformanceWarning
ModuleNotFoundError: No module named 'numba.errors'
```

**Solution**
Comment out the following lines in `/usr/local/lib/python3.9/site-packages/mmdet3d/datasets/pipelines/data_augment_utils.py`:
```python
# line 5: from numba.errors import NumbaPerformanceWarning
# line 9: warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)
```
