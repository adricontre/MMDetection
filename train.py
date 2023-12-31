from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from cfg import cfg

import os.path as osp
import mmcv

# Build dataset

if 'cfg' in cfg:
    cfg = cfg['cfg']
    

if 'model' not in cfg or 'data' not in cfg:
    print("Invalid configuration file. 'model' or 'data' keys are missing.")
    exit(1)
    
#TODO not for YOLO
# Remove the 'dataset' key and move its contents to the upper level
if 'dataset' in cfg.data.train:
    cfg.data.train = cfg.data.train['dataset']
    

datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(cfg.model)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)
