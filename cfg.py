from mmcv import Config
from mmdet.apis import set_random_seed
from dataset import XMLCustomDataset


cfg = Config.fromfile('/mmdetection/configs/faster_rcnn/faster_rcnn_r101_fpn_mstrain_3x_coco.py') 
print(f"Default Config:\n{cfg.pretty_text}")



# Modify dataset type and path.
cfg.dataset_type = 'XMLCustomDataset'
cfg.data_root = '/input/data_root/'

cfg.data.test.type = 'XMLCustomDataset'
cfg.data.test.data_root = '/input/data_root/'
cfg.data.test.ann_file = 'dataset/ImageSets/Main/val.txt'
cfg.data.test.img_prefix = 'dataset/'

# For yolo has to have .dataset after train.
cfg.data.train.type = 'XMLCustomDataset'
cfg.data.train.data_root = '/input/data_root/'
cfg.data.train.ann_file = 'dataset/ImageSets/Main/train.txt'
cfg.data.train.img_prefix = 'dataset/'

cfg.data.val.type = 'XMLCustomDataset'
cfg.data.val.data_root = '/input/data_root/'
cfg.data.val.ann_file = 'dataset/ImageSets/Main/val.txt'
cfg.data.val.img_prefix = 'dataset/'

# For faster rcnn (example)
#cfg.data.train.pipeline = [
#    dict(type='LoadImageFromFile'),
#    dict(type='LoadAnnotations', with_bbox=True),
#    dict(type='RandomFlip', flip_ratio=.5),
#    #dict(type='RandomCrop', height=800, width=800), # Random cropping
#    #dict(type='RandomRotate', level=10), # Random rotation
#    dict(type='Resize', img_scale=[(800, 800), (1000, 1000), (1200, 1200)], keep_ratio=True, multiscale_mode='value'), # multiscaling
#    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
#    dict(type='Pad', size_divisor=32),  # You may need to adjust the size_divisor according to the network requirements
#    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
#    dict(type='DefaultFormatBundle'),
#    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
#]

# For YOLO (example)
#cfg.data.train.pipeline = [
#   dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
#    dict(
#        type='RandomAffine',
#        scaling_ratio_range=(0.1, 2),
#        border=(-320, -320)),
#    dict(
#        type='MixUp',
#        img_scale=(640, 640),
#        ratio_range=(0.8, 1.6),
#        pad_val=114.0),
#    dict(type='YOLOXHSVRandomAug'),
#    dict(type='RandomFlip', flip_ratio=0.5),
#    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
#    dict(
#        type='Pad',
#        pad_to_square=True,
#        pad_val=dict(img=(114.0, 114.0, 114.0))),
#    dict(
#        type='FilterAnnotations',
#        min_gt_bbox_wh=(1, 1),
#        keep_empty=False),
#    dict(type='DefaultFormatBundle'),
#    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
#    ]

# Batch size (samples per GPU).
cfg.data.samples_per_gpu = 2

# Modify number of classes as per the model head.
#cfg.model.bbox_head.num_classes = 1
cfg.model.roi_head.bbox_head.num_classes = 1 # for faster rcnn

# Comment/Uncomment this to training from scratch/fine-tune according to the 
# model checkpoint path. 
cfg.load_from = '/home/adrian/Debugger_cafe/Custom_Dataset_Training_using_MMDetection/checkpoints/faster_rcnn_r101_fpn_mstrain_3x_coco_20210524_110822-4d4d2ca8.pth'


# Learning rate
cfg.optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.0001)
#cfg.optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001) #SGD (Stochastic Gradient Descent): The most basic but effective optimizer.
#cfg.optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.001) #AdamW: A variant of Adam that decouples weight decay from the optimization steps.
#cfg.optimizer = dict(type='Adagrad', lr=0.01)# cfg.optimizer = dict(type='Adagrad', lr=0.01)
#cfg.optimizer = dict(type='Adadelta', lr=1.0, rho=0.9) #Adadelta: Similar to Adagrad but tries to overcome its rapidly decreasing learning rates.

#Learning rates Scheduling
#cfg.lr_config = dict(policy='CosineAnnealing', min_lr_ratio=0.01) #Cosine Annealing: Gradually reduces the learning rate following a cosine curve.
cfg.lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[20,40]
) #Step Decay: Reduces the learning rate after a certain number of epochs.


cfg.log_config.interval = 10

# The output directory for training. As per the model name.
cfg.work_dir = '/outputs/faster_rcnn_r101_fpn_mstrain_3x_coco'
# Evaluation Metric.
cfg.evaluation.metric = 'mAP'
cfg.evaluation.save_best = 'mAP'
# Evaluation times.
cfg.evaluation.interval = 5
# Checkpoint storage interval.
cfg.checkpoint_config.interval = 20

# Set random seed for reproducible results.
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = 'cuda'
cfg.runner.max_epochs = 40

# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

# We can initialize the logger for training and have a look
# at the final config used for training
print('#'*50)
print(f'Config:\n{cfg.pretty_text}')
