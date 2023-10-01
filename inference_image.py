from mmdet.apis import inference_detector
from mmdet.apis import init_detector
from cfg import cfg

import argparse
import mmcv
import glob as glob
import os

# Contruct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', default='input/inference_data',
    help='path to the input data'
)
parser.add_argument(
    '-w', '--weights', 
    default='/outputs/faster_rcnn_hrnetv2p_w18_1x_coco/example_path.pth',
    help='weight file name'
)
parser.add_argument(
    '-t', '--threshold', default=0.5, type=float,
    help='detection threshold for bounding box visualization'
)
args = vars(parser.parse_args())

# Build the model.
model = init_detector(cfg, args['weights'])

image_paths = glob.glob(f"{args['input']}/*.jpg")

for i, image_path in enumerate(image_paths):
    image = mmcv.imread(image_path)
    # Carry out the inference.
    result = inference_detector(model, image)
    # Show the results.
    frame = model.show_result(image, result, score_thr=args['threshold'])
    #mmcv.imshow(frame)
    # Initialize a file name to save the reuslt.
    save_name = f"{image_path.split(os.path.sep)[-1].split('.')[0]}"
    mmcv.imwrite(frame, f"outputs/{save_name}.jpg")
