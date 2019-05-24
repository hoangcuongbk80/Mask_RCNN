"""
Mask R-CNN
Train on the toy Warehouse dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

python3 warehouse_inference.py --weights=/home/aass/Hoang-Cuong/Mask_RCNN/
logs/warehouse20190524T1156/mask_rcnn_warehouse_0060.h5 --image=1.png

python3 warehouse_inference.py --weights=/home/aass/Hoang-Cuong/Mask_RCNN/
logs/warehouse20190524T1156/mask_rcnn_warehouse_0060.h5 --video=/home/aass/Hoang-Cuong/datasets/warehouse_ECMR/0008/

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import glob

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class WarehouseConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Warehouse"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 13  # Background + objects

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

def get_masks(image, mask, class_ids):
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255

    instance_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    semantic_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    if mask.shape[-1] > 0:
        mask_zero = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        for i in range(mask.shape[-1]):
            semantic_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)            
            semantic_mask_one = semantic_mask_one * class_ids[i]
            semantic_masks = np.where(mask[:, :, i], semantic_mask_one, semantic_masks).astype(np.uint8)
            instance_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)            
            instance_mask_one = instance_mask_one * (i+1)
            instance_masks = np.where(mask[:, :, i], instance_mask_one, instance_masks).astype(np.uint8)           
    
    return semantic_masks, instance_masks

def detect_and_get_masks(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        semantic_masks, instance_masks = get_masks(image, r['masks'], r['class_ids'])
        plt.title('mask')
        plt.imshow(instance_masks)
        plt.show()
        plt.title('label')
        plt.imshow(instance_masks)
        plt.show()
        # Save output
        file_name = "mask_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, instance_masks)
        print("Saved to ", file_name)
    elif video_path:
        print('cuong')
        print(video_path)
        data_path = video_path
        rgb_path = data_path + 'rgb/'
        label_path = data_path + 'mask_rcnn/label/'
        mask_path = data_path + 'mask_rcnn/mask/'
        rgb_addrs = glob.glob(rgb_path + '*.png')
        label_addrs = glob.glob(label_path + '*.png')
        inferenced_list = []
        
        for label_addr in label_addrs:
            inferenced_list.append(label_addr[len(label_path):])
            #print("inferenced_list")
            #print(inferenced_list)

        if data_path[len(data_path)-1] != '/':
            print ('The data path should have / in the end')
            exit()

        for i in range(len(rgb_addrs)):
            print ( 'Image: {}/{}'.format(i, len(rgb_addrs)))
            print (rgb_addrs[i])
            if rgb_addrs[i][len(rgb_path):] in inferenced_list:
                print('already inferenced')
            else:
                # Read image
                image = skimage.io.imread(rgb_addrs[i])
                # Detect objects
                r = model.detect([image], verbose=1)[0]
                # get instance_masks
                semantic_masks, instance_masks  = get_masks(image, r['masks'], r['class_ids'])
                
                label_addr = data_path + 'mask_rcnn/label/' + rgb_addrs[i][len(rgb_path):]
                skimage.io.imsave(label_addr, semantic_masks)
                mask_addr = data_path + 'mask_rcnn/mask/' + rgb_addrs[i][len(rgb_path):]
                skimage.io.imsave(mask_addr, instance_masks)
                print(mask_addr)

    
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
                        description='Train Mask R-CNN to detect Warehouses.')
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/Warehouse/dataset/",
                        help='Directory of the Warehouse dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    class InferenceConfig(WarehouseConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
    weights_path = args.weights
    model.load_weights(weights_path, by_name=True)
    
    detect_and_get_masks(model, image_path=args.image, video_path=args.video)