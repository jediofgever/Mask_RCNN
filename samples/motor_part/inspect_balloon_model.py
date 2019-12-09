#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN - Inspect Ballon Trained Model
# 
# Code and visualizations to test, debug, and evaluate the Mask R-CNN model.



import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.draw
from skimage import img_as_ubyte
import cv2
import h5py

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.motor_part import motor_part



# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")




config = motor_part.MotorPartConfig()
MOTOR_PART_DATA_DIR =  "/home/atas/catkin_ws/src/ROS_FANUC_LRMATE200ID/inference"


# In[4]:


# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Notebook Preferences

# In[5]:


# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"



def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# ## Load test Dataset


dataset = motor_part.MotorPartDataset()
dataset.load_motor_part(MOTOR_PART_DATA_DIR)
dataset.prepare()



# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)


# Or, load the last model you trained
weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

def run_inference():

    # run inference in first image
    image_id = 0

    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]

    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results

    r = results[0]
    '''
    outfile = "/home/atas/catkin_ws/src/ROS_FANUC_LRMATE200ID/inference/masks.npy"
    np.save(outfile, r['masks'])
    
    data = np.array(r['masks'], dtype = np.bool_)
    f = h5py.File('/home/atas/catkin_ws/src/ROS_FANUC_LRMATE200ID/inference/masks.h5', 'w')
    f.create_dataset('data', data = data)
   


      
    '''
    cv_image = cv2.imread(info["path"],cv2.IMREAD_COLOR)

    segmented_image = motor_part.segment_objects_on_white_image(cv_image,r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores']) 

    window_name = "/home/atas/catkin_ws/src/ROS_FANUC_LRMATE200ID/inference/masks.png"
   
    cv2.imwrite(window_name, segmented_image)  
    
    print("inferring more images  ...")
 
if __name__ == '__main__':
    while True:
        run_inference()
    

 


