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

import datetime
import numpy as np
import skimage.draw
import cv2
import json

from skimage import img_as_ubyte
from skimage.draw import rectangle
from skimage.color import rgb2xyz
import random
from mrcnn.config import Config



class MotorPartConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "motor_part"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class MotorPartDataset(utils.Dataset):

    def load_motor_part(self, dataset_dir):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("motor_part", 1, "motor_part")

        # Train or validation dataset?
        #assert subset in ["train", "val"]
        #dataset_dir = os.path.join(dataset_dir)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        with open('/home/atas/catkin_ws/src/ROS_FANUC_LRMATE200ID/inference/test.json', 'r') as myfile:
            data=myfile.read()

        # parse file
        annotations = json.loads(data)
        #annotations = json.load(open('/media/atas/b7743b4f-8b7a-46b5-bd01-cb2efaeedf63/home/atas/Dataset_Synthesizer/Source/NVCapturedData/NewMap/label.json'))
        #annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        #annotations = [a for a in annotations if a['regions']]


        via_1_check = annotations.get('regions')
        via_2_check = annotations.get('_via_img_metadata')

        # JSON is formatted with VIA-1.x
        if via_1_check:
            annotations = list(annotations.values())
        # JSON is formatted with VIA-2.x
        elif via_2_check:
            annotations = list(annotations['_via_img_metadata'].values())
        # Unknown JSON formatting
        else:
            raise ValueError('The JSON provided is not in a recognised via-1.x or via-2.x format.')
    
        #annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "motor_part",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "motor_part":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "motor_part":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def segment_objects_on_white_image(image, boxes, masks, class_ids, class_names,
                      scores=None,):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    #xyz = rgb2xyz(image)
    height = 1024#image.shape[0]
    width = 1024#image.shape[1]
    channels = 3#image.shape[2] 
 
    # Copy color pixels from the original color image where mask is set
    if masks.shape[-1] > 0:
        fuck_python = True

    N = boxes.shape[0]

    white_image = np.zeros((height, width, channels), np.uint8)
    white_image[:] = (255, 255, 255)

    for i in range(N):

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        #y1, x1, y2, x2 = boxes[i]
        
        #boxed_image = cv2.rectangle(image,(x1,y1),(x2,y2),(0,225,0),1)
        

        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        caption = "{} {:.3f}".format(label, score) if score else label
        
        # Mask
        mask = masks[:, :, i]
        object_mask_image = np.zeros((height,width,1), np.uint8)
        

        for i in range(0,height):
            for j in range(0,width):
                val = mask[i][j]
                object_mask_image[i][j] = val

        contours, hierarchy = cv2.findContours(
                object_mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
        cv2.fillPoly(white_image, contours, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))


    return white_image



# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")




config = MotorPartConfig()
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


dataset = MotorPartDataset()
dataset.load_motor_part(MOTOR_PART_DATA_DIR)
dataset.prepare()



# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)


# Or, load the last model you trained
#weights_path = model.find_last()
weights_path = "/home/atas/catkin_ws/src/ROS_FANUC_LRMATE200ID/Mask_RCNN/logs/motor_part20191203T2300/mask_rcnn_motor_part_0030.h5"

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)
'''
from scripts.export_model import export

export(config, MODEL_DIR, weights_path)

model.keras_model.save("mrcnn.h5")
print("EXITING .....")

exit(0)
'''
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

    segmented_image = segment_objects_on_white_image(cv_image,r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores']) 

    window_name = "/home/atas/catkin_ws/src/ROS_FANUC_LRMATE200ID/inference/masks.png"
   
    cv2.imwrite(window_name, segmented_image)  
    
    print("inferring more images  ...")
 
if __name__ == '__main__':
    while True:
        run_inference()
    

 


