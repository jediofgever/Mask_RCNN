#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN - Inspect Ballon Trained Model
#
# Code and visualizations to test, debug, and evaluate the Mask R-CNN model.


from motor_part.maskrcnn_train import  MotorPartConfig
from mrcnn.config import Config
import json
import datetime
from mrcnn.model import log
import mrcnn.model as modellib
import mrcnn.utils as utils
from skimage.io import imsave, imread

from mrcnn import utils
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import skimage

import cv2
import h5py
import sys
import time

# Ros libraries
import roslib
import rospy
# Ros Messages
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from skimage.util import img_as_float


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


config = MotorPartConfig()
# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
 
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
weights_path = "/home/atas/catkin_build_ws/src/ROS_NNs_FANUC_LRMATE200ID/Mask_RCNN/logs/real_data_30_epoch.h5"
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)
graph = tf.get_default_graph()


class Mask_RCNN_ROS_Node:

    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.image_pub = rospy.Publisher("/output/maskrcnn/segmented",
                                         Image)

        self.subscriber = rospy.Subscriber("/camera/color/image_raw",
                                           Image, self.callback,  queue_size=1)
        self.bridge = CvBridge()
        self.counter = 500 

    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and OBJECTS detected'''
        #### direct conversion to CV2 ####
        cv_image = self.bridge.imgmsg_to_cv2(ros_data, desired_encoding="bgr8")

        image = img_as_float(cv_image)
        imsave("/home/atas/sKI.png", image)
 

        imsave("/home/atas/real_img_data/"+str(self.counter)+".png", image)
        image = imread("/home/atas/sKI.png")
        self.counter +=1
        image = self.load_image(image)
        # Run object detection
        global graph
        with graph.as_default():
            results = model.detect([image], verbose=1)

        # GET results

        r = results[0]

        segmented_image = self.segment_objects_on_white_image(cv_image, r['rois'], r['masks'], r['class_ids'],
                                                          r['scores'])
        #### PUBLISH SEGMENTED IMAGE ####
        msg = self.bridge.cv2_to_imgmsg(segmented_image, "bgr8")
        msg.header.stamp = rospy.Time.now()
        self.image_pub.publish(msg)

    def load_image(self, image):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def segment_objects_on_white_image(self,image, boxes, masks, class_ids,
                                       scores=None,):
        """Apply color splash effect.
        image: RGB image [height, width, 3]
        mask: instance segmentation mask [height, width, instance count]
        Returns result image.
        """
        # Make a grayscale copy of the image. The grayscale copy still
        # has 3 RGB channels, though.
        #xyz = rgb2xyz(image)
        height = image.shape[0]
        width = image.shape[1]
        channels = image.shape[2]





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
            if(score < 0.99):
                break
 
          
            #label = class_names[class_id]
            print(class_id, score)
            #caption = "{} {:.3f}".format(label, score) if score else label

            # Mask
            mask = masks[:, :, i]
            object_mask_image = np.zeros((height, width, 1), np.uint8)

            for i in range(0, height):
                for j in range(0, width):
                    val = mask[i][j]
                    object_mask_image[i][j] = val

            contours, hierarchy = cv2.findContours(
                object_mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.fillPoly(white_image, contours, (random.randint(
                0, 255), random.randint(0, 255), random.randint(0, 255)))

        return white_image


# Run Node
if __name__ == '__main__':
    '''Initializes and cleanup ros node'''
    ic = Mask_RCNN_ROS_Node()
    rospy.init_node('Mask_RCNN_ROS_Node', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")