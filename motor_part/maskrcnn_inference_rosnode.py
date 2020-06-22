 

from motor_part.maskrcnn_train import  MotorPartConfig
from mrcnn.config import Config
import json
import datetime
from mrcnn.model import log
import mrcnn.model as modellib
import mrcnn.utils as utils
from skimage.io import imsave, imread

from mrcnn import utils
import os, os.path
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
    BATCH_SIZE = 1
    IMAGE_MIN_DIM = 540
    IMAGE_MAX_DIM = 960


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
this_file_path = os.path.dirname(os.path.realpath(__file__))
weights_path = this_file_path + "/../logs/real_data_30_epoch.h5"
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
                                           Image, self.callback, queue_size=1, buff_size=2002428800)
        self.bridge = CvBridge()                                                         
        self.counter = 1200
        self.start_time = time.time()
        self.x = 1 # displays the frame rate every 1 second


    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and OBJECTS detected'''
        #### direct conversion to CV2 ####
        cv_image = self.bridge.imgmsg_to_cv2(ros_data, desired_encoding="bgr8")
        #cv_image =  cv2.resize(cv_image, (480,240), interpolation = cv2.INTER_AREA)        
        # Uncomment thefollowing block in order to collect training data
        '''
        cv2.imwrite("/home/atas/MASKRCNN_REAL_DATASET/"+str(self.counter)+".png",cv_image)
        self.counter = self.counter +1 
        '''
        # Run object detection
        global graph
        with graph.as_default():
            results = model.detect([cv_image], verbose=1)

        # GET results
        r = results[0]
         
        segmented_image = self.segment_objects_on_white_image(cv_image, r['rois'], r['masks'], r['class_ids'],
                                                          r['scores'])
        #### PUBLISH SEGMENTED IMAGE ####
        msg = self.bridge.cv2_to_imgmsg(segmented_image, "bgr8")
        msg.header.stamp = rospy.Time.now()
        self.image_pub.publish(msg)
        self.counter+=1
        if (time.time() - self.start_time) > self.x :
            print("FPS: ", self.counter / (time.time() - self.start_time))
            self.counter = 0
            self.start_time = time.time()    
         

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

        N = boxes.shape[0]

        white_image = np.zeros((height, width, channels), np.uint8)
        white_image[:] = (255, 255, 255)
        object_mask_image = np.zeros((height, width), np.uint8)
        kernel = np.ones((21,21),np.uint8)

        for i in range(N):

            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue

            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            if(score < 0.6):
                break
            # Mask
            object_mask_image[:,:] = masks[:, :, i]
    
            contours, hierarchy = cv2.findContours(
                object_mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.fillPoly(white_image, contours, (random.randint(
                0, 255), random.randint(0, 255), random.randint(0, 255)))
               
  
        #white_image = cv2.erode(white_image,kernel,iterations = 1)
        return white_image


# Run Node
if __name__ == '__main__':
    '''Initializes and cleanup ros node'''
    ic = Mask_RCNN_ROS_Node()
    rospy.init_node('Mask_RCNN_ROS_Node', anonymous=True)
    rospy.Rate(30)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
