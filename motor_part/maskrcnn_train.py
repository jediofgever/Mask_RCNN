
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2

from skimage import img_as_ubyte
from skimage.draw import rectangle
from skimage.color import rgb2xyz
import random

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
print(ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = ROOT_DIR + "/logs/real_data_30_epoch.h5"
TRAIN_JSON_FILE = "/home/royle/catkin_ws/ws_py3_nn/src/CNN_DETS_PICKPLACE/Mask_RCNN/data/labels.json"
TRANSFER_WEIGHTS ="coco" 
DATASET_DIR = "/home/royle/catkin_ws/ws_py3_nn/src/CNN_DETS_PICKPLACE/Mask_RCNN/data/"

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class MotorPartConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.

    """
    GPU_COUNT = 1

    # Give the configuration a recognizable name
    NAME = "motor_part"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    IMAGE_RESIZE_MODE = "square"

    BATCH_SIZE = 1




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
        with open(TRAIN_JSON_FILE, 'r') as myfile:
            data=myfile.read()

        # parse file
        annotations = json.loads(data)

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


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = MotorPartDataset()
    dataset_train.load_motor_part(DATASET_DIR)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = MotorPartDataset()
    dataset_val.load_motor_part(DATASET_DIR)
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')
############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse
 
    config = MotorPartConfig()
    config.display()
    model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

    # Select weights file to load
    if TRANSFER_WEIGHTS == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif TRANSFER_WEIGHTS == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif TRANSFER_WEIGHTS == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        
    train(model)

