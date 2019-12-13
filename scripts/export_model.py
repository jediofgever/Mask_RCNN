import os
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import graph_util

import mrcnn.model as modellib

def export(inference_config,
           train_log_dirpath,
           model_filepath = None):
    # pb filename
    filename = 'mask_rcnn.pb'
    # Build the inference model
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=train_log_dirpath)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_filepath = model_filepath if model_filepath else model.find_last()[1]

    # Load trained weights (fill in path to trained weights here)
    assert model_filepath, "Provide path to trained weights"
    print("Loading weights from ", model_filepath)
    model.load_weights(model_filepath, by_name=True)

    # Get keras model and save
    model_keras= model.keras_model

    # All new operations will be in test mode from now on.
    K.set_learning_phase(0)

    # Create output layer with customized names
    num_output = 7
    pred_node_names = ["detections", "mrcnn_class", "mrcnn_bbox", "mrcnn_mask",
                       "rois", "rpn_class", "rpn_bbox"]
    pred_node_names = ["output_" + name for name in pred_node_names]
    pred = [tf.identity(model_keras.outputs[i], name = pred_node_names[i])
            for i in range(num_output)]


    sess = K.get_session()


    # Get the object detection graph
    od_graph_def = graph_util.convert_variables_to_constants(sess,
                       sess.graph.as_graph_def(),
                       pred_node_names)

    model_dirpath = os.path.dirname(model_filepath)
    pb_filepath = os.path.join(model_dirpath, filename)
    print('Saving frozen graph {} ...'.format(os.path.basename(pb_filepath)))

    frozen_graph_path = pb_filepath
    with tf.gfile.GFile(frozen_graph_path, 'wb') as f:
        f.write(od_graph_def.SerializeToString())
    print('{} ops in the frozen graph.'.format(len(od_graph_def.node)))
    print()
