# Copyright (C) 2017 DataArt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time
import logging.config
import cv2
import pafy
import tensorflow as tf

# Yitao-TLS-Begin
import os
import sys
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS
# Yitao-TLS-End

from models import yolo
from log_config import LOGGING
from utils.general import format_predictions, find_class_by_name, is_url

logging.config.dictConfig(LOGGING)

logger = logging.getLogger('detector')
FLAGS = tf.flags.FLAGS


def evaluate(_):

    imgcv = cv2.imread("/home/yitao/Documents/fun-project/devicehive-video-analysis/1.png")
    source_h, source_w, _ = imgcv.shape

    model_cls = find_class_by_name(FLAGS.model_name, [yolo])
    model = model_cls(input_shape=(source_h, source_w, 3))
    # model = model_cls(input_shape=(608, 608, 3))        # hard-coded (608, 608) here, so all image for model.evaluate(image) should be resized to (608, 608) first
    model.init()

    # # Yitao-TLS-Begin
    # export_path_base = "devicehive_yolo"
    # export_path = os.path.join(
    #   compat.as_bytes(export_path_base),
    #   compat.as_bytes(str(FLAGS.model_version)))
    # print 'Exporting trained model to', export_path
    # builder = saved_model_builder.SavedModelBuilder(export_path)

    # tensor_info_x = tf.saved_model.utils.build_tensor_info(model._eval_inp)
    # tensor_info_y = tf.saved_model.utils.build_tensor_info(model._eval_ops)

    # prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
    #   inputs={'input': tensor_info_x},
    #   outputs={'output': tensor_info_y},
    #   method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    # legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    # builder.add_meta_graph_and_variables(
    #   model._sess, [tf.saved_model.tag_constants.SERVING],
    #   signature_def_map={
    #       'predict_images':
    #           prediction_signature,
    #   },
    #   legacy_init_op=legacy_init_op)

    # builder.save()

    # print('Done exporting!')
    # # Yitao-TLS-End




    # # Yitao-TLS-Begin
    # y1, y2, y3 = model._eval_ops

    # export_path_base = "devicehive_yolo"
    # export_path = os.path.join(
    #   compat.as_bytes(export_path_base),
    #   compat.as_bytes(str(FLAGS.model_version)))
    # print 'Exporting trained model to', export_path
    # builder = saved_model_builder.SavedModelBuilder(export_path)

    # tensor_info_x = tf.saved_model.utils.build_tensor_info(model._eval_inp)
    # tensor_info_y1 = tf.saved_model.utils.build_tensor_info(y1)
    # tensor_info_y2 = tf.saved_model.utils.build_tensor_info(y2)
    # tensor_info_y3 = tf.saved_model.utils.build_tensor_info(y3)

    # prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
    #   inputs={'input': tensor_info_x},
    #   outputs={ 'boxes': tensor_info_y1,
    #             'scores': tensor_info_y2,
    #             'classes': tensor_info_y3},
    #   method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    # legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    # builder.add_meta_graph_and_variables(
    #   model._sess, [tf.saved_model.tag_constants.SERVING],
    #   signature_def_map={
    #       'predict_images':
    #           prediction_signature,
    #   },
    #   legacy_init_op=legacy_init_op)

    # builder.save()

    # print('Done exporting!')
    # # Yitao-TLS-End



    for i in range(1):
        imgcv = cv2.imread("/home/yitao/Documents/fun-project/devicehive-video-analysis/1.png")
        print(imgcv.shape)
        # imgcv = cv2.resize(imgcv, (608, 608))
        predictions = model.evaluate(imgcv)
        print(predictions)
    print("warmup finished...")

    # iteration_list = [1]
    # for iteration in iteration_list:
    #     start = time.time()
    #     for i in range(iteration):
    #         imgcv = cv2.imread("/home/yitao/Documents/TF-Serving-Downloads/dog.jpg")
    #         print(imgcv.shape)
    #         imgcv = cv2.resize(imgcv, (608, 608))
    #         predictions = model.evaluate(imgcv)
    #         print(predictions)
    #     end = time.time()
    #     print("[iteration = %d] It takes %s sec to run %d images for YOLO-devicehive" % (iteration, str(end - start), iteration))        



if __name__ == '__main__':
    # tf.flags.DEFINE_string('video', 0, 'Path to the video file.')
    tf.flags.DEFINE_string('model_name', 'Yolo2Model', 'Model name to use.')

    tf.app.run(main=evaluate)
