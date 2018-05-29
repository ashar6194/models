# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

r"""Convert raw COCO dataset to TFRecord for object_detection.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import os
import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags

tf.flags.DEFINE_string('train_image_dir', '/media/mcao/Miguel/TT100k/data/train_out',
                       'Training image directory.')
tf.flags.DEFINE_string('val_image_dir', '/media/mcao/Miguel/TT100k/data/val_out',
                       'Validation image directory.')
tf.flags.DEFINE_string('test_image_dir', '/media/mcao/Miguel/TT100k/data/test_out',
                       'Test image directory.')
tf.flags.DEFINE_string('train_annotations_file', '/media/mcao/Miguel/TT100k/data/annotations.json',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('val_annotations_file', '/media/mcao/Miguel/TT100k/data/annotations.json',
                       'Validation annotations JSON file.')
tf.flags.DEFINE_string('test_annotations_file', '/media/mcao/Miguel/TT100k/data/annotations.json',
                       'Test-dev annotations JSON file.')
tf.flags.DEFINE_string('output_dir', '/media/mcao/Miguel/TT100k/data/tfrecord', 'Output data directory.')
tf.flags.DEFINE_string('label_map_path', './object_detection/data/tt100k_label_map.pbtxt',
                       'label_map_path')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def create_tf_example(image_full_path, annotated_data, image_id, label_map_dict):
  filename = image_full_path
  image_id = image_id

  full_path = filename
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()

  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  image = np.asarray(image)
  image_width = int(image.shape[1])
  image_height = int(image.shape[0])
  key = hashlib.sha256(encoded_jpg).hexdigest()

  xmin = []
  xmax = []
  ymin = []
  ymax = []
  category_ids = []
  num_annotations_skipped = 0

  object_list = annotated_data['imgs'][image_id]['objects']
  for objects in object_list:
    category = objects['category']
    if category in ['pn', 'pne']:
      category_ids.append(label_map_dict[objects['category']])
      xmin.append(objects['bbox']['xmin'])
      xmax.append(objects['bbox']['xmax'])
      ymin.append(objects['bbox']['ymin'])
      ymax.append(objects['bbox']['ymax'])

  feature_dict = {
      'image/height':
          dataset_util.int64_feature(image_height),
      'image/width':
          dataset_util.int64_feature(image_width),
      'image/filename':
          dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_util.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256':
          dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded':
          dataset_util.bytes_feature(encoded_jpg),
      'image/format':
          dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin':
          dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax':
          dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin':
          dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax':
          dataset_util.float_list_feature(ymax),
      'image/object/class/label':
          dataset_util.int64_list_feature(category_ids)
  }

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return key, example, num_annotations_skipped


def _create_tf_record_from_coco_annotations(annotations_file, image_dir, output_path, label_map_path):

  with tf.gfile.GFile(annotations_file, 'r') as fid:
    annotated_data = json.load(fid)
    imglist = tf.gfile.Glob('%s/*.jpg' % image_dir)
    imgid_list = [img.strip().split('/')[-1].split('.')[0] for img in imglist]
    tf.logging.info('The image list is as follows')
    tf.logging.info(image_dir)
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)

    tf.logging.info('writing to output path: %s', output_path)

    writer = tf.python_io.TFRecordWriter(output_path)
    total_num_annotations_skipped = 0
    # Write a loop over all images
    img_count = 0
    for image_full_path, image_id in zip(imglist, imgid_list):
      if img_count % 50 == 0:
        tf.logging.info('Done writing %d files' % img_count)
      img_count += 1

      _, tf_example, num_annotations_skipped = create_tf_example(
        image_full_path, annotated_data, image_id, label_map_dict)

      total_num_annotations_skipped += num_annotations_skipped
      writer.write(tf_example.SerializeToString())
    writer.close()

    tf.logging.info('Finished writing, skipped %d annotations.',
                    total_num_annotations_skipped)

def main(_):
  assert FLAGS.train_image_dir, '`train_image_dir` missing.'
  assert FLAGS.val_image_dir, '`val_image_dir` missing.'
  assert FLAGS.test_image_dir, '`test_image_dir` missing.'
  assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
  assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'
  assert FLAGS.test_annotations_file, '`test_annotations_file` missing.'

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  train_output_path = os.path.join(FLAGS.output_dir, 'tt100k_train.record')
  # val_output_path = os.path.join(FLAGS.output_dir, 'tt100k_val.record')
  test_output_path = os.path.join(FLAGS.output_dir, 'tt100k_test.record')

  # _create_tf_record_from_coco_annotations(
  #     FLAGS.train_annotations_file,
  #     FLAGS.train_image_dir,
  #     train_output_path,
  #     FLAGS.label_map_path)
  _create_tf_record_from_coco_annotations(
    FLAGS.test_annotations_file,
    FLAGS.test_image_dir,
    test_output_path,
    FLAGS.label_map_path)

if __name__ == '__main__':
  tf.app.run()
