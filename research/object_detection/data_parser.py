# import tensorflow as tf
import os
import sys
import glob
import numpy as np
import json
import shutil


def train_data_generator(annotated_data, train_dir):
  imglist = glob.glob('%s*.jpg' % train_dir)
  imgid_list = [img.strip().split('/')[-1].split('.')[0] for img in imglist]
  # print len(imglist)
  # print imglist
  # print imgid_list
  return imglist, imgid_list


# if __name__ == 'main':
data_dir = '/media/mcao/Miguel/TT100k/data/'
annotations_file = data_dir + 'annotations.json'
train_dir = data_dir + 'test/'
train_output_dir = data_dir + 'test_out/'
if not os.path.exists(train_output_dir ):
  os.makedirs(train_output_dir)

with open(annotations_file, 'r') as f:
  annotated_data = json.load(f)

imglist, imgid_list = train_data_generator(annotations_file, train_dir)
# imgid_list = imgid_list[0:6]

count = 0
for image_name, image in zip(imglist, imgid_list):
  object_list = annotated_data['imgs'][image]['objects']
  for objects in object_list:
    category = objects['category']
    flag_one_object = True
    if category in ['pn', 'pne'] and flag_one_object:
      xmin = objects['bbox']['xmin']
      xmax = objects['bbox']['xmax']
      ymin = objects['bbox']['ymin']
      ymax = objects['bbox']['ymax']
      count += 1
      flag_one_object = False
      shutil.copy(image_name, train_output_dir)
      print image, category, xmin, xmax, ymin, ymax


print count

# print annotated_data['imgs'][imgid_list[0]].keys()
# print 'Number of objects - ', len(annotated_data['imgs'][imgid_list[0]]['objects'])
# print annotated_data['imgs'][imgid_list[0]]['objects'][0]
# print annotated_data['imgs'][imgid_list[0]]['objects'][0].keys()
# print annotated_data['imgs'][imgid_list[0]]['objects'][0]['category']
# print annotated_data['imgs'][imgid_list[0]]['objects'][1]['category']
# a = 10
