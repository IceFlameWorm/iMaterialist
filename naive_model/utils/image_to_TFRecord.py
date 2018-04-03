'''
transform image data to TFRecord format
'''

import tensorflow as tf
from PIL import Image
import numpy as np
import glob
import re
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # use gpu 0 to train


def __trans_image_data(dataset_name, image_path):
    tfrecords_path = TF_RECORDS_PATH + 'iMaterialist_' + dataset_name + '.tfrecords'
    print('tfrecords_path:', tfrecords_path)
    writer = tf.python_io.TFRecordWriter(tfrecords_path)
    imagefile_paths = glob.glob(image_path + '/' + dataset_name + '2/*.jpg')
    for index, imgpath in enumerate(imagefile_paths):
        # print(imgpath)
        if index % 1000 == 0:
            print('........index:%d........' % index)
        try:
            img = Image.open(imgpath)
            img = img.resize((image_size, image_size))
            img_raw = img.tobytes()
            label = int(re.findall(r'\d+', imgpath).pop(-1))
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
        except ValueError as VE:
            if str(VE) == 'Decompressed Data Too Large':
                print('some picture has big metadata %s' % image_path)
        except IOError:
            print('IOError', image_path)
    writer.close()


def trans_validation_image_data(image_path):
    __trans_image_data(dataset_name='validation', image_path=image_path)


def trans_train_image_data(image_path):
    __trans_image_data(dataset_name='train', image_path=image_path)


def __save_img():
    filename_queue = tf.train.string_input_producer(["iMaterialist_train.tfrecords"])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # return file and file_name
    print(serialized_example)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [image_size, image_size, 3])
    label = tf.cast(features['label'], tf.int64)
    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(5):
            example, l = sess.run([image, label])  # take out image and label
            print(np.shape(example))
            img = Image.fromarray(example, 'RGB')
            img.save(str(i) + '_''Label_' + str(l) + '.jpg')  # save image
            print(example, l)
        coord.request_stop()
        coord.join(threads)


ROOT_PATH = '/media/data2/lzhang_data/dataset/iMaterial/'
TF_RECORDS_PATH = ROOT_PATH + 'tfrecords/'
image_size = 256
# trans_validation_image_data(ROOT_PATH )
trans_train_image_data(ROOT_PATH)
# __save_img()
