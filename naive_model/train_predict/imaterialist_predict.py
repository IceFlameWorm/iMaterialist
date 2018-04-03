import tensorflow as tf
import numpy as np
import os
from PIL import Image
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # use gpu 0 to train

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
sess = tf.Session(config=config)
saver = tf.train.import_meta_graph('./model/iMaterialist-model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./model/'))

graph = tf.get_default_graph()
y_pred = graph.get_tensor_by_name("y_pred:0")
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, 128))

dir_path = '/media/data2/lzhang_data/dataset/iMaterial/test2/'
image_paths = range(12800)

image_size = 256
num_channels = 3

with open('predict_result.csv', 'a+') as predict_result:
    writer = csv.writer(predict_result)
    writer.writerows([('id', 'predicted')])
    for path in image_paths:
        abso_path = dir_path + str(path + 1) + '.jpg'
        images = []
        try:
            img = Image.open(abso_path)
            img = img.resize((image_size, image_size))
            images.append(np.asarray(img, dtype=np.uint8))
            images = np.array(images, dtype=np.uint8)
            x_batch = images.reshape(1, image_size, image_size, num_channels)

            feed_dict_testing = {x: x_batch, y_true: y_test_images}
            result = sess.run(y_pred, feed_dict=feed_dict_testing)
            print('image_index:', path, 'predicted:', np.argmax(result) + 1)
            writer.writerows([(path + 1, np.argmax(result) + 1)])
        except IOError:
            # if some images don't exists, predict with a random value between 1-128
            writer.writerows([(path + 1, np.random.randint(1, 129))])
predict_result.close()
sess.close()
