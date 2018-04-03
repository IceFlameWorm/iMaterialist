import tensorflow as tf
import os
import numpy as np
# Adding Seed so that random initialization is consistent
from numpy.random import seed
import glob
import math
from tensorflow import set_random_seed

from utils import image_dataset

seed(1)

set_random_seed(2)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'  # use gpu 0 to train
IMAGE_PATH = '/media/data2/lzhang_data/dataset/iMaterial/'
batch_size = 64
num_classes = 128
epoc_times = 50
img_size = 256
# img_croped_size = 224
num_channels = 3
num_data_augment_method = 8
num_trainset = num_data_augment_method * len(glob.glob(IMAGE_PATH + 'train2/*.jpg'))

train_images, train_labels = image_dataset.get_batched_train_dataset(batch_size)
valid_images, valid_labels = image_dataset.get_batched_validation_dataset(batch_size)

# dynamic gpu memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
session = tf.Session(config=config)

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

##Network graph params
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 224


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters):
    # We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    # We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    # Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    # We shall be using max-pooling.
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    # Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    # We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    # Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    # Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    # Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


layer_conv1 = create_convolutional_layer(input=x,
                                         num_input_channels=num_channels,
                                         conv_filter_size=filter_size_conv1,
                                         num_filters=num_filters_conv1)
layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                         num_input_channels=num_filters_conv1,
                                         conv_filter_size=filter_size_conv2,
                                         num_filters=num_filters_conv2)

layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                         num_input_channels=num_filters_conv2,
                                         conv_filter_size=filter_size_conv3,
                                         num_filters=num_filters_conv3)

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat,
                            num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=fc_layer_size,
                            use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                            num_inputs=fc_layer_size,
                            num_outputs=num_classes,
                            use_relu=False)

y_pred = tf.nn.softmax(layer_fc2, name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch, acc, val_acc, val_loss))


saver = tf.train.Saver()

iteration_times_each_epoc = int(math.ceil(1.0 * num_trainset / batch_size))


# not a efficient way,it should be more fast with numpy

def get_label_batch_matrix(y_batch):
    m = len(y_batch)
    batch_matrix = np.zeros((m, num_classes), dtype=int)
    for i in range(m):
        batch_matrix[i, y_batch[i] - 1] = 1
    return batch_matrix


def train():
    x_batch, y_true_batch = train_images, train_labels
    # print('shape[train_images]:', np.shape(train_images), np.shape(train_labels))
    x_valid_batch, y_valid_batch = valid_images, valid_labels

    for i in range(epoc_times):
        for j in range(iteration_times_each_epoc):
            x_batch1, y_true_batch1 = session.run([x_batch, y_true_batch])

            x_batch1 = np.reshape(x_batch1, (np.shape(x_batch1)[0] * np.shape(x_batch1)[1], 256, 256, 3))
            y_true_batch1 = np.reshape(y_true_batch1, (np.shape(y_true_batch1)[0] * np.shape(y_true_batch1)[1], 1))
            y_true_batch1 = get_label_batch_matrix(y_true_batch1)
            # print('shape[x_batch]:', np.shape(x_batch1), np.shape(y_true_batch1))
            # print(y_true_batch1)
            x_valid_batch1, y_valid_batch1 = session.run([x_valid_batch, y_valid_batch])
            y_valid_batch1 = get_label_batch_matrix(y_valid_batch1)
            # print('shape[x_valid_batch]:', np.shape(x_valid_batch1), np.shape(y_valid_batch1))
            feed_dict_tr = {x: x_batch1,
                            y_true: y_true_batch1}
            feed_dict_val = {x: x_valid_batch1,
                             y_true: y_valid_batch1}

            session.run(optimizer, feed_dict=feed_dict_tr)
            if j == iteration_times_each_epoc - 1:
                val_loss = session.run(cost, feed_dict=feed_dict_val)

                show_progress(i + 1, feed_dict_tr, feed_dict_val, val_loss)
                saver.save(session, './model/iMaterialist-model')


train()
session.close()
