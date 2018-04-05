
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim
#加载通过TensorFlow-Slim定义好的inception_v3模型
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3


#处理好之后的图片数据
input_data = '../flower_processed_data.npy'
#保存训练好的模型
train_file = '../save_model'
#谷歌提供的训练好的模型文件
ckpt_file = '../inception_v3.ckpt'

#定义训练使用的参数
learning_rate = 0.01
steps = 300   #size/batch
batch = 64
n_classes = 128

#不需要从谷歌训练好的模型中加载的参数（即需要重新训练的参数）
checkpoint_exclude_scopes = 'InceptionV3/Logits,InceptionV3/AuxLogits,Mixed_7c,Mixed_7b,Mixed_7a,Mixed_6e,Mixed_6d,Mixed_6c,Mixed_6b,Mixed_6a'
#需要训练的网络层参数名称
#这里给出的是参数的前缀
trainable_scope = 'InceptionV3/Logits,InceptionV3/AuxLogits,Mixed_7c,Mixed_7b,Mixed_7a,Mixed_6e,Mixed_6d,Mixed_6c,Mixed_6b,Mixed_6a'


#获取所有需要从google训练好的模型中加载参数
def get_tuned_variables():
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    
    variable_to_restore = []
    #枚举inception-v3模型中所有的参数，然后判断是否需要从加载列表中移除
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variable_to_restore.append(var)
    return variable_to_restore

#获取所有需要训练的变量列表
def get_trainable_variables():
    scopes = [scope.strip() for scope in trainable_scope.split(',')]
    variables_to_train = []
    #枚举所有需要的训练参数前缀，并通过这些前缀找到所有的参数
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)  #获取所有可训练的的variables
        variables_to_train.extend(variables)
    return variables_to_train

def main():
    #加载预处理好的数据
    processed_data = np.load(input_data)
    train_images = processed_data[0]
    n_training_examples = len(train_images)
    train_labels = processed_data[1]
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    test_images = processed_data[4]
    test_labels = processed_data[5]
    print('%d training examples, %d validation examples and %d testing examples.' %(
    n_training_examples, len(validation_labels), len(test_labels)))
    
    #定义inception-v3 的输入，images为输入图片，labels为每一张图片对应的标签
    images = tf.placeholder(tf.float32, [None, 299, 299, 3],name='input_images')
    labels = tf.placeholder(tf.int64, [None],name='labels')
    
    #定义inception-v3模型。因为谷歌给出的模型只有参数值，所以需要定义模型结构。 
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images,num_classes=n_classes)
    
    #获取需要训练的变量
    trainable_variables = get_trainable_variables()
    #定义交叉熵损失。注意在模型定义的时候已经将正则化损失加入损失集合了
    tf.losses.softmax_cross_entropy(tf.one_hot(labels,n_classes),logits, weights=1)
    #定义训练过程。这里minimize的过程中指定了需要优化的变量集合
    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(tf.losses.get_total_loss())
    
    #计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits,1),labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))   #tf.cast转换成float32类型
    
    #定义加载模型的函数
    load_fn = slim.assign_from_checkpoint_fn(ckpt_file,get_tuned_variables(),ignore_missing_vars=True)
    
    #定义保存新的训练的好的模型
    saver = tf.train.Saver()    #声明tf.train.Saver类用于保存模型
    with tf.Session() as sess:
        #初始化没有加载进来的变量。
        #这个过程一定要在模型加载之前，否则初始化过程会将已经加载好的变量重新赋值
        init = tf.global_variables_initializer()
        sess.run(init)
        
        #加载谷歌已经训练好的模型
        print('Loaging tuned variable from %s' % ckpt_file)
        load_fn(sess)
        
        start = 0
        end = batch
        for i in range(steps):
            #运行训练过程，这里不会更新全部的参数，只会更新制定的部分参数
            sess.run(train_step, feed_dict = {
                images: train_images[start:end],
                labels: train_labels[start:end]
            })
            
            #输出日志
            if i % 30 == 0 or i + 1 == steps:
                saver.save(sess,train_file,global_step=i)  
                validation_accuracy = sess.run(evaluation_step,feed_dict={
                    images: validation_images, labels: validation_labels
                })
                print('Step %d: Validation accuracy = %.1f%%' %(i,validation_accuracy*100.0))
                
            #因为数据预处理的时候已经做过了打乱数据的操作，所以这里只需要顺序使用训练集就好
            start = end
            if start == n_training_examples:
                start = 0
            
            end = start + batch
            if end > n_training_examples:
                end = n_training_examples
                
        #最后在测试集数据上使用正确率
        test_accuracy = sess.run(evaluation_step,feed_dict={
            images: test_images, labels: test_labels
        })
        print('Final test accuracy = %.1f%%' %(test_accuracy * 100))
    
if __name__ == '__main__':
    main()

