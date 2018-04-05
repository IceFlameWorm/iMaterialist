## **环境**
     1.python3.5
	 2.tensor-flow-gpu 1.4.1
	 3.Ubuntu 16.04.3 LTS


----------


 ### **1**：把png格式转成jpg
 naive_model/utils/PNG2JPG.py
 ### **2**：制作TFRecords 文件
 naive_model/utils/image_to_TFRecord.py
### **3**：train
naive_model/train_predict/imaterialist_train.py
### **4**：predict
naive_model/train_predict/imaterialist_predict.py

## **迁移学习**
训练好的Inception-v3
code:

wget: http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz

tar xzf inception_v3_2016_08_28.tar.gz #解压之后可以得到训练好的模型文件

更多训练好的模型可以在 http://github.com/tensorflow/models/tree/master/research/slim

inceptionv3 code：
https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py

tensorflow之inception_v3模型的部分加载及权重的部分恢复
https://blog.csdn.net/u014038273/article/details/78603800

《Tensorflow：实战Google深度学习框架》 才云科技Caicloud,郑泽宇,顾思宇 第六章迁移部分


因为里面用到了slim模块

github
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim

中文
https://blog.csdn.net/guvcolie/article/details/77686555
