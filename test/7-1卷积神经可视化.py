
# coding: utf-8

# # 导包

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5) 


# # 导入数据

# In[2]:


mnist = input_data.read_data_sets(
    "G:\Pycharmworkspace\jupyter\MNIST_data", one_hot=True)
DIR = 'G:/Pycharmworkspace/jupyter/Test/'
#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size


# # 创建神经网络

# In[3]:


#参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)  #平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)  #标准差
        tf.summary.scalar('max', tf.reduce_max(var))  #最大值
        tf.summary.scalar('min', tf.reduce_min(var))  #最小值
        tf.summary.histogram('histogram', var)  #直方图


#初始化权值
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)  #生成一个截断的正态分布
    return tf.Variable(initial, name=name)


#初始化偏置
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


#卷积层
def conv2d(x, W):
    #x input tensor of shape `[batch, in_height, in_width, in_channels]`
    #W filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    #`strides[0] = strides[3] = 1`. strides[1]代表x方向的步长，strides[2]代表y方向的步长
    #padding: A `string` from: `"SAME", "VALID"`
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


#池化层
def max_pool_2x2(x):
    #ksize [1,x,y,1]
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#命名空间
with tf.name_scope('input'):
    #定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')
    with tf.name_scope('x_image'):
        #改变x的格式转为4D的向量[batch, in_height, in_width, in_channels]`
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')

with tf.name_scope('Conv1'):
    #初始化第一个卷积层的权值和偏置
    with tf.name_scope('W_conv1'):
        W_conv1 = weight_variable(
            [5, 5, 1, 20], name='W_conv1')  #5*5的采样窗口，20个卷积核从1个平面抽取特征
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([20], name='b_conv1')  #每一个卷积核一个偏置值

    #把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_1'):
        conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)  #进行max-pooling
        
with tf.name_scope('fc1'):
    #初始化第一个全连接层的权值
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable(
            [14*14*20, 100], name='W_fc1')  #上一场有14*14*20个神经元，全连接层有100个神经元
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([100], name='b_fc1')  #100个节点

    #把池化层2的输出扁平化为1维
    with tf.name_scope('h_pool1_flat'):
        h_pool1_flat = tf.reshape(
            h_pool1, [-1, 14*14*20], name='h_pool1_flat')
    #求第一个全连接层的输出
    with tf.name_scope('wx_plus_b1'):
        wx_plus_b1 = tf.matmul(h_pool1_flat, W_fc1) + b_fc1
    with tf.name_scope('softmax'):
        #计算输出
        prediction = tf.nn.softmax(wx_plus_b1)
        


# # 代价函数

# In[4]:


#交叉熵代价函数
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),
        name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

#使用AdamOptimizer进行优化
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#求准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        #结果存放在一个布尔列表中
        correct_prediction = tf.equal(
            tf.argmax(prediction, 1), tf.argmax(y, 1))  #argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        #求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

#合并所有的summary
merged = tf.summary.merge_all()


# # 训练

# In[ ]:


with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(DIR + 'logs/train', sess.graph)
    test_writer = tf.summary.FileWriter(DIR + 'logs/test', sess.graph)
    for i in range(1001):
        #训练模型
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        print(batch_xs.shape,batch_ys.shape)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        #记录训练集计算的参数
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys})
        train_writer.add_summary(summary, i)
        #记录测试集计算的参数
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys})
        test_writer.add_summary(summary, i)

        if i % 100 == 0:
            test_acc = sess.run(
                accuracy,
                feed_dict={
                    x: mnist.test.images,
                    y: mnist.test.labels
                })
            train_acc = sess.run(
                accuracy,
                feed_dict={
                    x: mnist.train.images[:10000],
                    y: mnist.train.labels[:10000]
                })
            print("Iter " + str(i) + ", Testing Accuracy= " + str(test_acc) +
                  ", Training Accuracy= " + str(train_acc))

