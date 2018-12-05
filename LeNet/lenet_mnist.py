import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 注意这里是相对路径
mnist = input_data.read_data_sets("../../data/", one_hot=True)

# 训练数据
x = tf.placeholder("float", shape=[None, 784])
# 训练标签数据
y_ = tf.placeholder("float", shape=[None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层：卷积层
# 过滤器大小为5*5，当前深度为1，过滤器深度为32
conv1_weight = tf.get_variable("conv1_weights", [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.1))
conv1_biases = tf.get_variable("conv1_biases", [32], initializer=tf.constant_initializer(0.0))
# 移动步长为1，使用全0填充 strides 表示卷积移动步长，前后必须为1，中间的我们可以修改
# padding 中 SAME表示补零，而VALID 表示把后面的去掉
conv1 = tf.nn.conv2d(x_image, conv1_weight, strides=[1, 1, 1, 1], padding="SAME")
# 激活函数Relu
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

#  第二层： 最大池化层
#  池化层过滤器的大小为2*2，移动步长尾2，全0填充
pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# 第三层：卷积层
conv2_weights = tf.get_variable("conv2_weights", [5, 5, 32, 64],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
conv2_biases = tf.get_variable("conv2_biases", [64], initializer=tf.constant_initializer(0.0))
conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

# 第四层：最大池化层
pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# 第五层：全链接层
fc1_weights = tf.get_variable("fc1_weights", [7 * 7 * 64, 1024],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
fc1_baises = tf.get_variable("fc1_baises", [1024], initializer=tf.constant_initializer(0.1))
pool2_vector = tf.reshape(pool2, [-1, 7 * 7 * 64])
fc1 = tf.nn.relu(tf.matmul(pool2_vector, fc1_weights) + fc1_baises)

# 为了减少过分拟合，加入Dropout层
keep_prob = tf.placeholder(tf.float32)
fc1_dropout = tf.nn.dropout(fc1, keep_prob)

# 第六层：全连接层
# 神经元结点1024 分类节点10
fc2_weights = tf.get_variable("fc2_weights", [1024, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
fc2_biases = tf.get_variable("fc2_biases", [10], initializer=tf.constant_initializer(0.1))
fc2 = tf.matmul(fc1_dropout, fc2_weights) + fc2_biases

# 第7层
# softmax
y_conv = tf.nn.softmax(fc2)


# 定义交叉熵损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

# 选择优化器，并让优化器最小化损失函数/收敛，反向传播
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# tf.argmax()返回的是某一维度上数据所在的索引值，这里即代表预测值和真实值
# 判断预测值y和真实值y_中最大数的索引是否一致，y的值为1-10概率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

# 用平均值来统计测试准确率
#tf.cast 用来做类型转换
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 开始训练
# 使用tf.InteractiveSession()来构建会话的时候，我们可以先构建一个session然后再定义操作
# 使用tf.Session()，我们需要在会话构建之前定义好全部的操作（operation）然后再构建会话
with tf.Session() as sess:
    with tf.device("/gpu:2"):
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            batch = mnist.train.next_batch(100)
            if i % 100 == 0:
                # 评估阶段不适用Dropout
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("step%d, training accuracy %g" % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  # 训练阶段使用50%的Droupout

        # 在测试数据上测试准确率
        print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
