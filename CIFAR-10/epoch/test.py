import tensorflow as tf

# 建立一个Session
with tf.Session() as sess:
    # 要读3张图片A.jpg,B.jpg,C.jpg
    filename = ['A.JPG', 'B.JPG', 'C.JPG']
    # string_input_producer 会产生一个文件名队列
    filename_queue = tf.train.string_input_producer(filename, shuffle=False, num_epochs=5)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    tf.local_variables_initializer().run()
    threads = tf.train.start_queue_runners(sess=sess)
    i = 0
    while True:
        i += 1
        image_data = sess.run(value)
        with open("read/test_%d.JPG" % i, "wb") as f:
            f.write(image_data)
