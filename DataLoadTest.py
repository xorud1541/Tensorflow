import tensorflow as tf
import numpy as np
import re
import matplotlib.pyplot as plt
import sys

TRAINING_FILE = '/home/lhwpc/dataset/training.csv'
VALIDATION_FILE = '/home/lhwpc/dataset/validation.csv'

flags = tf.app.flags
FLAGS = flags.FLAGS
FLAGS.image_size = 256
FLAGS.image_color = 3
FLAGS.maxpool_filter_size = 3
FLAGS.num_classes = 9
FLAGS.batch_size = 100
FLAGS.learning_rate = 0.001
FLAGS.training_epochs = 0
FLAGS.log_dir = '/home/lhwpc/dataset'

def get_input_queue(csv_file_name,num_epochs=None):
    train_images = []
    train_labels = []

    for line in open(csv_file_name,'r'):
        cols = re.split(',|\n', line)
        train_images.append(cols[0])
        train_labels.append(int(cols[2]))

    input_queue = tf.train.slice_input_producer([train_images, train_labels], num_epochs=num_epochs
                                                , shuffle=True)
    return input_queue

def read_data(input_queue):
    image_file = input_queue[0]
    label = input_queue[1]

    image = tf.image.decode_png(tf.read_file(image_file), channels=FLAGS.image_color)
    return image, label

def read_data_batch(csv_file_name, batch_size=FLAGS.batch_size):
    input_queue = get_input_queue(csv_file_name)
    image, label = read_data(input_queue)
    image = tf.reshape(image, [FLAGS.image_size, FLAGS.image_size, FLAGS.image_color])

    batch_image, batch_label = tf.train.batch([image, label], batch_size=batch_size)

    batch_label_on_hot = tf.one_hot(tf.to_int64(batch_label),
                                    FLAGS.num_classes, on_value=1.0, off_value=0.0)
    return batch_image, batch_label_on_hot

def conv1(input_data):
    #layer 1(convolutional layer)
    FLAGS.conv1_filter_size = 5
    FLAGS.conv1_layer_size = 32
    FLAGS.stride1 = 3

    with tf.name_scope('conv_1'):
        W_conv1 = tf.Variable(tf.random_normal(
            [FLAGS.conv1_filter_size, FLAGS.conv1_filter_size, FLAGS.image_color, FLAGS.conv1_layer_size], stddev=0.1
        ))

        h_conv1 = tf.nn.conv2d(input_data, W_conv1, strides=[1, 3, 3, 1], padding='SAME')
        print(h_conv1.get_shape())
        h_conv1_relu = tf.nn.relu(h_conv1)
        print(h_conv1_relu.get_shape())
        h_conv1_maxpool = tf.nn.max_pool(h_conv1_relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(h_conv1_maxpool.get_shape())

    return h_conv1_maxpool

def conv2(input_data):
    # layer 2(convolutional layer)
    FLAGS.conv2_filter_size = 5
    FLAGS.conv2_layer_size = 128
    FLAGS.stride2 = 3

    with tf.name_scope('conv_2'):
        W_conv2 = tf.Variable(tf.truncated_normal(
            [FLAGS.conv2_filter_size, FLAGS.conv2_filter_size, FLAGS.conv1_layer_size, FLAGS.conv2_layer_size], stddev=0.1
        ))

        h_conv2 = tf.nn.conv2d(input_data, W_conv2, strides=[1, 3, 3, 1], padding='SAME')
        print(h_conv2.get_shape())
        h_conv2_relu = tf.nn.relu(h_conv2)
        print(h_conv2_relu.get_shape())
        h_conv2_maxpool = tf.nn.max_pool(h_conv2_relu, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        print(h_conv2_maxpool.get_shape())

    return h_conv2_maxpool

def conv3(input_data):
    # layer 3(convolutional layer)
    FLAGS.conv3_filter_size = 5
    FLAGS.conv3_layer_size = 192
    FLAGS.stride3 = 2

    with tf.name_scope('conv_3'):
        W_conv3 = tf.Variable(tf.truncated_normal(
            [FLAGS.conv3_filter_size, FLAGS.conv3_filter_size, FLAGS.conv2_layer_size, FLAGS.conv3_layer_size], stddev=0.1
        ))

        h_conv3 = tf.nn.conv2d(input_data, W_conv3, strides=[1, 3, 3, 1], padding='SAME')
        print(h_conv3.get_shape())
        h_conv3_relu = tf.nn.relu(h_conv3)
        print(h_conv3_relu.get_shape())
        h_conv3_maxpool = tf.nn.max_pool(h_conv3_relu, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        print(h_conv3_maxpool.get_shape())

    return h_conv3_maxpool

def conv4(input_data):
    # layer 4(convolutional layer)
    FLAGS.conv4_filter_size = 2
    FLAGS.conv4_layer_size = 256
    FLAGS.stride4 = 2

    with tf.name_scope('conv_4'):
        W_conv4 = tf.Variable(tf.truncated_normal(
            [FLAGS.conv4_filter_size, FLAGS.conv4_filter_size, FLAGS.conv3_layer_size, FLAGS.conv4_layer_size], stddev=0.1
        ))

        h_conv4 = tf.nn.conv2d(input_data, W_conv4, strides=[1, 3, 3, 1], padding='SAME')
        print(h_conv4.get_shape())
        h_conv4_relu = tf.nn.relu(h_conv4)
        print(h_conv4_relu.get_shape())
        h_conv4_maxpool = tf.nn.max_pool(h_conv4_relu, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        print(h_conv4_maxpool.get_shape())

    return h_conv4_maxpool

def fc1(input_data):
    input_layer_size = 2*2*FLAGS.conv4_layer_size
    FLAGS.fc1_layer_size = 256

    with tf.name_scope('fc_1'):
        input_data_reshape = tf.reshape(input_data, [-1, input_layer_size])
        W_fc1 = tf.Variable(tf.truncated_normal([input_layer_size, FLAGS.fc1_layer_size], stddev=0.1))
        b_fc1 = tf.Variable(tf.truncated_normal([FLAGS.fc1_layer_size], stddev=0.1))
        h_fc1 = tf.add(tf.matmul(input_data_reshape, W_fc1), b_fc1)
        h_fc1_relu = tf.nn.relu(h_fc1)
    return h_fc1_relu

def final_out(input_data):
    with tf.name_scope('final_out'):
        W_fo = tf.Variable(tf.truncated_normal([FLAGS.fc1_layer_size, FLAGS.num_classes], stddev=0.1))
        b_fo = tf.Variable(tf.truncated_normal([FLAGS.num_classes], stddev=0.1))
        h_fo = tf.add(tf.matmul(input_data, W_fo), b_fo)
    return h_fo

def build_layer(images, keep_prob):
    r_cnn1 = conv1(images)
    print("shape after cnn1 : ", r_cnn1.get_shape())
    r_cnn2 = conv2(r_cnn1)
    print("shape after cnn2 : ", r_cnn2.get_shape())
    r_cnn3 = conv3(r_cnn2)
    print("shape after cnn3 : ", r_cnn3.get_shape())
    r_cnn4 = conv4(r_cnn3)
    print("shape after cnn4 : ", r_cnn4.get_shape())
    r_fc1 = fc1(r_cnn4)
    print("shape after fc1 : ", r_fc1.get_shape())
    r_dropout = tf.nn.dropout(r_fc1, keep_prob)
    print("shape after dropout : ", r_dropout.get_shape())
    r_out = final_out(r_dropout)
    print("shape after final layer : ", r_out.get_shape())

    return r_out

def main(argv = None):
    images = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, FLAGS.image_color])
    labels = tf.placeholder(tf.int32, [None, FLAGS.num_classes])
    image_batch, label_batch = read_data_batch(TRAINING_FILE)

    keep_prob = tf.placeholder(tf.float32) #dropout ratio
    prediction = build_layer(images, keep_prob)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    train = optimizer.minimize(loss)

    validate_image_batch, validate_label_batch = read_data_batch(VALIDATION_FILE)

    label_max = tf.argmax(labels, 1)
    pre_max = tf.argmax(prediction, 1)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    summary = tf.summary.merge_all()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        init_op = tf.global_variables_initializer()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(init_op)

        for i in range(103):
            images_, labels_ = sess.run([image_batch, label_batch])
            sess.run(train, feed_dict={images:images_, labels:labels_, keep_prob:0.7})

            if(i%10 == 0):
                #print out training status
                rt = sess.run([label_max, pre_max, loss, accuracy], feed_dict={images:images_,
                                                                             labels:labels_,
                                                                               keep_prob:1.0})
                print('Prediction loss:', rt[2], 'accuracy:', rt[3])
                #validation steps
                validate_images_, validate_labels_ = sess.run([validate_image_batch, validate_label_batch])
                rv = sess.run([label_max, pre_max, loss, accuracy], feed_dict={images: validate_images_
                    , labels: validate_labels_, keep_prob:1.0})
                print('Validation loss:', rv[2], ' accuracy:', rv[3])


        coord.request_stop()
        coord.join(threads)
        print("finished")
main()