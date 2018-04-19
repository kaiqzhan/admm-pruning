import tensorflow as tf
import numpy as np

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape,name=''):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name=name)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


class Lenet_300_100():
  def __init__(self):
    self.x = tf.placeholder(tf.float32, [None, 784],name='x')
    self.y_ = tf.placeholder(tf.float32, [None, 10],name='y')
    with tf.name_scope('reshape'):
        x_image = tf.reshape(self.x, [-1, 784])
    with tf.name_scope('fc1'):
      self.W_fc1 = weight_variable([784, 300], 'W_fc1')
      b_fc1 = bias_variable([300])
      h_fc1 = tf.nn.relu(tf.matmul(x_image, self.W_fc1) + b_fc1)
    with tf.name_scope('dropout'):
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
    with tf.name_scope('fc2'):
        self.W_fc2 = weight_variable([300, 100], 'W_fc2')
        b_fc2 = bias_variable([100])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, self.W_fc2) + b_fc2)
    with tf.name_scope('fc3'):
        self.W_fc3 = weight_variable([100, 10], 'W_fc3')
        b_fc3 = bias_variable([10])
        self.logits = tf.matmul(h_fc2, self.W_fc3) + b_fc3
    with tf.name_scope('loss'):
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                                logits=self.logits)
    self.cross_entropy = tf.reduce_mean(cross_entropy)
    self.layers = {}
    self.layers['fc1/W_fc1'] = self.W_fc1
    self.layers['fc2/W_fc2'] = self.W_fc2
    self.layers['fc3/W_fc3'] = self.W_fc3
  def get_layers(self):
    return self.layers


def get_lenet_300_100():
  return Lenet_300_100()
