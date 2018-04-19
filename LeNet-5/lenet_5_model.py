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

class Lenet_5():
  def __init__(self):  
    self.x = tf.placeholder(tf.float32, [None, 784],name='x')
    # Define loss and optimizer                                                   
    self.y_ = tf.placeholder(tf.float32, [None, 10],name='y')
    with tf.name_scope('reshape'):
      self.x_image = tf.reshape(self.x, [-1, 28, 28, 1],name='x_image')
    with tf.name_scope('conv1'):
      self.W_conv1 = weight_variable([5, 5, 1, 20],'W_conv1')
      b_conv1 = bias_variable([20])
      h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + b_conv1)
    with tf.name_scope('pool1'):
      h_pool1 = max_pool_2x2(h_conv1)
    with tf.name_scope('conv2'):
      self.W_conv2 = weight_variable([5, 5, 20, 50],'W_conv2')
      b_conv2 = bias_variable([50])
      h_conv2 = tf.nn.relu(conv2d(h_pool1, self.W_conv2) + b_conv2)
    with tf.name_scope('pool2'):
      h_pool2 = max_pool_2x2(h_conv2)
    with tf.name_scope('fc1'):
      self.W_fc1 = weight_variable([4 * 4 * 50, 500],'W_fc1')
      b_fc1 = bias_variable([500])
      h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*50])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + b_fc1)
    with tf.name_scope('dropout'):
      self.keep_prob = tf.placeholder(tf.float32)
      h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
    with tf.name_scope('fc2'):
      self.W_fc2 = weight_variable([500, 10],'W_fc2')
      b_fc2 = bias_variable([10])
    self.logits = tf.add(tf.matmul(h_fc1_drop, self.W_fc2),b_fc2,name='logits')
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                                logits=self.logits)
    self.cross_entropy = tf.reduce_mean(cross_entropy,name='entropy_loss')
    self.layers = {}
    self.layers['conv1'] = self.W_conv1
    self.layers['conv2'] = self.W_conv2
    self.layers['fc1'] = self.W_fc1
    self.layers['fc2'] = self.W_fc2
  def get_layers(self):
    return self.layers

def get_lenet_5():
  return Lenet_5()

