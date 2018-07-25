# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
from model import create_model
from solver import create_admm_solver
from tensorflow.examples.tutorials.mnist import input_data
from prune_utility import apply_prune_on_grads,apply_prune,get_configuration,projection
import tensorflow as tf
import numpy as np
from numpy import linalg as LA

FLAGS = None
# pruning ratio


prune_configuration = get_configuration()
dense_w = {}
P1 = prune_configuration.P1
P2 = prune_configuration.P2
P3 = prune_configuration.P3
P4 = prune_configuration.P4

prune_configuration.display()


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  model = create_model()
  x = model.x
  y_ = model.y_
  cross_entropy = model.cross_entropy
  layers = model.layers
  logits = model.logits
  solver = create_admm_solver(model)
  keep_prob = model.keep_prob
  train_step = solver.train_step
  train_step1 = solver.train_step1
  
  W_conv1 = model.W_conv1
  W_conv2 = model.W_conv2
  W_fc1 = model.W_fc1
  W_fc2 = model.W_fc2
  
  A = solver.A
  B = solver.B
  C = solver.C
  D = solver.D
  E = solver.E
  F = solver.F
  G = solver.G
  H = solver.H

  my_trainer = tf.train.AdamOptimizer(1e-3)
  grads = my_trainer.compute_gradients(cross_entropy)
    
  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    Z1 = sess.run(W_conv1)
    Z1 = projection(Z1, percent=P1)

    U1 = np.zeros_like(Z1)

    Z2 = sess.run(W_conv2)
    Z2 = projection(Z2, percent=P2)

    U2 = np.zeros_like(Z2)

    Z3 = sess.run(W_fc1)
    Z3 = projection(Z3, percent=P3)

    U3 = np.zeros_like(Z3)

    Z4 = sess.run(W_fc2)
    Z4 = projection(Z4, percent=P4)

    U4 = np.zeros_like(Z4)

    for j in range(30):
        for i in range(5000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step1.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0, A:Z1, B:U1, C:Z2, D:U2, E:Z3, F:U3, G:Z4, H:U4})
        Z1 = sess.run(W_conv1) + U1
        Z1 = projection(Z1, percent=P1)

        U1 = U1 + sess.run(W_conv1) - Z1

        Z2 = sess.run(W_conv2) + U2
        Z2 = projection(Z2, percent=P2)

        U2 = U2 + sess.run(W_conv2) - Z2

        Z3 = sess.run(W_fc1) + U3
        Z3 = projection(Z3, percent=P3)

        U3 = U3 + sess.run(W_fc1) - Z3

        Z4 = sess.run(W_fc2) + U4
        Z4 = projection(Z4, percent=P4)

        U4 = U4 + sess.run(W_fc2) - Z4

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        print(LA.norm(sess.run(W_conv1) - Z1))
        print(LA.norm(sess.run(W_conv2) - Z2))
        print(LA.norm(sess.run(W_fc1) - Z3))
        print(LA.norm(sess.run(W_fc2) - Z4))

    dense_w['conv1/W_conv1'] = W_conv1
    dense_w['conv2/W_conv2'] = W_conv2
    dense_w['fc1/W_fc1'] = W_fc1
    dense_w['fc2/W_fc2'] = W_fc2
    
    dict_nzidx = apply_prune(dense_w,sess)
    print ("checking space dictionary")
    print (dict_nzidx.keys())
    grads = apply_prune_on_grads(grads,dict_nzidx)
    apply_gradient_op = my_trainer.apply_gradients(grads)
    for var in tf.global_variables():
                if tf.is_variable_initialized(var).eval() == False:
                    sess.run(tf.variables_initializer([var]))
    print ("start retraining after pruning")
    for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))

      apply_gradient_op.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print('test accuracy %g' % accuracy.eval(feed_dict={
          x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    print(np.sum(sess.run(W_conv1)!=0))
    print(np.sum(sess.run(W_conv2) != 0))
    print(np.sum(sess.run(W_fc1) != 0))
    print(np.sum(sess.run(W_fc2) != 0))
    # do the saving.
    saver = tf.train.Saver()
    saver.save(sess,"./lenet_5_pruned_model.ckpt")
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  
