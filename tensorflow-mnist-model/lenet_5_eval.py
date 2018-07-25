from __future__ import division
import numpy as np
import tensorflow as tf
from  lenet_5_model import *
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

sess = tf.InteractiveSession()
model_type = 'lenet_5'
path = '%s/%s_pruned_model'%(model_type,model_type)
model = get_lenet_5()
layers = model.get_layers()

saver = tf.train.Saver()

cross_entropy = model.cross_entropy
logits = model.logits


x = model.x
y_ = model.y_
logits = model.logits
keep_prob = model.keep_prob

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

saver.restore(sess,'%s.ckpt'%path)
print('test accuracy %g' % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

original = []
compressed = []
for name,weight in layers.items():
    weight = sess.run(weight)
    print ("in weight %s "  % name)
    nonzero = np.sum(weight!=0)
    zero = np.sum(weight==0)
    original.append(float(nonzero+zero))
    compressed.append(float(nonzero))
    print ("pruning ratio is %f" %((zero)*1.0/(zero+nonzero)))
    

print ("total compression ratio is %f" % (sum(original)/sum(compressed)))
