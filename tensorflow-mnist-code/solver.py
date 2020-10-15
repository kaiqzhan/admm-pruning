import tensorflow as tf
import numpy as np


class AdmmSolver():
  def __init__(self,model):    
    A=self.A = tf.placeholder(tf.float32, shape = [5, 5, 1, 20])
    B=self.B = tf.placeholder(tf.float32, shape = [5, 5, 1, 20])
    C=self.C = tf.placeholder(tf.float32, shape = [5, 5, 20, 50])
    D=self.D = tf.placeholder(tf.float32, shape = [5, 5, 20, 50])
    E=self.E = tf.placeholder(tf.float32, shape = [4 * 4 * 50, 500])
    F=self.F = tf.placeholder(tf.float32, shape = [4 * 4 * 50, 500])
    G=self.G = tf.placeholder(tf.float32, shape = [500, 10])
    H=self.H = tf.placeholder(tf.float32, shape = [500, 10])

    W_conv1 = model.W_conv1
    W_conv2 = model.W_conv2
    W_fc1 = model.W_fc1
    W_fc2 = model.W_fc2
    cross_entropy = model.cross_entropy
    with tf.name_scope('adam_optimizer'):
        train_loss = cross_entropy+0.00005*(tf.nn.l2_loss(W_conv1)+tf.nn.l2_loss(W_conv2)+tf.nn.l2_loss(W_fc1)+tf.nn.l2_loss(W_fc2))
        admm_loss  = 0.0001*(tf.nn.l2_loss(W_conv1-A+B)+ tf.nn.l2_loss(W_conv2-C+D)+tf.nn.l2_loss(W_fc1-E+F)+tf.nn.l2_loss(W_fc2-G+H))
        train_step = tf.train.AdamOptimizer(1e-3).minimize(train_loss)
        train_step1 = tf.train.AdamOptimizer(1e-3).minimize(train_loss + admm_loss)
    self.train_step = train_step
    self.train_step1 = train_step1
    self.train_loss = train_loss
    self.admm_loss = admm_loss

def create_admm_solver(model):
  return AdmmSolver(model)


