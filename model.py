import tensorflow as tf
import numpy as np

from ops import *
from logging import getLogger

from tensorflow.examples.tutorials.mnist import input_data

logger = getLogger(__name__)

def inference(inputs, keep_prob, phase_train):
    inputs_image = tf.reshape(inputs, [-1, 28, 28, 1])
    scope = "conv_1"
    conv1 = conv2d(inputs_image, 32, [5, 5], "None",
                   normalizer_fn=batch_norm_layer, 
                   normalizer_params={'phase_train': phase_train, 'scope': scope+'_bn'},
                   activation_fn=tf.nn.relu, weights_regularizer=None, scope=scope)
    scope = "maxpool_1"
    pool1 = maxpool2d(conv1, scope=scope)

    scope = "conv_2"
    conv2 = conv2d(pool1, 64, [5, 5], "None",
                   normalizer_fn=batch_norm_layer, 
                   normalizer_params={'phase_train': phase_train, 'scope': scope+'_bn'},
                   activation_fn=tf.nn.relu, weights_regularizer=None, scope=scope)
    scope = "maxpool_2"
    pool2 = maxpool2d(conv2, scope=scope)    
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    
    scope = "full_connect_1"    
    fc1 = fully_connected(pool2_flat, 7*7*64, 1024,
                          activation_fn=tf.nn.relu, scope=scope)
    fc1_dropped = tf.nn.dropout(fc1, keep_prob)
    scope = "readout"    
    label_pred = fully_connected(fc1_dropped, 1024, 10,
                                 activation_fn=tf.nn.softmax, scope=scope)
    
    return label_pred

def loss(label_pred, label): 
    cross_entropy = tf.reduce_mean( -tf.reduce_sum(label * tf.log(label_pred), reduction_indices=[1]) )
    loss = cross_entropy
    
    return loss

def training(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    
    return train_op

