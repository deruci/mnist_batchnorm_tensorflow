from __future__ import division

import logging
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib.layers import variance_scaling_initializer
import numpy as np

WEIGHT_INITIALIZER = tf.contrib.layers.xavier_initializer()
logger = logging.getLogger(__name__)

def get_shape(layer):
    return layer.get_shape().as_list()

def batch_norm_layer(inputs, phase_train, scope=None):
    return tf.cond(phase_train,  
                   lambda: tf.contrib.layers.python.layers.batch_norm(inputs, is_training=True, scale=True, 
                                                                      updates_collections=None, scope=scope),  
                   lambda: tf.contrib.layers.python.layers.batch_norm(inputs, is_training=False, scale=True,
                                                                      updates_collections=None, scope=scope, reuse = True)) 

def conv2d(
    inputs,
    num_outputs,
    kernel_shape, # [kernel_height, kernel_width]
    mask_type, # None, "A" or "B",
    strides=[1, 1], # [column_wise_stride, row_wise_stride]
    padding="SAME",
    activation_fn=None,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=WEIGHT_INITIALIZER,
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer,
    biases_regularizer=None,
    scope="conv2d"):
    with tf.variable_scope(scope):
        mask_type = mask_type.lower()
        batch_size, height, width, channel = inputs.get_shape().as_list()

        kernel_h, kernel_w = kernel_shape
        stride_h, stride_w = strides

        assert kernel_h % 2 == 1 and kernel_w % 2 == 1, "kernel height and width should be odd number"

        center_h = kernel_h // 2
        center_w = kernel_w // 2

        weights_shape = [kernel_h, kernel_w, channel, num_outputs]
        weights = tf.get_variable("weights", weights_shape, tf.float32, weights_initializer, weights_regularizer)

        if mask_type is not None:
            mask = np.ones((kernel_h, kernel_w, channel, num_outputs), dtype=np.float32)
            mask[center_h, center_w+1: ,: ,:] = 0.
            mask[center_h+1:, :, :, :] = 0.
            
            if mask_type == 'a':
                mask[center_h,center_w,:,:] = 0.
                
            weights *= tf.constant(mask, dtype=tf.float32)
            tf.add_to_collection('conv2d_weights_%s' % mask_type, weights)

        outputs = tf.nn.conv2d(inputs, weights, [1, stride_h, stride_w, 1], padding=padding, name='outputs')
        tf.add_to_collection('conv2d_outputs', outputs)        
       
        if normalizer_fn:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)
        else:
            if biases_initializer is not None:
                biases = tf.get_variable("biases", [num_outputs], tf.float32, biases_initializer, biases_regularizer)
                outputs = tf.nn.bias_add(outputs, biases, name='outputs_plus_b')
                
        if activation_fn:
            outputs = activation_fn(outputs, name='outputs_with_fn')
 
    logger.debug('[conv2d_%s] %s : %s %s -> %s %s' % 
                 (mask_type, scope, inputs.name, inputs.get_shape(), outputs.name, outputs.get_shape()))

    return outputs

def maxpool2d(
    inputs,
    kernel_shape=[2, 2],
    padding="SAME",
    strides=[2, 2],
    scope="maxpool2d"):
    with tf.variable_scope(scope):
        kernel_h, kernel_w = kernel_shape
        stride_h, stride_w = strides
        
        outputs = tf.nn.max_pool(inputs, [1, kernel_h, kernel_w, 1], [1, stride_h, stride_w, 1], padding=padding, name='maxpool')
    
    logger.debug('[maxpool2d] %s : %s %s -> %s %s' % 
                 (scope, inputs.name, inputs.get_shape(), outputs.name, outputs.get_shape()))

    return outputs

def fully_connected(
    inputs,
    num_inputs,
    num_outputs,
    activation_fn=None,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=WEIGHT_INITIALIZER,
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer,
    biases_regularizer=None,
    scope="fc"):
    with tf.variable_scope(scope):
        batch, channel = inputs.get_shape().as_list()
        weights = tf.get_variable("weights", [channel, num_outputs], tf.float32, weights_initializer, weights_regularizer)        
        outputs = tf.matmul(inputs, weights, name='outputs')
        
        if normalizer_fn:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)
        else:
            if biases_initializer is not None:
                biases = tf.get_variable("biases", [num_outputs], tf.float32, biases_initializer, biases_regularizer)
                outputs = tf.nn.bias_add(outputs, biases, name='outputs_plus_b')
                
        if activation_fn:
            outputs = activation_fn(outputs, name='outputs_with_fn')
        
    logger.debug('[full_connected] %s : %s %s -> %s %s' % 
                 (scope, inputs.name, inputs.get_shape(), outputs.name, outputs.get_shape()))
        
    return outputs
