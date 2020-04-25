import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def VGG_16(inputs):
    with slim.arg_scope([slim.conv2d], normalizer_fn=None, activation_fn=tf.nn.relu):
        conv1 = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
        conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')
        conv3 = slim.repeat(pool2, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')
        conv4 = slim.repeat(pool3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        pool4 = slim.max_pool2d(conv4, [2, 2], scope='pool4')
        norm = tf.nn.l2_normalize(pool4, dim=3)
    return norm


def CorrelationLayer(A,B):
    with tf.variable_scope("Correlation_Layer"):
        Af = tf.reshape(A, [-1, A.shape.as_list()[1] * A.shape.as_list()[2], A.shape.as_list()[3]])
        Bf = tf.reshape(B, [-1, B.shape.as_list()[1] * B.shape.as_list()[2], B.shape.as_list()[3]])
        Bf = tf.transpose(Bf, perm=[0, 2, 1])
        f_ab = tf.matmul(Af, Bf)
        f_ab = tf.reshape(f_ab,
                          [-1, A.shape.as_list()[1], A.shape.as_list()[2], A.shape.as_list()[1] * A.shape.as_list()[2]])
        f_ab = tf.nn.relu(f_ab)
        f_ab = tf.nn.l2_normalize(f_ab, dim=3)
    return f_ab


def RegressionLayer(inputs):
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
        cnv1 = slim.conv2d(inputs, 128, [7, 7], stride=1, padding="VALID", scope='cnv1')
        cnv2 = slim.conv2d(cnv1, 64, [5, 5], stride=1, padding="VALID", scope='cnv2')
        flat = slim.flatten(cnv2, scope='flat')
        fc = slim.fully_connected(flat, 3, activation_fn=None, normalizer_fn=None, scope='fc')
    return fc


def nets(input_a, input_b):
    with tf.variable_scope('siamese') as scope:
        feature_a = VGG_16(input_a)
        scope.reuse_variables()
        feature_b = VGG_16(input_b)
    f_ab = CorrelationLayer(feature_a, feature_b)
    Pred = RegressionLayer(f_ab)
    print("paramas: ", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    return Pred