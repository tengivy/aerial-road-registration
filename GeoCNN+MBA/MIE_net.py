import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def residual_block(_input, input_channels, output_channels, strides=1, scope="residual_block"):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(0.0005),
                            # weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
            preact = slim.batch_norm(_input, activation_fn=tf.nn.relu, scope='preact')
            res_conv1 = slim.conv2d(preact, output_channels / 4, [1, 1], stride=1, scope='res_conv1')
            res_conv2 = slim.conv2d(res_conv1, output_channels / 4, [3, 3], stride=strides, scope='res_conv2')
            res_conv3 = slim.conv2d(res_conv2, output_channels, [1, 1], stride=1, normalizer_fn=None,
                                    activation_fn=None, scope='res_conv3')

            if (input_channels != output_channels) or (strides != 1):
                _input = slim.conv2d(preact, output_channels, [1, 1], stride=strides, normalizer_fn=None,
                                     activation_fn=None)
            res_output = res_conv3 + _input
    return res_output


def attention_module1(inputs, in_ch, out_ch, scope="soft_mask_branch_1"):
    # down_sampling * 3
    with tf.variable_scope(scope):
        with tf.variable_scope("down_sampling_1"):
            # max pooling
            out_mask1 = slim.max_pool2d(inputs, [3, 3], stride=2, padding='SAME')
            out_mask1 = residual_block(out_mask1, in_ch, out_ch, scope="soft_block1_1")

        with tf.variable_scope("skip_connection1"):
            skip_connection1 = residual_block(out_mask1, in_ch, out_ch, scope="skip_block1_1")

        with tf.variable_scope("down_sampling_2"):
            # max pooling
            out_mask1 = slim.max_pool2d(out_mask1, [3, 3], stride=2, padding='SAME')
            out_mask1 = residual_block(out_mask1, in_ch, out_ch, scope="soft_block2_1")

        with tf.variable_scope("skip_connection2"):
            skip_connection2 = residual_block(out_mask1, in_ch, out_ch, scope="skip_block2_1")

        with tf.variable_scope("down_sampling_3"):
            # max pooling
            out_mask1 = slim.max_pool2d(out_mask1, [3, 3], stride=2, padding='SAME')

        with tf.variable_scope("res_blocks"):
            out_mask1 = residual_block(out_mask1, in_ch, out_ch, scope="soft_block3_1")

        with tf.variable_scope("up_sampling_1"):
            out_mask1 = residual_block(out_mask1, in_ch, out_ch, scope="soft_block5_1")

            # interpolation
            out_mask1 = tf.keras.layers.UpSampling2D([2, 2])(out_mask1)

        # add skip connection
        out_mask1 += skip_connection2

        with tf.variable_scope("up_sampling_2"):
            out_mask1 = residual_block(out_mask1, in_ch, out_ch, scope="soft_block6_1")

            # interpolation
            out_mask1 = tf.keras.layers.UpSampling2D([2, 2])(out_mask1)

        # add skip connection
        out_mask1 += skip_connection1

        with tf.variable_scope("up_sampling_3"):
            out_mask1 = residual_block(out_mask1, in_ch, out_ch, scope="soft_block7_1")

            # interpolation
            out_mask1 = tf.keras.layers.UpSampling2D([2, 2])(out_mask1)

    return out_mask1


def attention_module2(inputs, in_ch, out_ch, scope="soft_mask_branch_2"):
    # down_sampling * 2
    with tf.variable_scope(scope):
        with tf.variable_scope("down_sampling_1"):
            # max pooling
            out_mask2 = slim.max_pool2d(inputs, [3, 3], stride=2, padding='SAME')
            out_mask2 = residual_block(out_mask2, in_ch, out_ch, scope="soft_block1_2")

        with tf.variable_scope("skip_connection1"):
            skip_connection1 = residual_block(out_mask2, in_ch, out_ch, scope="skip_block1_2")

        with tf.variable_scope("down_sampling_2"):
            # max pooling
            out_mask2 = slim.max_pool2d(out_mask2, [3, 3], stride=2, padding='SAME')

        with tf.variable_scope("res_blocks"):
            out_mask2 = residual_block(out_mask2, in_ch, out_ch, scope="soft_block2_2")

        with tf.variable_scope("up_sampling_1"):
            out_mask2 = residual_block(out_mask2, in_ch, out_ch, scope="soft_block4_2")

            # interpolation
            out_mask2 = tf.keras.layers.UpSampling2D([2, 2])(out_mask2)

        # add skip connection
        out_mask2 += skip_connection1

        with tf.variable_scope("up_sampling_2"):
            out_mask2 = residual_block(out_mask2, in_ch, out_ch, scope="soft_block5_2")

            # interpolation
            out_mask2 = tf.keras.layers.UpSampling2D([2, 2])(out_mask2)
            out_mask2 = residual_block(out_mask2, in_ch, out_ch, scope="soft_block5_3")

    return out_mask2


def attention_module3(inputs, in_ch, out_ch, scope="soft_mask_branch_3"):
    # down_sampling * 1
    with tf.variable_scope(scope):
        with tf.variable_scope("down_sampling_1"):
            # max pooling
            out_mask3 = slim.max_pool2d(inputs, [3, 3], stride=2, padding='SAME')
            out_mask3 = residual_block(out_mask3, in_ch, out_ch, scope="soft_block1_3")

        with tf.variable_scope("res_blocks"):
            out_mask3 = residual_block(out_mask3, in_ch, out_ch, scope="soft_block2_3")

        with tf.variable_scope("up_sampling_1"):
            # interpolation
            out_mask3 = tf.keras.layers.UpSampling2D([2, 2])(out_mask3)
            out_mask3 = residual_block(out_mask3, in_ch, out_ch, scope="soft_block4_3")

    return out_mask3


def multi_attention(inputs, in_ch, out_ch, scope="multi_attention"):
    with tf.variable_scope(scope):
        # out_mask1 = attention_module1(inputs, in_ch, out_ch)
        out_mask2 = attention_module2(inputs, in_ch, out_ch)
        out_mask3 = attention_module3(inputs, in_ch, out_ch)
        if in_ch != out_ch:
            inputs = slim.conv2d(inputs, out_ch, [1, 1], stride=1, normalizer_fn=None, activation_fn=None)
        with tf.variable_scope("attention"):
            out_mask = out_mask2 + out_mask3
            out_mask = tf.nn.tanh(out_mask)
            output = out_mask * inputs
    return output


def extraction_net(inputs):
    with tf.variable_scope("feature_extraction"):
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


def CorrelationLayer(A, B):
    with tf.variable_scope("Correlation_Layer"):
        Af = tf.reshape(A, [-1, A.shape.as_list()[1] * A.shape.as_list()[2], A.shape.as_list()[3]])
        Bf = tf.reshape(B, [-1, B.shape.as_list()[1] * B.shape.as_list()[2], B.shape.as_list()[3]])
        Bf = tf.transpose(Bf, perm=[0, 2, 1])
        f_ab = tf.matmul(Af, Bf)
        f_ab = tf.reshape(f_ab,
                          [-1, A.shape.as_list()[1], A.shape.as_list()[2], A.shape.as_list()[1] * A.shape.as_list()[2]])
        # 16*16*256
        input_channel = f_ab.get_shape()[3].value
        res = multi_attention(f_ab, input_channel, 256, scope="Correlation_attention_1")
        res = tf.nn.relu(res)
        res = tf.nn.l2_normalize(res, dim=3)
    return res


def RegressionLayer(regression_inputs):
    with tf.variable_scope("Regression_Layer"):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
            cnv1 = slim.conv2d(regression_inputs, 128, [7, 7], stride=1, padding="VALID", scope='cnv1')
            cnv2 = slim.conv2d(cnv1, 64, [5, 5], stride=1, padding="VALID", scope='cnv2')
            flat = slim.flatten(cnv2, scope='flat')
            fc1 = slim.fully_connected(flat, 3, activation_fn=None, normalizer_fn=None, scope='fc1')
    return fc1


def nets(input_a, input_b):
    with tf.variable_scope('siamese') as scope:
        feature_a = extraction_net(input_a)
        scope.reuse_variables()
        feature_b = extraction_net(input_b)
    correlation_ab = CorrelationLayer(feature_a, feature_b)
    Pred = RegressionLayer(correlation_ab)
    print("paramas: ", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    return Pred
