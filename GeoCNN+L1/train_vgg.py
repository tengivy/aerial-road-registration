import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
import random
import re
import numpy as np
import matplotlib.pyplot as plt
from vgg16 import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

Validation = True
BATCH_SIZE = 16
LEARNING_RATE = 5e-4 #1e-6 #5e-5 #5e-5
Decay_steps = 10000 #10000 #10000 #5000
Decay_rate = 0.96 # 0.94 #0.94 #0.94
MAX_STEPS = 1000000
MAX_VAL = 254  # 3049/batchsize
EPOCH = 2452  # 39236/batchsize

DATA_DIR = 'DATASET_TRAIN_30_60_FR_AU/'
VAL_DIR = 'DATASET_VAL_30_60_FR_AU/'
CHECKPOINT_DIR = 'vgg_checkpoints_30_60/'
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
INPUT_WIDTH = 400
INPUT_HEIGHT = 400


def get_files(data_root):
    with open(data_root + 'train.txt', 'r') as f:
        frames = f.readlines()
    subfolders = [x.split()[0] for x in frames]
    image_osm_list = [os.path.join(data_root, subfolders[i], 'OSM.jpg') for i in range(len(frames))]
    image_rt_list = [os.path.join(data_root, subfolders[i], 'RT.jpg') for i in range(len(frames))]
    label_list = [x.split('_')[0] for x in frames]
    label_list = [list(map(float, re.findall(r"\-*\d+\.?\d*", s))) for s in label_list]

    return image_osm_list, image_rt_list, label_list


def get_batches(osm_image, rt_image, label):
    # convert the list of images and labels to tensor
    osm_image = tf.cast(osm_image, tf.string)
    rt_image = tf.cast(rt_image, tf.string)
    label = tf.cast(label, tf.float32)
    queue = tf.train.slice_input_producer([osm_image, rt_image, label])
    label = queue[2]
    image_osm = tf.read_file(queue[0])
    image_rt = tf.read_file(queue[1])
    image_osm = tf.image.decode_jpeg(image_osm, channels=3)
    image_rt = tf.image.decode_jpeg(image_rt, channels=3)
    # resize
    image_osm = tf.image.resize_images(image_osm, size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    image_rt = tf.image.resize_images(image_rt, size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    # image_both = tf.concat([image, image_r], axis=2)
    # (x - mean) / adjusted_stddev
    image_osm = tf.image.per_image_standardization(image_osm)
    image_rt = tf.image.per_image_standardization(image_rt)

    image_osm_batch, image_rgb_batch, label_batch = tf.train.batch([image_osm, image_rt, label], batch_size=BATCH_SIZE)
    image_osm_batch = tf.cast(image_osm_batch, tf.float32)
    image_rgb_batch = tf.cast(image_rgb_batch, tf.float32)
    labels_batch = tf.reshape(label_batch, [BATCH_SIZE, 3])

    return image_osm_batch, image_rgb_batch, labels_batch


def loss(pred, label):
    # huber loss
    delta = 1.0
    residual = tf.abs(pred - label)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    loss = tf.where(condition, small_res, large_res)
    loss = tf.reduce_mean(loss[:, 0] + loss[:, 1] + loss[:, 2])
    return loss


def training(loss, lr, global_steps):
    learning_rate = tf.train.exponential_decay(lr, global_steps, Decay_steps, Decay_rate, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    tf.summary.scalar('learning_rate', learning_rate)
    return train_op


def main(_):
    is_training = tf.placeholder(tf.bool, shape=())
    image_osm_list, image_rt_list, label_list = get_files(DATA_DIR)
    train_osm_batches, train_rgb_batches, train_label_batches = get_batches(image_osm_list, image_rt_list, label_list)

    val_list, val_rt_list, val_label_list = get_files(VAL_DIR)
    test_osm_batches, test_rgb_batches, val_label_batches = get_batches(val_list, val_rt_list, val_label_list)

    image_osm_batches = tf.cond(is_training, lambda: train_osm_batches, lambda: test_osm_batches)
    image_rgb_batches = tf.cond(is_training, lambda: train_rgb_batches, lambda: test_rgb_batches)
    label_batches = tf.cond(is_training, lambda: train_label_batches, lambda: val_label_batches)

    pred = nets(image_osm_batches, image_rgb_batches)
    # pred = nets(image_osm_batches, image_rgb_batches)
    # loss
    cost = loss(pred, label_batches)
    tf.summary.scalar('loss', cost)
    t_cost = loss(pred, label_batches)
    # train
    global_step = tf.Variable(0, trainable=False)
    increment_global_step = tf.assign(global_step, global_step + 1)
    train_op = training(cost, LEARNING_RATE, global_step)

    saver = tf.train.Saver(max_to_keep=1)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Continue training...')
        print('Previous step: ', global_step)
    else:
        print('Start training')
        global_step = 0

    tf.summary.image('input_a', image_osm_batches, max_outputs=1)
    tf.summary.image('input_b', image_rgb_batches, max_outputs=1)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('vgg_logs/train2', sess.graph)
    writer_val = tf.summary.FileWriter('vgg_logs/validation2')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEPS):
            step = step + int(global_step) + 1
            if coord.should_stop():
                break
            _, _, train_loss, label1, pred1, result = sess.run([increment_global_step, train_op, cost, label_batches,
                                                                pred, merged], feed_dict={is_training: True})
            if step % 10 == 0:
                # print(step)
                print(step, " loss:{}".format(train_loss))
                print("label:{}".format(label1))
                print("pred:{}".format(pred1))
                writer.add_summary(result, step)
            if step % EPOCH == 0:
                check = os.path.join(CHECKPOINT_DIR, "model.ckpt")
                saver.save(sess, check, global_step=step)
                if Validation:
                    result_all = 0
                    for num in np.arange(MAX_VAL):
                        val_loss, label2, pred2, result1 = sess.run([t_cost, label_batches, pred, merged],
                                                                    feed_dict={is_training: False})
                        print(num, "train loss:{}".format(val_loss))
                        print("label:{}".format(label2))
                        print("pred:{}".format(pred2))
                        result_all += val_loss
                    eval_loss = result_all / MAX_VAL
                    print(eval_loss)
                    writer_val.add_summary(tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=eval_loss)]),
                                       step)
        writer.close()
        writer_val.close()
    except tf.errors.OutOfRangeError:
        print("Done!!!")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    tf.app.run()
