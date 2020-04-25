import scipy.misc
import tensorflow as tf
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import time
from MIE_net import nets


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

VAL_DIR = 'DATASET_TEST_30_60_CHALLENGE/'
LOG_DIR = 'MIE_checkpoints_30_60'
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
INPUT_WIDTH = 400
INPUT_HEIGHT = 400

def get_list():
    with open(VAL_DIR + 'challenge_list.txt', 'r') as f:
        frames = f.readlines()
    subfolders = [x.split()[0] for x in frames]
    osm_list = np.array([os.path.join(VAL_DIR, subfolders[i], 'OSM.jpg') for i in range(len(frames))])
    rt_list = np.array([os.path.join(VAL_DIR, subfolders[i], 'RT.jpg') for i in range(len(frames))])
    return osm_list, rt_list


def get_input(input_osm, input_rgb):
    image_osm = tf.image.resize_images(input_osm, size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    image_rgb = tf.image.resize_images(input_rgb, size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    # (x - mean) / adjusted_stddev
    image_osm = tf.image.per_image_standardization(image_osm)
    image_rgb = tf.image.per_image_standardization(image_rgb)
    image_osm = tf.reshape(image_osm, [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    image_rgb = tf.reshape(image_rgb, [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    image_osm = tf.cast(image_osm, tf.float32)
    image_rgb = tf.cast(image_rgb, tf.float32)
    return image_osm, image_rgb


def main(_):
    with tf.Graph().as_default():
        input_osm = tf.placeholder(tf.uint8, [INPUT_HEIGHT, INPUT_WIDTH, 3], name='osm_input')
        input_rgb = tf.placeholder(tf.uint8, [INPUT_HEIGHT, INPUT_WIDTH, 3], name='rt_input')

        image_osm, image_rgb = get_input(input_osm, input_rgb)
        pose_output = nets(image_osm, image_rgb)

        osm_list, rgb_list = get_list()
        N = len(osm_list)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(LOG_DIR)
            print(ckpt.model_checkpoint_path)
            print("Number of input images: ", N)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success')
                print('global step: ', global_step)
            else:
                print('No checkpoint')

            n1 = n2 = n3 = n4 = 0.0
            dr1 = dr2 = dr3 = dr4 = float(0)
            dx1 = dx2 = dx3 = dx4 = float(0)
            dy1 = dy2 = dy3 = dy4 = float(0)
            dr_all = dx_all = dy_all = float(0)
            overlap_rate_list = []
            dr_list = []
            dx_list = []
            dy_list = []
            time_list = []

            for num in range(N):
                src_img = scipy.misc.imread(osm_list[num])
                rt_img = scipy.misc.imread(rgb_list[num])
                label = rgb_list[num].split('/')[1].split('_')[0]
                r = float(label.split(',')[0])
                tx = float(label.split(',')[1].split(',')[0])
                ty = float(label.split(',')[2])
                t1 = time.clock()
                prediction = sess.run(pose_output, feed_dict={input_osm: src_img, input_rgb: rt_img})
                t2 = time.clock()
                elapsed_time = t2 - t1
                time_list.append(elapsed_time)

                # compute overlap ratio
                inside_num = 0.0
                for x in range(-200, 201, 1):
                    for y in range(-200, 201, 1):
                        theta = math.radians(abs(r))
                        u = x * math.cos(theta) - y * math.sin(theta) + tx
                        v = x * math.sin(theta) + y * math.cos(theta) + ty
                        if -200 <=u<=200 and -200 <=v<=200:
                           inside_num = inside_num + 1
                overlap_rate = 100*inside_num/(400 * 400)

                dr = abs(r - prediction[0][0])
                dx = abs(tx - prediction[0][1])
                dy = abs(ty - prediction[0][2])
                d_loss = dr + dx + dy

                overlap_rate_list.append(overlap_rate)
                dr_list.append(dr)
                dx_list.append(dx)
                dy_list.append(dy)

                if d_loss > 30.:
                    print("No.", num, "    Elapsed Time: %.5f s" % elapsed_time)
                    # print("Overlap Rate: {}%".format(round(overlap_rate, 2)))
                    print("Label:", label)
                    print("Pred:", prediction)

                if 60 <= overlap_rate < 70:
                    n1 += 1
                    dr1 += dr
                    dx1 += dx
                    dy1 += dy
                elif overlap_rate < 80:
                    n2 = n2 + 1
                    dr2 += dr
                    dx2 += dx
                    dy2 += dy
                elif overlap_rate < 90:
                    n3 = n3 + 1
                    dr3 += dr
                    dx3 += dx
                    dy3 += dy
                elif overlap_rate <= 100:
                    n4 = n4 + 1
                    dr4 += dr
                    dx4 += dx
                    dy4 += dy
                dr_all += dr
                dx_all += dx
                dy_all += dy

            print("\nOverlap Rate:  60% - 70%  : ")
            print("Number:", int(n1))
            if int(n1) != 0:
                print("Avg Rotation Error: %.4f" % (dr1/n1))
                print("Avg X-Translation Error: %.4f" % (dx1/n1))
                print("Avg Y-Translation Error: %.4f" % (dy1/n1))
            print("\nOverlap Rate:  70% - 80%  : ")
            print("Number:", int(n2))
            if int(n2) != 0:
                print("Avg Rotation Error: %.4f" % (dr2 / n2))
                print("Avg X-Translation Error: %.4f" % (dx2 / n2))
                print("Avg Y-Translation Error: %.4f" % (dy2 / n2))
            print("\nOverlap Rate:  80% - 90%  : ")
            print("Number:", int(n3))
            if int(n3) != 0:
                print("Avg Rotation Error: %.4f" % (dr3 / n3))
                print("Avg X-Translation Error: %.4f" % (dx3 / n3))
                print("Avg Y-Translation Error: %.4f" % (dy3 / n3))
            print("\nOverlap Rate:  90% - 100%  : ")
            print("Number:", int(n4))
            if int(n4) != 0:
                print("Avg Rotation Error: %.4f" % (dr4 / n4))
                print("Avg X-Translation Error: %.4f" % (dx4 / n4))
                print("Avg Y-Translation Error: %.4f" % (dy4 / n4))
            print("\nALL : ")
            print("Avg Rotation Error:", round((dr_all / N), 4))
            print("Avg X-Translation Error:", round((dx_all / N), 4))
            print("Avg Y-Translation Error:", round((dy_all / N), 4))

            time_list.pop(0)
            print("\nFPS: %.4f" % (1/np.mean(time_list)))
            print("Avg Elapsed Time: %.4f s" % np.mean(time_list))
            print("\nMax dr: %.4f" % max(dr_list))
            print("Max dx: %.4f  Max dy: %.4f" % (max(dx_list), max(dy_list)))

            fig = plt.figure()
            ax1 = plt.subplot(2, 1, 1)
            ax2 = plt.subplot(2, 1, 2)
            plt.sca(ax1)
            plt.title("Rotation Error", fontsize=14)
            plt.scatter(overlap_rate_list, dr_list, s=5)
            plt.xlabel("Overlap Rate / %", fontsize=8)
            plt.ylabel("Rotation Error / degree", fontsize=8)
            plt.tick_params(axis='both', labelsize=8)
            plt.sca(ax2)
            plt.title("Translation Error", fontsize=14)
            plt.scatter(overlap_rate_list, dx_list, s=5, color='red', label='X-Translation Error')
            plt.scatter(overlap_rate_list, dy_list, s=5, color='blue', label='Y-Translation Error')
            plt.xlabel("Overlap Rate / %", fontsize=8)
            plt.ylabel(" Translation Error / pixel", fontsize=8)
            plt.tick_params(axis='both', labelsize=8)
            plt.legend()
            fig.tight_layout()
            plt.show()


if __name__ == '__main__':
    tf.app.run()
