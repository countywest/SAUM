# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018
# revised by Hyeontae Son
import importlib
import numpy as np
import os
import sys
import tensorflow as tf
import time
from utils.io_util import read_pcd
from utils.visu_util import plot_pcd_three_views
from utils.tf_util import chamfer, dist_to_nearest
from utils.args import testKittiArguments
from termcolor import colored
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))

def test(config):
    data_config = config['dataset'] # config for TRAIN dataset
    
    # Data
    inputs = tf.placeholder(tf.float32, (1, None, 3))
    npts = tf.placeholder(tf.int32, (1,))
    gt = tf.placeholder(tf.float32, (1, data_config['num_gt_points'], 3)) # dummy gt. there is no gt for kitti.
    output = tf.placeholder(tf.float32, (1, data_config['num_gt_points'], 3))

    # Model
    model_module = importlib.import_module(config['model']['decoder']['type'])
    model = model_module.model(config, inputs, npts, gt, False)

    # Metric
    nearest_dist_op = dist_to_nearest(inputs, output)

    # make results directory & save configuration
    if os.path.exists(config['results_dir']):
        delete_key = input(colored('%s exists. Delete? [y (or enter)/N]'
                                   % config['results_dir'], 'white', 'on_red'))
        if delete_key == 'y' or delete_key == "":
            os.system('rm -rf %s/*' % config['results_dir'])
    else:
        os.makedirs(os.path.join(config['results_dir']))

    os.system('cp %s %s' % (config['config_path'], config['results_dir']))
    os.system('cp test_kitti.py %s' % config['results_dir'])

    os.makedirs(os.path.join(config['results_dir'], 'plots'), exist_ok=True)

    # TF Config
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    config_proto.allow_soft_placement = True
    sess = tf.Session(config=config_proto)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(config['checkpoint']))

    def shape_complete(tracklet, car_id):
        origin_partial = read_pcd(os.path.join(config['pcd_dir'], '%s.pcd' % car_id))
        bbox = np.loadtxt(os.path.join(config['bbox_dir'], '%s.txt' % car_id))

        # Calculate center, rotation and scale
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                             [np.sin(yaw), np.cos(yaw), 0],
                             [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]
        bbox /= scale

        partial = np.dot(origin_partial - center, rotation) / scale
        partial = np.dot(partial, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        completion = sess.run(model.completion, feed_dict={inputs: [partial], npts: [partial.shape[0]]})
        if not model.use_decoder_only:
            completion, fps_indices = sess.run(model.fps(data_config['num_gt_points'], completion))
            is_from_decoder_fps = fps_indices >= config['model']['upsampling_ratio'] * partial.shape[0]
        completion = completion[0]

        nn_dists1, _ = sess.run(nearest_dist_op,
                                feed_dict={inputs: [partial], output: [completion]})

        fidelity = np.mean(nn_dists1)

        # visualize
        os.makedirs(os.path.join(config['results_dir'], 'plots', tracklet), exist_ok=True)
        plot_path = os.path.join(config['results_dir'], 'plots', tracklet, '%s.png' % car_id)
        if config['model']['use_decoder_only']:
            plot_pcd_three_views(plot_path, [partial, completion], ['input', 'output'], None,
                             '%d input points' % partial.shape[0], [5, 0.5])
        elif config['visualizing']['visu_split']:
            plot_pcd_three_views(plot_path, [partial, completion], ['input', 'output', 'upsampling', config['model']['decoder']['type']],
                                 is_from_decoder_fps[0], '%d input points' % partial.shape[0], [5, 0.5])

        return completion, fidelity

    tracklets = os.listdir(config['tracklet_dir'])
    total_time = 0
    total_frames = 0
    total_fidelity = 0

    print(colored("Testing...", 'grey', 'on_green'))
    for i, tracklet in enumerate(tracklets):
        start = time.time()

        with open(os.path.join(config['tracklet_dir'], '%s' % tracklet)) as file:
            contents = file.read()
            car_ids = contents.splitlines()

        for j, car_id in enumerate(car_ids):
            completion, fidelity = shape_complete(tracklet.split('.')[0], car_id)
            total_fidelity += fidelity
            total_frames += 1

        total_time += time.time() - start

    print('Average Fidelity: %.8f', total_fidelity / total_frames)
    print('Average time per tracklet:', total_time / len(tracklets))

    with open(os.path.join(config['results_dir'], 'results_summary.txt'), 'w') as log:
        log.write('Average Fidelity: %.8f \n' % (total_fidelity / total_frames))

    sess.close()

if __name__ == '__main__':
    config = testKittiArguments().to_config()
    test(config)
