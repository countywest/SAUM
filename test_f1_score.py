import csv
import importlib
import numpy as np
import os
import tensorflow as tf
import time
import datetime
import sys
import h5py
from utils.io_util import read_pcd, save_pcd
from utils.tf_util import dist_to_nearest
from utils.visu_util import plot_pcd_nn_dist
from utils.args import testF1Arguments
from termcolor import colored
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))

def test_vanilla(config):
    test_config = config['test_setting']
    data_config = config['dataset']
    
    # Data
    inputs = tf.placeholder(tf.float32, (1, None, 3))
    npts = tf.placeholder(tf.int32, (1,))
    gt = tf.placeholder(tf.float32, (1, data_config['num_gt_points'], 3))
    output = tf.placeholder(tf.float32, (1, data_config['num_gt_points'], 3))

    # Model
    model_module = importlib.import_module(config['model']['decoder']['type'])
    model = model_module.model(config, inputs, npts, gt, False)

    # Metric
    nearest_dist_op = dist_to_nearest(output, gt)

    # make results directory
    if os.path.exists(config['results_dir']):
        delete_key = input(colored('%s exists. Delete? [y (or enter)/N]'
                                   % config['results_dir'], 'white', 'on_red'))
        if delete_key == 'y' or delete_key == "":
            os.system('rm -rf %s/*' % config['results_dir'])
    else:
        os.makedirs(os.path.join(config['results_dir']))

    os.system('cp test_f1_score.py %s' % config['results_dir'])

    # TF Config
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    config_proto.allow_soft_placement = True
    sess = tf.Session(config=config_proto)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(config['checkpoint']))

    # Test
    test_start = time.time()
    print(colored("Testing...", 'grey', 'on_green'))

    with open(config['list_path']) as file:
        model_list = file.read().splitlines()

    total_time = 0
    total_f1_score = 0
    f1_score_per_cat = {}

    os.makedirs(config['results_dir'], exist_ok=True)
    csv_file = open(os.path.join(config['results_dir'], 'results.csv'), 'w')
    writer = csv.writer(csv_file, delimiter=',', quotechar='"')
    writer.writerow(['id', 'f1_score'])

    for i, model_id in enumerate(model_list):
        start = time.time()

        # data
        if data_config['type'] == 'pcn' or data_config['type'] == 'car':
            partial = read_pcd(os.path.join(data_config['dir'], 'partial', '%s.pcd' % model_id))
            partial_npts = partial.shape[0]
            gt_complete = read_pcd(os.path.join(data_config['dir'], 'complete', '%s.pcd' % model_id))
        elif data_config['type'] == 'topnet':
            with h5py.File(os.path.join(data_config['dir'], 'partial', '%s.h5' % model_id), 'r') as f:
                partial = f.get('data').value.astype(np.float32)
            partial_npts = partial.shape[0]
            with h5py.File(os.path.join(data_config['dir'], 'gt', '%s.h5' % model_id), 'r') as f:
                gt_complete = f.get('data').value.astype(np.float32)
        else:
            raise NotImplementedError

        # inference
        completion = sess.run(model.completion, feed_dict={inputs: [partial], npts: [partial_npts]})

        nn_dists1, nn_dists2 = sess.run(nearest_dist_op,
                                                feed_dict={output: completion, gt: [gt_complete]})
        P = len(nn_dists1[nn_dists1 < config['test_setting']['threshold']]) / data_config['num_gt_points']
        R = len(nn_dists2[nn_dists2 < config['test_setting']['threshold']]) / data_config['num_gt_points']
        f1_score = 2 * P * R / (P + R)
        total_f1_score += f1_score

        total_time += time.time() - start

        writer.writerow([model_id, f1_score])
        csv_file.flush()

        synset_id, model_id = model_id.split('/')
        if not f1_score_per_cat.get(synset_id):
            f1_score_per_cat[synset_id] = []

        f1_score_per_cat[synset_id].append(f1_score)

        # visualize
        if i % test_config['plot_freq'] == 0:
            os.makedirs(os.path.join(config['results_dir'], 'plots', synset_id), exist_ok=True)
            plot_path = os.path.join(config['results_dir'], 'plots', synset_id, '%s.png' % model_id)

            plot_pcd_nn_dist(plot_path, [completion[0], gt_complete], [nn_dists1[0], nn_dists2[0]],
                              ['inference', 'gt'], 'f1_score %.4f' % f1_score
                              )

        if test_config['save_pcd']:
            os.makedirs(os.path.join(config['results_dir'], 'pcds', synset_id), exist_ok=True)
            save_pcd(os.path.join(config['results_dir'], 'pcds', synset_id, '%s.pcd' % model_id), completion[0])

    writer.writerow(["average", total_f1_score / len(model_list)])

    for synset_id in f1_score_per_cat.keys():
        writer.writerow([synset_id, np.mean(f1_score_per_cat[synset_id])])

    with open(os.path.join(config['results_dir'], 'results_summary.txt'), 'w') as log:
        log.write('Average f1_score(threshold: %.4f): %.8f \n' % (test_config['threshold'], total_f1_score / len(model_list)))
        log.write('## Summary for each category ## \n')
        log.write('ID  f1_score  \n')
        for synset_id in f1_score_per_cat.keys():
            log.write('%s %.8f \n' % (synset_id,
                                          np.mean(f1_score_per_cat[synset_id])
                                     )
                      )

    # print results
    print('Average time: %f' % (total_time / len(model_list)))
    print('Average f1_score(threshold: %.4f): %f' % (test_config['threshold'], total_f1_score / len(model_list)))
    print('f1 score per category')
    for synset_id in f1_score_per_cat.keys():
        print(synset_id, '%f' % np.mean(f1_score_per_cat[synset_id]))

    csv_file.close()
    sess.close()
    print(colored("Test ended!", 'grey', 'on_green'))
    print('Total testing time', datetime.timedelta(seconds=time.time() - test_start))

def test_saum(config):
    test_config = config['test_setting']
    data_config = config['dataset']

    # Data
    inputs = tf.placeholder(tf.float32, (1, None, 3))
    npts = tf.placeholder(tf.int32, (1,))
    gt = tf.placeholder(tf.float32, (1, data_config['num_gt_points'], 3))
    output = tf.placeholder(tf.float32, (1, None, 3))

    # Model
    model_module = importlib.import_module(config['model']['decoder']['type'])
    model = model_module.model(config, inputs, npts, gt, False)

    # Metric
    nearest_dist_op = dist_to_nearest(output, gt)

    # make results directory
    if os.path.exists(config['results_dir']):
        delete_key = input(colored('%s exists. Delete? [y (or enter)/N]'
                                   % config['results_dir'], 'white', 'on_red'))
        if delete_key == 'y' or delete_key == "":
            os.system('rm -rf %s/*' % config['results_dir'])
    else:
        os.makedirs(os.path.join(config['results_dir']))

    os.system('cp test_f1_score.py %s' % config['results_dir'])

    # TF Config
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    config_proto.allow_soft_placement = True
    sess = tf.Session(config=config_proto)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(config['checkpoint']))

    # Test
    test_start = time.time()
    print(colored("Testing f1_score....", 'grey', 'on_green'))

    with open(config['list_path']) as file:
        model_list = file.read().splitlines()

    total_time = 0
    total_fps_f1_score = 0
    fps_f1_score_per_cat = {}

    os.makedirs(config['results_dir'], exist_ok=True)
    csv_file = open(os.path.join(config['results_dir'], 'results.csv'), 'w')
    writer = csv.writer(csv_file, delimiter=',', quotechar='"')
    writer.writerow(['id', 'fps_f1_score'])

    for i, model_id in enumerate(model_list):
        start = time.time()
        # data
        if data_config['type'] == 'pcn' or data_config['type'] == 'car':
            partial = read_pcd(os.path.join(data_config['dir'], 'partial', '%s.pcd' % model_id))
            partial_npts = partial.shape[0]
            gt_complete = read_pcd(os.path.join(data_config['dir'], 'complete', '%s.pcd' % model_id))
        elif data_config['type'] == 'topnet':
            with h5py.File(os.path.join(data_config['dir'], 'partial', '%s.h5' % model_id), 'r') as f:
                partial = f.get('data').value.astype(np.float32)
            partial_npts = partial.shape[0]
            with h5py.File(os.path.join(data_config['dir'], 'gt', '%s.h5' % model_id), 'r') as f:
                gt_complete = f.get('data').value.astype(np.float32)
        else:
            raise NotImplementedError

        # inference
        completion = sess.run(model.completion, feed_dict={inputs: [partial], npts: [partial_npts]})
        if not config['model']['use_decoder_only']:
            fps_completion, fps_indices = sess.run(model.fps(data_config['num_gt_points'], completion))

            # farthest point sampling
            fps_nn_dists1, fps_nn_dists2 = sess.run(nearest_dist_op,
                                                    feed_dict={output: fps_completion, gt: [gt_complete]})
            fps_P = len(fps_nn_dists1[fps_nn_dists1 < config['test_setting']['threshold']]) / data_config['num_gt_points']
            fps_R = len(fps_nn_dists2[fps_nn_dists2 < config['test_setting']['threshold']]) / data_config['num_gt_points']
            fps_f1_score = 2 * fps_P * fps_R / (fps_P + fps_R)
            total_fps_f1_score += fps_f1_score

        writer.writerow([model_id, fps_f1_score])
        csv_file.flush()

        synset_id, model_id = model_id.split('/')

        if not fps_f1_score_per_cat.get(synset_id):
            fps_f1_score_per_cat[synset_id] = []

        fps_f1_score_per_cat[synset_id].append(fps_f1_score)

        total_time += time.time() - start

        # visualize
        if i % config['test_setting']['plot_freq'] == 0:
            fps_dir = os.path.join(config['results_dir'], 'plots', 'fps', synset_id)

            os.makedirs(fps_dir, exist_ok=True)

            fps_plot_path = os.path.join(fps_dir, '%s.png' % model_id)

            plot_pcd_nn_dist(fps_plot_path, [fps_completion[0], gt_complete], [fps_nn_dists1[0], fps_nn_dists2[0]],
                              ['inference', 'gt'], 'FPS f1_score %.4f' % fps_f1_score
                              )

    # write average info in csv file
    writer.writerow(["average",
                     total_fps_f1_score / len(model_list)
                     ])

    for synset_id in fps_f1_score_per_cat.keys():
        writer.writerow([synset_id,
                         np.mean(fps_f1_score_per_cat[synset_id])
                         ])
    csv_file.close()
    sess.close()

    # write average f1_score in txt file
    with open(os.path.join(config['results_dir'], 'results_summary.txt'), 'w') as log:
        log.write('Average FPS f1_score(threshold: %.4f): %.8f \n' % (test_config['threshold'], total_fps_f1_score / len(model_list)))
        log.write('## Summary for each category ## \n')
        log.write('ID  FPS_f1_score\n')
        for synset_id in fps_f1_score_per_cat.keys():
            log.write('%s %.8f\n' % (synset_id,
                                          np.mean(fps_f1_score_per_cat[synset_id])
                                          )
                      )

    # print results
    print('Average time: %f' % (total_time / len(model_list)))
    print('Average FPS f1_score(threshold: %.4f): %.8f' % (test_config['threshold'], total_fps_f1_score / len(model_list)))

    print(colored("Test ended!", 'grey', 'on_green'))
    print('Total testing time', datetime.timedelta(seconds=time.time() - test_start))

def test(config):
    if config['model']['use_decoder_only']:
        test_vanilla(config)
    else:
        test_saum(config)

if __name__ == '__main__':
    config = testF1Arguments().to_config()
    test(config)
