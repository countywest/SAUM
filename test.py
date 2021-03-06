# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018
# revised by Hyeontae Son
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
from utils.tf_util import chamfer, earth_mover
from utils.visu_util import plot_pcd_three_views
from utils.args import testArguments
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
    cd_op = chamfer(output, gt)
    emd_op = earth_mover(output, gt)

    # make results directory & save configuration
    if os.path.exists(config['results_dir']):
        delete_key = input(colored('%s exists. Delete? [y (or enter)/N]'
                                   % config['results_dir'], 'white', 'on_red'))
        if delete_key == 'y' or delete_key == "":
            os.system('rm -rf %s/*' % config['results_dir'])
    else:
        os.makedirs(os.path.join(config['results_dir']))

    os.system('cp %s %s' % (config['config_path'], config['results_dir']))
    os.system('cp test.py %s' % config['results_dir'])

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
    total_cd = 0
    total_emd = 0
    cd_per_cat = {}
    emd_per_cat = {}
    os.makedirs(config['results_dir'], exist_ok=True)
    csv_file = open(os.path.join(config['results_dir'], 'results.csv'), 'w')
    writer = csv.writer(csv_file, delimiter=',', quotechar='"')
    writer.writerow(['id', 'cd', 'emd'])

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

        cd = sess.run(cd_op, feed_dict={output: completion, gt: [gt_complete]})
        emd = sess.run(emd_op, feed_dict={output: completion, gt:[gt_complete]})
        total_cd += cd
        total_emd += emd

        total_time += time.time() - start

        writer.writerow([model_id, cd, emd])
        csv_file.flush()

        synset_id, model_id = model_id.split('/')
        if not cd_per_cat.get(synset_id):
            cd_per_cat[synset_id] = []
        if not emd_per_cat.get(synset_id):
            emd_per_cat[synset_id] = []
        cd_per_cat[synset_id].append(cd)
        emd_per_cat[synset_id].append(emd)

        # visualize
        if i % test_config['plot_freq'] == 0:
            os.makedirs(os.path.join(config['results_dir'], 'plots', synset_id), exist_ok=True)
            plot_path = os.path.join(config['results_dir'], 'plots', synset_id, '%s.png' % model_id)
            plot_pcd_three_views(plot_path, [partial, completion[0], gt_complete],
                                 model.visualize_titles, None,
                                 'CD %.4f EMD %.4f' %
                                 (cd, emd)
                                 )
        if test_config['save_pcd']:
            os.makedirs(os.path.join(config['results_dir'], 'pcds', synset_id), exist_ok=True)
            save_pcd(os.path.join(config['results_dir'], 'pcds', synset_id, '%s.pcd' % model_id), completion[0])

    writer.writerow(["average", total_cd / len(model_list), total_emd / len(model_list)])

    for synset_id in cd_per_cat.keys():
        writer.writerow([synset_id, np.mean(cd_per_cat[synset_id]), np.mean(emd_per_cat[synset_id])])

    with open(os.path.join(config['results_dir'], 'results_summary.txt'), 'w') as log:
        log.write('Average Chamfer distance: %.8f \n' % (total_cd / len(model_list)))
        log.write('Average Earth mover distance: %.8f \n' % (total_emd / len(model_list)))
        log.write('## Summary for each category ## \n')
        log.write('ID  CD  EMD  \n')
        for synset_id in cd_per_cat.keys():
            log.write('%s %.8f %.8f\n' % (synset_id,
                                          np.mean(cd_per_cat[synset_id]),
                                          np.mean(emd_per_cat[synset_id])
                                          )
                      )

    # print results
    print('Average time: %f' % (total_time / len(model_list)))
    print('Average Chamfer distance: %f' % (total_cd / len(model_list)))
    print('Average Earth mover distance: %f' % (total_emd / len(model_list)))
    print('Chamfer distance per category')
    for synset_id in cd_per_cat.keys():
        print(synset_id, '%f' % np.mean(cd_per_cat[synset_id]))
    print('Earth mover distance per category')
    for synset_id in emd_per_cat.keys():
        print(synset_id, '%f' % np.mean(emd_per_cat[synset_id]))
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
    sampled_output = tf.placeholder(tf.float32, (1, data_config['num_gt_points'], 3))

    # Model
    model_module = importlib.import_module(config['model']['decoder']['type'])
    model = model_module.model(config, inputs, npts, gt, False)

    # Metric
    cd_op = chamfer(output, gt)
    emd_op = earth_mover(sampled_output, gt)

    # make results directory & save configuration
    if os.path.exists(config['results_dir']):
        delete_key = input(colored('%s exists. Delete? [y (or enter)/N]'
                                   % config['results_dir'], 'white', 'on_red'))
        if delete_key == 'y' or delete_key == "":
            os.system('rm -rf %s/*' % config['results_dir'])
    else:
        os.makedirs(os.path.join(config['results_dir']))

    os.system('cp %s %s' % (config['config_path'], config['results_dir']))
    os.system('cp test.py %s' % config['results_dir'])

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

    total_cd = 0
    total_fps_cd = 0
    total_fps_emd = 0

    cd_per_cat = {}
    fps_cd_per_cat = {}
    fps_emd_per_cat = {}

    os.makedirs(config['results_dir'], exist_ok=True)
    csv_file = open(os.path.join(config['results_dir'], 'results.csv'), 'w')
    writer = csv.writer(csv_file, delimiter=',', quotechar='"')
    writer.writerow(['id', 'cd', 'fps_cd', 'fps_emd'])

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

        fps_completion, fps_indices = sess.run(model.fps(data_config['num_gt_points'], completion))

        is_from_decoder_raw = \
            np.arange(0, config['model']['decoder']['num_decoder_points'] + config['model']['upsampling_ratio'] * partial_npts)\
            >= config['model']['upsampling_ratio'] * partial_npts
        is_from_decoder_fps = fps_indices >= config['model']['upsampling_ratio'] * partial_npts

        total_time += time.time() - start

        # raw
        cd = sess.run(cd_op, feed_dict={output: completion, gt: [gt_complete]})
        total_cd += cd

        # farthest point sampling
        fps_cd = sess.run(cd_op, feed_dict={output: fps_completion, gt: [gt_complete]})
        fps_emd = sess.run(emd_op, feed_dict={sampled_output: fps_completion, gt: [gt_complete]})
        total_fps_cd += fps_cd
        total_fps_emd += fps_emd

        writer.writerow([model_id, cd, fps_cd, fps_emd])
        csv_file.flush()

        synset_id, model_id = model_id.split('/')
        if not cd_per_cat.get(synset_id):
            cd_per_cat[synset_id] = []
        if not fps_cd_per_cat.get(synset_id):
            fps_cd_per_cat[synset_id] = []
        if not fps_emd_per_cat.get(synset_id):
            fps_emd_per_cat[synset_id] = []

        cd_per_cat[synset_id].append(cd)
        fps_cd_per_cat[synset_id].append(fps_cd)
        fps_emd_per_cat[synset_id].append(fps_emd)

        # visualize
        if i % test_config['plot_freq'] == 0:
            if config['visualizing']['visu_split']:
                raw_dir = os.path.join(config['results_dir'], 'plots', 'raw', synset_id)
                fps_dir = os.path.join(config['results_dir'], 'plots', 'fps', synset_id)

                os.makedirs(raw_dir, exist_ok=True)
                os.makedirs(fps_dir, exist_ok=True)

                raw_plot_path = os.path.join(raw_dir, '%s.png' % model_id)
                fps_plot_path = os.path.join(fps_dir, '%s.png' % model_id)

                plot_pcd_three_views(raw_plot_path, [partial, completion[0], gt_complete],
                                     model.visualize_titles, is_from_decoder_raw,
                                     'CD %.4f' % cd
                                     )

                plot_pcd_three_views(fps_plot_path, [partial, fps_completion[0], gt_complete],
                                     model.visualize_titles, is_from_decoder_fps[0],
                                     'FPS_CD %.4f FPS_EMD %.4f' % (fps_cd, fps_emd)
                                     )
            else:
                os.makedirs(os.path.join(config['results_dir'], 'plots', synset_id), exist_ok=True)

                plot_path = os.path.join(config['results_dir'], 'plots', synset_id, '%s.png' % model_id)
                plot_pcd_three_views(plot_path, [partial, completion[0], gt_complete],
                                     model.visualize_titles, None,
                                     'CD %.4f FPS_CD %.4f FPS_EMD %.4f' %
                                     (cd, fps_cd, fps_emd)
                                     )

        if test_config['save_pcd']:
            os.makedirs(os.path.join(config['results_dir'], 'pcds', synset_id), exist_ok=True)
            save_pcd(os.path.join(config['results_dir'], 'pcds', synset_id, '%s.pcd' % model_id), completion[0])
            save_pcd(os.path.join(config['results_dir'], 'pcds', synset_id, '%s_fps.pcd' % model_id), fps_completion[0])

    # write average info in csv file
    writer.writerow(["average", total_cd / len(model_list),
                     total_fps_cd / len(model_list), total_fps_emd / len(model_list)
                     ])
    for synset_id in cd_per_cat.keys():
        writer.writerow([synset_id, np.mean(cd_per_cat[synset_id]),
                         np.mean(fps_cd_per_cat[synset_id]), np.mean(fps_emd_per_cat[synset_id])
                         ])

    # write average distances(cd, emd) in txt file
    with open(os.path.join(config['results_dir'], 'results_summary.txt'), 'w') as log:
        log.write('Average Chamfer distance: %.8f \n' % (total_cd / len(model_list)))
        log.write('Average FPS Chamfer distance: %.8f \n' % (total_fps_cd / len(model_list)))
        log.write('Average FPS Earth mover distance: %.8f \n' % (total_fps_emd / len(model_list)))

        log.write('## Summary for each category ## \n')
        log.write('ID  CD  FPS_CD  FPS_EMD\n')
        for synset_id in cd_per_cat.keys():
            log.write('%s %.8f %.8f %.8f\n' % (synset_id,
                                                         np.mean(cd_per_cat[synset_id]),
                                                         np.mean(fps_cd_per_cat[synset_id]),
                                                         np.mean(fps_emd_per_cat[synset_id])
                                                         )
                      )

    # print results
    print('Average time: %f' % (total_time / len(model_list)))
    print('Average Chamfer distance: %f' % (total_cd / len(model_list)))
    print('Average FPS Chamfer distance: %f' % (total_fps_cd / len(model_list)))
    print('Average FPS Earth mover distance: %f' % (total_fps_emd / len(model_list)))

    print('Chamfer distance per category')
    for synset_id in cd_per_cat.keys():
        print(synset_id, '%f' % np.mean(cd_per_cat[synset_id]))
    print('Average FPS Chamfer distance per catergory')
    for synset_id in fps_cd_per_cat.keys():
        print(synset_id, '%f' % np.mean(fps_cd_per_cat[synset_id]))
    print('Average FPS Earth mover distance per category')
    for synset_id in fps_emd_per_cat.keys():
        print(synset_id, '%f' % np.mean(fps_emd_per_cat[synset_id]))
    csv_file.close()
    sess.close()

    print(colored("Test ended!", 'grey', 'on_green'))
    print('Total testing time', datetime.timedelta(seconds=time.time() - test_start))

def test(config):
    if config['model']['use_decoder_only']:
        test_vanilla(config)
    else:
        test_saum(config)

if __name__ == '__main__':
    config = testArguments().to_config()
    test(config)
