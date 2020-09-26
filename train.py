# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018
# revised by Hyeontae Son
import datetime
import importlib
import os
import sys
import time

import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
from utils.data_util import lmdb_dataflow, DataLoader
from utils.visu_util import plot_pcd_three_views
from utils.args import trainArguments
from termcolor import colored


def train(config):
    data_config = config['dataset']
    train_config = config['train_setting']
    lr_config = train_config['learning_rate']

    # Data
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    inputs_pl = tf.placeholder(tf.float32, (1, None, 3), 'inputs')
    npts_pl = tf.placeholder(tf.int32, (train_config['batch_size'],), 'num_points')
    gt_pl = tf.placeholder(tf.float32, (train_config['batch_size'], data_config['num_gt_points'], 3), 'ground_truths')

    if data_config['type'] == 'pcn' or data_config['type'] == 'car':
        df_train, num_train = lmdb_dataflow(
            data_config['lmdb_train'], train_config['batch_size'], train_config['num_input_points'],
            data_config['num_gt_points'], is_training=True)
        df_valid, num_valid = lmdb_dataflow(
            data_config['lmdb_valid'], train_config['batch_size'], train_config['num_input_points'],
            data_config['num_gt_points'], is_training=False)
        train_gen = df_train.get_data()
        valid_gen = df_valid.get_data()
    elif data_config['type'] == 'topnet':
        dataset_train = DataLoader(data_config['dir'], train_config['batch_size'], is_train=True)
        dataset_valid = DataLoader(data_config['dir'], train_config['batch_size'], is_train=False)
        num_train = dataset_train.get_num_data()
        num_valid = dataset_valid.get_num_data()
        train_get_next = dataset_train.get_next()
        valid_get_next = dataset_valid.get_next()
    else:
        raise NotImplementedError

    # Model
    model_module = importlib.import_module(config['model']['decoder']['type'])
    model = model_module.model(config, inputs_pl, npts_pl, gt_pl, is_training_pl)

    # Optimizer
    optimizer = importlib.import_module('optimizer').optimizer(lr_config, model.global_step, model.target_loss)

    # TF Config
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    config_proto.allow_soft_placement = True
    sess = tf.Session(config=config_proto)
    saver = tf.train.Saver()
    train_summary = tf.summary.merge_all('train_summary')
    valid_summary = tf.summary.merge_all('valid_summary')

    # restart training
    if config['restore']:
        saver.restore(sess, tf.train.latest_checkpoint(config['log_dir']))
        writer = tf.summary.FileWriter(config['log_dir'])

        # calc the last best valid loss
        num_eval_steps = num_valid // train_config['batch_size']
        total_eval_loss = 0
        sess.run(tf.local_variables_initializer())

        for i in range(num_eval_steps):
            if data_config['type'] == 'pcn' or data_config['type'] == 'car':
                ids, inputs, npts, gt = next(valid_gen)
            elif data_config['type'] == 'topnet':
                ids, inputs, npts, gt = sess.run(valid_get_next)
                inputs = np.reshape(inputs, (1, -1, 3)) # for same input format with PCN
            else:
                raise NotImplementedError

            gt = gt[:, :data_config['num_gt_points'], :]
            feed_dict = {inputs_pl: inputs, npts_pl: npts, gt_pl: gt, is_training_pl: False}
            evaluation_loss = sess.run(model.evaluation_loss, feed_dict=feed_dict)
            total_eval_loss += evaluation_loss
        best_valid_loss = total_eval_loss / num_eval_steps

    # train from scratch
    else:
        sess.run(tf.global_variables_initializer())
        if os.path.exists(config['log_dir']):
            delete_key = input(colored('%s exists. Delete? [y (or enter)/N]'
                                       % config['log_dir'], 'white', 'on_red'))
            if delete_key == 'y' or delete_key == "":
                os.system('rm -rf %s/*' % config['log_dir'])
                os.makedirs(os.path.join(config['log_dir'], 'plots'))
        else:
            os.makedirs(os.path.join(config['log_dir'], 'plots'))

        # save configuration in log directory
        os.system('cp %s %s' % (config['config_path'], config['log_dir']))
        os.system('cp train.py %s' % config['log_dir'])

        writer = tf.summary.FileWriter(config['log_dir'], sess.graph)
        best_valid_loss = 1e5  # initialize with enough big num

    print(colored("Training will begin.. ", 'grey', 'on_green'))
    print(colored("Batch_size: " + str(train_config['batch_size']), 'grey', 'on_green'))
    print(colored("Batch norm use?: " + str(config['model']['use_bn']), 'red', 'on_green'))
    print(colored("Decoder arch: " + config['model']['decoder']['type'], 'grey', 'on_green'))
    print(colored("Last best_validation_loss: " + str(best_valid_loss), 'grey', 'on_green'))

    # Training
    total_time = 0
    train_start = time.time()
    init_step = sess.run(model.global_step)

    for step in range(init_step + 1, train_config['max_step'] + 1):
        epoch = step * train_config['batch_size'] // num_train + 1

        if data_config['type'] == 'pcn' or data_config['type'] == 'car':
            ids, inputs, npts, gt = next(train_gen)
        elif data_config['type'] == 'topnet':
            ids, inputs, npts, gt = sess.run(train_get_next)
            inputs = np.reshape(inputs, (1, -1, 3)) # for same input format with PCN
        else:
            raise NotImplementedError

        gt = gt[:, :data_config['num_gt_points'], :]
        start = time.time()
        feed_dict = {inputs_pl: inputs, npts_pl: npts, gt_pl: gt, is_training_pl: True}
        _, target_loss, summary = sess.run([optimizer.train, model.target_loss, train_summary], feed_dict=feed_dict)

        total_time += time.time() - start
        writer.add_summary(summary, step)

        # logging
        if step % train_config['steps_per_print'] == 0:
            print('epoch %d  step %d  target_loss %.8f - time per batch %.4f' %
                  (epoch, step, target_loss, total_time / train_config['steps_per_print']))
            total_time = 0

        # eval on validation set
        if step % train_config['steps_per_eval'] == 0:
            print(colored('Testing...', 'grey', 'on_green'))
            num_eval_steps = num_valid // train_config['batch_size']
            total_eval_loss = 0
            total_time = 0
            sess.run(tf.local_variables_initializer())
            for i in range(num_eval_steps):
                start = time.time()
                if data_config['type'] == 'pcn' or data_config['type'] == 'car':
                    ids, inputs, npts, gt = next(valid_gen)
                elif data_config['type'] == 'topnet':
                    ids, inputs, npts, gt = sess.run(valid_get_next)
                    inputs = np.reshape(inputs, (1, -1, 3))  # for same input format with PCN
                else:
                    raise NotImplementedError
                gt = gt[:, :data_config['num_gt_points'], :]
                feed_dict = {inputs_pl: inputs, npts_pl: npts, gt_pl: gt, is_training_pl: False}
                evaluation_loss, _ = sess.run([model.evaluation_loss, model.update], feed_dict=feed_dict)
                total_eval_loss += evaluation_loss
                total_time += time.time() - start
            summary = sess.run(valid_summary, feed_dict={is_training_pl: False})
            writer.add_summary(summary, step)
            temp_valid_loss = total_eval_loss / num_eval_steps
            print(colored('epoch %d  step %d  eval_loss %.8f - time per batch %.4f' %
                          (epoch, step, temp_valid_loss, total_time / num_eval_steps),
                          'grey', 'on_green'))
            if temp_valid_loss <= best_valid_loss: # save best model for validation set
                best_valid_loss = temp_valid_loss
                saver.save(sess, os.path.join(config['log_dir'], 'model'), step)
                print(colored('Model saved at %s' % config['log_dir'], 'white', 'on_blue'))
            total_time = 0

        # visualize
        if step % config['visualizing']['steps_per_visu'] == 0:
            print('visualizing!')
            if data_config['type'] == 'pcn' or data_config['type'] == 'car':
                vis_ids, vis_inputs, vis_npts, vis_gt = next(valid_gen)
            elif data_config['type'] == 'topnet':
                vis_ids, vis_inputs, vis_npts, vis_gt = sess.run(valid_get_next)

                # for replace the character "/" to "_"
                vis_ids = vis_ids.astype('U')
                vis_ids = np.char.split(vis_ids, sep='/', maxsplit=1)
                vis_ids = np.char.join(['_'] * train_config['batch_size'], vis_ids)

                # for same input format with PCN
                vis_inputs = np.reshape(vis_inputs, (1, -1, 3))
            else:
                raise NotImplementedError

            vis_feed_dict = {inputs_pl:vis_inputs, npts_pl:vis_npts, gt_pl:vis_gt, is_training_pl:False}
            all_pcds = sess.run(model.visualize_ops, feed_dict=vis_feed_dict)
            is_from_decoder = \
                np.arange(0, config['model']['decoder']['num_decoder_points'] + config['model']['upsampling_ratio'] * train_config['num_input_points'])\
                >= config['model']['upsampling_ratio'] * train_config['num_input_points']

            for i in range(0, train_config['batch_size'], config['visualizing']['visu_freq']):
                plot_path = os.path.join(config['log_dir'], 'plots',
                                        'epoch_%d_step_%d_%s.png' % (epoch, step, vis_ids[i]))
                pcds = [x[i] for x in all_pcds]
                if config['visualizing']['visu_split']:
                    plot_pcd_three_views(plot_path, pcds, model.visualize_titles, is_from_decoder)
                else:
                    plot_pcd_three_views(plot_path, pcds, model.visualize_titles, None)

    print(colored("Training ended!", 'grey', 'on_green'))
    print('Total training time', datetime.timedelta(seconds=time.time() - train_start))
    sess.close()


if __name__ == '__main__':
    config = trainArguments().to_config()
    train(config)
