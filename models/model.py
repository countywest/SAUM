import os
import sys
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../fps/'))
from utils.tf_util import *
from abc import ABCMeta, abstractmethod
from tf_sampling import gather_point, farthest_point_sample

class abstract_model(metaclass=ABCMeta):
    def __init__(self, config, inputs, npts, gt, is_training):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # config
        self.config = config
        self.dataset = config['dataset']['type']
        self.num_gt_points = config['dataset']['num_gt_points']
        self.batch_size = npts.get_shape()[0]

        # architecture
        self.encoder_dims = config['model']['encoder_dims']
        self.decoder_arch = config['model']['decoder']['type']
        self.num_decoder_points = config['model']['decoder']['num_decoder_points']
        self.use_decoder_only = config['model']['use_decoder_only']
        self.upsampling_ratio = config['model']['upsampling_ratio']
        self.upsampling_dims = config['model']['upsampling_dims']
        self.use_bn = config['model']['use_bn']
        self.outputs = self.network(inputs, npts, is_training)
        self.coarse = self.outputs[0] # meaningful only when decoder is pcn
        self.completion = self.outputs[1]

        # loss & training
        self.target_loss, self.evaluation_loss, self.update = self.create_loss(gt)

        # visualization
        self.visu_split = config['visualizing']['visu_split']
        self.visualize_ops, self.visualize_titles = self.create_visualize_config(inputs, npts, gt)


    def encoder(self, incomplete_point_cloud, npts, is_training):
        encoder_feature_list = []
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = mlp_conv(incomplete_point_cloud, [self.encoder_dims[0]], is_training, self.use_bn)
            encoder_feature_list.append(features)

        with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
            features = mlp_conv(tf.nn.relu(features), [self.encoder_dims[1]], is_training, self.use_bn)
            encoder_feature_list.append(features)
            features_global = point_unpool(point_maxpool(features, npts, keepdims=True), npts)
            features = tf.concat([features, features_global], axis=2) # tiling

        with tf.variable_scope('encoder_2', reuse=tf.AUTO_REUSE):
            features = mlp_conv(features, [self.encoder_dims[2]], is_training, self.use_bn)
            encoder_feature_list.append(features)

        with tf.variable_scope('encoder_3', reuse=tf.AUTO_REUSE):
            features = mlp_conv(tf.nn.relu(features), [self.encoder_dims[3]], is_training, self.use_bn)
            encoder_feature_list.append(features)
            features_global = point_maxpool(features, npts, keepdims=True)
            GFV = tf.squeeze(features_global, axis=1)

        return GFV, encoder_feature_list

    @abstractmethod
    def decoder(self, GFV, is_training):
        pass

    # SAUM
    def upsampling_module(self, encoder_feature_list, npts, is_training):
        upsampled_points_list = []
        if self.dataset == 'pcn' or self.dataset == 'car': # pcn or shapenet car dataset
            for i, point_features in enumerate(encoder_feature_list):
                with tf.variable_scope('enc_layer_' + str(i), reuse=tf.AUTO_REUSE):
                    each_layer_upsampling_ratio = 2 if self.upsampling_ratio == 8 else 1  # upsampling ratio 8 or 4
                    for j in range(each_layer_upsampling_ratio):
                        with tf.variable_scope('_upsampling_' + str(j), reuse=tf.AUTO_REUSE):
                            expansioned_points = mlp_conv(point_features, self.upsampling_dims, is_training, self.use_bn)
                            upsampled_points_list.append(expansioned_points)

        elif self.dataset == 'topnet': # topnet dataset
            for i, point_features in enumerate(encoder_feature_list[:self.upsampling_ratio]):
                with tf.variable_scope('enc_layer_' + str(i), reuse=tf.AUTO_REUSE):
                    with tf.variable_scope('_upsampling_' + str(0), reuse=tf.AUTO_REUSE):
                        expansioned_points = mlp_conv(point_features, self.upsampling_dims, is_training, self.use_bn)
                        upsampled_points_list.append(expansioned_points)
        else:
            pass

        upsampled_points = tf.concat(upsampled_points_list, axis=2)
        upsampled_points = tf.reshape(upsampled_points, [1, -1, 3])
        upsampled_points = [f for f in tf.split(upsampled_points, npts * self.upsampling_ratio, axis=1)]

        return upsampled_points

    def network(self, incomplete_point_cloud, npts, is_training):
        if self.use_decoder_only:
            GFV, _ = self.encoder(incomplete_point_cloud, npts, is_training)
            decoder_points = self.decoder(GFV, is_training)
            return None, decoder_points # coarse, fine

        GFV, encoder_feature_list = self.encoder(incomplete_point_cloud, npts, is_training)
        upsampled_points = self.upsampling_module(encoder_feature_list, npts, is_training)
        decoder_points = self.decoder(GFV, is_training)

        output_list=[]
        for i in range(self.batch_size):
            output_list.append(upsampled_points[i])
            output_list.append(decoder_points[i])

        output = tf.concat(output_list, axis=1)
        output = tf.reshape(output, [self.batch_size, -1, 3])

        return None, output # coarse, fine

    def create_loss(self, gt):
        _, fine_outputs = self.outputs
        target_loss = chamfer(fine_outputs, gt)
        evaluation_loss = target_loss

        # for tensorboard
        add_train_summary('train/target_loss', target_loss)
        add_train_summary('train/evaluation_loss', evaluation_loss)

        update_target = add_valid_summary('valid/target_loss', target_loss)
        update_eval = add_valid_summary('valid/evaluation_loss', evaluation_loss)

        return target_loss, evaluation_loss, [update_target, update_eval]

    def create_visualize_config(self, inputs, npts, gt):
        visualize_ops = [tf.split(inputs[0], npts, axis=0), self.completion, gt]
        if self.visu_split:
            visualize_titles = ['input', 'output', 'upsampling', self.decoder_arch, 'gt']
        else:
            visualize_titles = ['input', 'output', 'gt']

        return visualize_ops, visualize_titles

    def fps(self, num_gt_points, completion): # farthest point sampling
        fps_indices = farthest_point_sample(num_gt_points, completion)
        fps_completion = gather_point(completion, fps_indices)  # same point number with gt
        return fps_completion, fps_indices
