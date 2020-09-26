import sys
import os
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
from model import abstract_model
from utils.tf_util import *

class model(abstract_model):
    def __init__(self, config, inputs, npts, gt, is_training):
        super().__init__(config, inputs, npts, gt, is_training)

    def decoder(self, GFV, is_training):
        with tf.variable_scope('pcn_decoder_1', reuse=tf.AUTO_REUSE):
            coarse_feat_dims = self.config['model']['decoder']['pcn_hp']['coarse_feat_dims']
            num_coarse_points = coarse_feat_dims[-1] // 3
            coarse = mlp(GFV, coarse_feat_dims, is_training, self.use_bn)
            coarse = tf.reshape(coarse, [-1, num_coarse_points, 3])

        with tf.variable_scope('pcn_decoder_2', reuse=tf.AUTO_REUSE):
            grid_size = self.config['model']['decoder']['pcn_hp']['folding_grid_size']
            fine_feat_dims = self.config['model']['decoder']['pcn_hp']['fine_feat_dims']
            grid = tf.meshgrid(tf.linspace(-0.05, 0.05, grid_size), tf.linspace(-0.05, 0.05, grid_size))
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
            grid_feat = tf.tile(grid, [GFV.shape[0], num_coarse_points, 1])

            point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, grid_size ** 2, 1])
            point_feat = tf.reshape(point_feat, [-1, self.num_decoder_points, 3])

            global_feat = tf.tile(tf.expand_dims(GFV, 1), [1, self.num_decoder_points, 1])

            feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)

            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, grid_size ** 2, 1])
            center = tf.reshape(center, [-1, self.num_decoder_points, 3])

            decoder_points = mlp_conv(feat, fine_feat_dims, is_training, self.use_bn) + center

            if self.use_decoder_only:
                return coarse, decoder_points
            else:
                decoder_points = tf.reshape(decoder_points, [1, self.batch_size * self.num_decoder_points, 3])
                return coarse, [f for f in
                                tf.split(decoder_points, [self.num_decoder_points] * self.batch_size, axis=1)]

    def network(self, incomplete_point_cloud, npts, is_training):
        if self.use_decoder_only:
            GFV, _ = self.encoder(incomplete_point_cloud, npts, is_training)
            coarse, decoder_points = self.decoder(GFV, is_training)
            return coarse, decoder_points

        GFV, encoder_feature_list = self.encoder(incomplete_point_cloud, npts, is_training)
        upsampled_points = self.upsampling_module(encoder_feature_list, npts, is_training)
        coarse, decoder_points = self.decoder(GFV, is_training)

        batch_size = self.batch_size
        output_list=[]
        for i in range(batch_size):
            output_list.append(upsampled_points[i])
            output_list.append(decoder_points[i])
        output = tf.concat(output_list, axis=1)
        output = tf.reshape(output, [batch_size, -1, 3])

        return coarse, output

    def create_loss(self, gt):
        self.alpha = tf.train.piecewise_constant(self.global_step, [10000, 20000, 50000],
                                                 [0.01, 0.1, 0.5, 1.0], 'alpha_op')
        coarse_outputs, fine_outputs = self.outputs
        coarse_loss = chamfer(coarse_outputs, gt)
        fine_loss = chamfer(fine_outputs, gt)
        target_loss = coarse_loss + self.alpha * fine_loss
        evaluation_loss = fine_loss

        # for tensorboard
        add_train_summary('train/target_loss', target_loss)
        add_train_summary('train/evaluation_loss', evaluation_loss)

        update_target = add_valid_summary('valid/target_loss', target_loss)
        update_eval = add_valid_summary('valid/evaluation_loss', evaluation_loss)


        return target_loss, evaluation_loss, [update_target, update_eval]