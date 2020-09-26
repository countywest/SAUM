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
        with tf.variable_scope('atlas_decoder', reuse=tf.AUTO_REUSE):
            patch_num = self.config['model']['decoder']['atlas_hp']['patch_num']
            x_sample_num = self.config['model']['decoder']['atlas_hp']['x_sample_num']
            y_sample_num = self.config['model']['decoder']['atlas_hp']['y_sample_num']
            feat_dims = self.config['model']['decoder']['atlas_hp']['feat_dims']
            decoder_points = []
            for i in range(patch_num):
                with tf.variable_scope('patch_' + str(i), reuse=tf.compat.v1.AUTO_REUSE):
                    x = tf.linspace(-0.5, 0.5, x_sample_num)
                    y = tf.linspace(-0.5, 0.5, y_sample_num)
                    grid = tf.meshgrid(x, y)
                    grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
                    grid_feat = tf.tile(grid, [self.batch_size, 1, 1])
                    global_feat = tf.tile(tf.expand_dims(GFV, 1), [1, x_sample_num * y_sample_num, 1])
                    feat = tf.concat([grid_feat, global_feat], axis=2)
                    feat = mlp_conv(feat, feat_dims, is_training, self.use_bn)
                    decoder_points.append(feat)
            decoder_points = tf.concat(decoder_points, axis=1)

            if self.use_decoder_only:
                return decoder_points
            else:
                # x_sample_num * y_sample_num * patch_num == self.num_decoder_points
                decoder_points = tf.reshape(decoder_points, [1, self.num_decoder_points * self.batch_size, 3])
                return [f for f in tf.split(decoder_points, [self.num_decoder_points] * self.batch_size, axis=1)]
