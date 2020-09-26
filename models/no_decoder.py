import os
import sys
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
from model import abstract_model
from utils.tf_util import *

class model(abstract_model):
    def __init__(self, config, inputs, npts, gt, is_training):
        super().__init__(config, inputs, npts, gt, is_training)

    def decoder(self, GFV, is_training):
        pass

    def upsampling_module(self, per_point_features, npts, is_training):
        upsampled_points_list = []
        for i, point_features in enumerate(per_point_features):
            with tf.variable_scope('enc_layer_' + str(i), reuse=tf.AUTO_REUSE):
                each_layer_upsampling_ratio = 2 if self.upsampling_ratio == 8 else 1  # upsampling ratio: 4 or 8
                for j in range(each_layer_upsampling_ratio):
                    with tf.variable_scope('_upsampling_' + str(j), reuse=tf.AUTO_REUSE):
                        expansioned_points = mlp_conv(point_features, self.upsampling_dims, is_training, self.use_bn)
                        upsampled_points_list.append(expansioned_points)

        upsampled_points = tf.concat(upsampled_points_list, axis=2)
        upsampled_points = tf.reshape(upsampled_points, [1, -1, 3])
        upsampled_points = tf.reshape(upsampled_points, [self.batch_size, -1, 3])

        return upsampled_points

    def network(self, incomplete_point_cloud, npts, is_training):
        GFV, per_point_features = self.encoder(incomplete_point_cloud, npts, is_training)
        upsampled_points = self.upsampling_module(per_point_features, npts, is_training)

        output = upsampled_points

        return None, output