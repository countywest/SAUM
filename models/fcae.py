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
        with tf.variable_scope('fcae_decoder', reuse=tf.AUTO_REUSE):
            assert self.config['model']['decoder']['fcae_hp']['feat_dims'][-1] == self.num_gt_points * 3, \
                'Num of the decoder points should be equal to the num of gt points.'
            decoder_points = mlp(GFV, self.config['model']['decoder']['fcae_hp']['feat_dims'], is_training, self.use_bn)
            if self.use_decoder_only:
                decoder_points = tf.reshape(decoder_points, [-1, self.num_decoder_points, 3])
                return decoder_points
            else:
                decoder_points = tf.reshape(decoder_points, [1, self.batch_size * self.num_decoder_points, 3])
                return [f for f in tf.split(decoder_points, [self.num_decoder_points] * self.batch_size, axis=1)]