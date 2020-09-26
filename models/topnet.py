# code from https://github.com/lynetcha/completion3d (Paper:Iynetcha et al, TopNet. CVPR 2019)
# revised by Hyeontae Son
import numpy as np
import math
import sys
import os
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
from model import abstract_model
from utils.tf_util import *

tree_arch = {}
tree_arch[2] = [32, 64]
tree_arch[4] = [4, 8, 8, 8]
tree_arch[6] = [2, 4, 4, 4, 4, 4]
tree_arch[8] = [2, 2, 2, 2, 2, 4, 4, 4]

class model(abstract_model):
    def __init__(self, config, inputs, npts, gt, is_training):
        super().__init__(config, inputs, npts, gt, is_training)

    def create_level(self, level, input_channels, output_channels, inputs, tarch, is_training):
        with tf.variable_scope('level_%d' % (level), reuse=tf.AUTO_REUSE):
            features = mlp_conv(inputs, [input_channels, int(input_channels / 2),
                                         int(input_channels / 4), int(input_channels / 8),
                                         output_channels * int(tarch[level])],
                                is_training, self.use_bn)
            features = tf.reshape(features, [tf.shape(features)[0], -1, output_channels])
        return features

    def get_arch(self, nlevels, npts):
        logmult = int(math.log2(npts / 2048))
        assert 2048 * (2 ** (logmult)) == npts, "Number of points is %d, expected 2048x(2^n)" % (npts)
        arch = tree_arch[nlevels]
        while logmult > 0:
            last_min_pos = np.where(arch == np.min(arch))[0][-1]
            arch[last_min_pos] *= 2
            logmult -= 1
        return arch

    def create_decoder(self, features, NFEAT, code_nfts, tarch, is_training):
        Nin = NFEAT + code_nfts
        Nout = NFEAT
        N0 = int(tarch[0])
        nlevels = len(tarch)
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            level0 = mlp(features, [256, 64, NFEAT * N0], is_training, bn=self.use_bn)
            level0 = tf.tanh(level0, name='tanh_0')
            level0 = tf.reshape(level0, [-1, N0, NFEAT])
            outs = [level0, ]
            for i in range(1, nlevels):
                if i == nlevels - 1:
                    Nout = 3
                inp = outs[-1]
                y = tf.expand_dims(features, 1)
                y = tf.tile(y, [1, tf.shape(inp)[1], 1])
                y = tf.concat([inp, y], 2)
                outs.append(tf.tanh(self.create_level(i, Nin, Nout, y, tarch, is_training), name='tanh_%d' % (i)))

            if self.use_decoder_only:
                return outs[-1]
            else:
                decoder_points = outs[-1]
                decoder_points = tf.reshape(decoder_points, [1, self.num_decoder_points * self.batch_size, 3])
                return [f for f in tf.split(decoder_points, [self.num_decoder_points] * self.batch_size, axis=1)]

    def decoder(self, GFV, is_training):
        nlevels = self.config['model']['decoder']['topnet_hp']['nlevels']
        node_feat_dim = self.config['model']['decoder']['topnet_hp']['node_feat_dim']
        GFV_dim = self.config['model']['encoder_dims'][-1]
        return self.create_decoder(GFV, node_feat_dim, GFV_dim, self.get_arch(nlevels, self.num_decoder_points), is_training)

