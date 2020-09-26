import tensorflow as tf

class optimizer:
    def __init__(self, lr_config, global_step, target_loss):
        # global step & loss function
        self.global_step = global_step
        self.target_loss = target_loss

        # learning rate
        self.lr_decay = lr_config['lr_decay']
        self.lr_decay_steps = lr_config['lr_decay_steps']
        self.lr_decay_rate = lr_config['lr_decay_rate']
        self.lr_clip = lr_config['lr_clip']
        self.lr = self.create_lr(lr_config['init_lr'], name='learning_rate')
        self.train = self.create_train()

    def create_lr(self, init_lr, name):
        if self.lr_decay:
            lr = tf.train.exponential_decay(init_lr, self.global_step, self.lr_decay_steps, self.lr_decay_rate,
                                            staircase=True, name=name)
            lr = tf.maximum(lr, self.lr_clip)
        else:
            lr = tf.constant(init_lr, name=name)
        return lr

    def create_train(self):
        vars_net = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.target_loss, var_list=vars_net,
                                                                           global_step=self.global_step)
        return train