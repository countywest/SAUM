# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018
# revised by Hyeontae Son
import numpy as np
import tensorflow as tf
from tensorpack import dataflow
import os
import h5py
import random

### For all dataset ###
class Dataset:
    def __init__(self, data_config, train_config, is_training):
        self.data_config = data_config
        self.train_config = train_config
        self.is_training = is_training

        if data_config['type'] == 'pcn' or data_config['type'] == 'car':
            target = 'train' if is_training else 'valid'
            self.df, self.num_data = lmdb_dataflow(
                data_config['lmdb_' + target], train_config['batch_size'], train_config['num_input_points'],
                data_config['num_gt_points'], is_training=is_training)
            self.data_gen = self.df.get_data()
        elif data_config['type'] == 'topnet':
            self.loader = DataLoader(data_config['dir'], train_config['batch_size'], is_train=is_training)
            self.num_data = self.loader.get_num_data()
            self.get_next = self.loader.get_next()
        else:
            raise NotImplementedError

    def get_num_data(self):
        return self.num_data

    def fetch(self, sess):
        if self.data_config['type'] == 'pcn' or self.data_config['type'] == 'car':
            ids, inputs, npts, gt = next(self.data_gen)
        elif self.data_config['type'] == 'topnet':
            ids, inputs, npts, gt = sess.run(self.get_next)
            inputs = np.reshape(inputs, (1, -1, 3))  # for same input format with PCN
        else:
            raise NotImplementedError

        return ids, inputs, npts, gt


### For PCN and shapenet_car dataset ###
def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]


class PreprocessData(dataflow.ProxyDataFlow):
    def __init__(self, ds, input_size, output_size):
        super(PreprocessData, self).__init__(ds)
        self.input_size = input_size
        self.output_size = output_size

    def get_data(self):
        for id, input, gt in self.ds.get_data():
            input = resample_pcd(input, self.input_size)
            gt = resample_pcd(gt, self.output_size)
            yield id, input, gt


class BatchData(dataflow.ProxyDataFlow):
    def __init__(self, ds, batch_size, input_size, gt_size, remainder=False, use_list=False):
        super(BatchData, self).__init__(ds)
        self.batch_size = batch_size
        self.input_size = input_size
        self.gt_size = gt_size
        self.remainder = remainder
        self.use_list = use_list

    def __len__(self):
        ds_size = len(self.ds)
        div = ds_size // self.batch_size
        rem = ds_size % self.batch_size
        if rem == 0:
            return div
        return div + int(self.remainder)

    def __iter__(self):
        holder = []
        for data in self.ds:
            holder.append(data)
            if len(holder) == self.batch_size:
                yield self._aggregate_batch(holder, self.use_list)
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield self._aggregate_batch(holder, self.use_list)

    def _aggregate_batch(self, data_holder, use_list=False):
        ''' Concatenate input points along the 0-th dimension
            Stack all other data along the 0-th dimension
        '''
        ids = np.stack([x[0] for x in data_holder])
        inputs = [resample_pcd(x[1], self.input_size) for x in data_holder]
        inputs = np.expand_dims(np.concatenate([x for x in inputs]), 0).astype(np.float32)
        npts = np.stack([self.input_size for x in data_holder]).astype(np.int32)
        gts = np.stack([resample_pcd(x[2], self.gt_size) for x in data_holder]).astype(np.float32)
        return ids, inputs, npts, gts


def lmdb_dataflow(lmdb_path, batch_size, input_size, output_size, is_training, test_speed=False):
    df = dataflow.LMDBSerializer.load(lmdb_path, shuffle=False)
    size = df.size()
    if is_training:
        df = dataflow.LocallyShuffleData(df, buffer_size=2000)
        df = dataflow.PrefetchData(df, nr_prefetch=500, nr_proc=1)
    df = BatchData(df, batch_size, input_size, output_size)
    if is_training:
        df = dataflow.PrefetchDataZMQ(df, nr_proc=8)
    df = dataflow.RepeatedData(df, -1)
    if test_speed:
        dataflow.TestDataSpeed(df, size=1000).start()
    df.reset_state()
    return df, size


def get_queued_data(generator, dtypes, shapes, queue_capacity=10):
    assert len(dtypes) == len(shapes), 'dtypes and shapes must have the same length'
    queue = tf.FIFOQueue(queue_capacity, dtypes, shapes)
    placeholders = [tf.placeholder(dtype, shape) for dtype, shape in zip(dtypes, shapes)]
    enqueue_op = queue.enqueue(placeholders)
    close_op = queue.close(cancel_pending_enqueues=True)
    feed_fn = lambda: {placeholder: value for placeholder, value in zip(placeholders, next(generator))}
    queue_runner = tf.contrib.training.FeedingQueueRunner(queue, [enqueue_op], close_op, feed_fns=[feed_fn])
    tf.train.add_queue_runner(queue_runner)
    return queue.dequeue()


### For TopNet dataset ###
class DataLoader:
    def __init__(self, dataset_dir, batch_size, is_train=False):
        self.batch_size = batch_size
        if is_train:
            with open(os.path.join(dataset_dir, 'train.list')) as f:
                self.id_list_path = f.read().splitlines()
            random.shuffle(self.id_list_path)
            self.data_dir = os.path.join(dataset_dir, 'train')
        else:
            with open(os.path.join(dataset_dir, 'val.list')) as f:
                self.id_list_path = f.read().splitlines()
            self.data_dir = os.path.join(dataset_dir, 'val')

        self.partial_list = [os.path.join(self.data_dir, 'partial', f + ".h5") for f in self.id_list_path]
        self.gt_list = [os.path.join(self.data_dir, 'gt', f + ".h5") for f in self.id_list_path]
        dataset = tf.data.Dataset.from_tensor_slices((self.id_list_path, self.partial_list, self.gt_list))
        dataset = dataset.map(lambda id, partial, gt: tuple(tf.py_func(self.parser,
                                                                       [id, partial, gt],
                                                                       (tf.string, tf.float32, tf.int32, tf.float32)
                                                                       )
                                                            )
                              )
        dataset = dataset.repeat()
        if is_train:
            dataset = dataset.shuffle(buffer_size=2000)
        dataset = dataset.batch(self.batch_size)
        self.iterator = dataset.make_one_shot_iterator()

    def get_num_data(self):
        return len(self.id_list_path)

    def get_next(self):
        next_batch = self.iterator.get_next()
        return next_batch

    @staticmethod
    def h5_reader(path):
        with h5py.File(path, 'r') as f:
            data = f.get('data').value.astype(np.float32)
            return data

    @staticmethod
    def parser(id, partial_path, gt_path):
        partial = DataLoader.h5_reader(partial_path)
        gt = DataLoader.h5_reader(gt_path)
        npts = np.int32(2048)
        return id, partial, npts, gt
