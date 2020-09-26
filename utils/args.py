import argparse
import yaml
import os
import glob

class trainArguments():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--restore', action='store_true')
        self.parser.add_argument('--config_path', default='configs/pcn.yaml')
        self.parser.add_argument('--log_dir', default='logs/dummy')
        self.args = self.parser.parse_args()

    def to_config(self):
        # use config in log directory
        if self.args.restore:
            log_directory = self.args.log_dir
            yaml_file_list = glob.glob(os.path.join(log_directory, '*.yaml'))
            assert len(yaml_file_list) == 1, "A config file must be in log directory."
            config_path = yaml_file_list[0]
        # use new config file
        else:
            config_path = self.args.config_path

        # read config info
        config = yaml.load(open(config_path), Loader=yaml.FullLoader)

        # save basic config info
        config['config_path'] = config_path
        config['restore'] = self.args.restore
        config['log_dir'] = self.args.log_dir


        # link lmdb files for pcn and shapenet_car dataset
        config['dataset']['dir'] = os.path.join('data', config['dataset']['type'])
        if config['dataset']['type'] == 'pcn' or 'car':
            config['dataset']['lmdb_train'] = os.path.join(config['dataset']['dir'], 'train.lmdb')
            config['dataset']['lmdb_valid'] = os.path.join(config['dataset']['dir'], 'valid.lmdb')

        # num_points
        config['dataset']['num_gt_points'] = 2048 if config['dataset']['type'] == 'topnet' else 16384
        config['model']['decoder']['num_decoder_points'] = config['dataset']['num_gt_points']

        # use upsampling module or not
        config['model']['use_decoder_only'] = True if config['model']['upsampling_ratio'] == 0 else False

        return config


class testArguments():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--checkpoint', default='logs/dummy')
        self.parser.add_argument('--results_dir', default='results/dummy')
        self.parser.add_argument('--plot_freq', type=int, default=40)
        self.parser.add_argument('--save_pcd', action='store_true')
        self.args = self.parser.parse_args()

    def to_config(self):
        log_directory = self.args.checkpoint
        yaml_file_list = glob.glob(os.path.join(log_directory, '*.yaml'))
        assert len(yaml_file_list) == 1, "A config file must be in log directory."
        config_path = yaml_file_list[0]

        # read config info
        config = yaml.load(open(config_path), Loader=yaml.FullLoader)

        # save basic config info
        config['config_path'] = config_path
        config['checkpoint'] = self.args.checkpoint
        config['results_dir'] = self.args.results_dir

        # link test dataset
        if config['dataset']['type'] == 'pcn':
            config['dataset']['dir'] = os.path.join('data', config['dataset']['type'], 'test')
            config['list_path'] = os.path.join('data', config['dataset']['type'], 'test.list')
        elif config['dataset']['type'] == 'topnet':
            config['dataset']['dir'] = os.path.join('data', config['dataset']['type'], 'test')
            config['list_path'] = os.path.join('data', config['dataset']['type'], 'test.list')
        else:
            raise NotImplementedError

        # num points
        config['dataset']['num_gt_points'] = 2048 if config['dataset']['type'] == 'topnet' else 16384
        config['model']['decoder']['num_decoder_points'] = config['dataset']['num_gt_points']

        # setting for testing
        config['test_setting'] = {}
        config['test_setting']['plot_freq'] = self.args.plot_freq
        config['test_setting']['save_pcd'] = self.args.save_pcd

        # use upsampling module or not
        config['model']['use_decoder_only'] = True if config['model']['upsampling_ratio'] == 0 else False

        return config

class testF1Arguments(testArguments):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--threshold', type=float, default=0.01)
        self.parser.add_argument('--checkpoint', default='logs/dummy')
        self.parser.add_argument('--results_dir', default='results/dummy/f1_score')
        self.parser.add_argument('--plot_freq', type=int, default=40)
        self.parser.add_argument('--save_pcd', action='store_true')
        self.args = self.parser.parse_args()

    def to_config(self):
        config = super().to_config()
        config['test_setting']['threshold'] = self.args.threshold
        return config

class testSelfConsistencyArguments(testF1Arguments):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--threshold', type=float, default=0.01)
        self.parser.add_argument('--checkpoint', default='logs/dummy')
        self.parser.add_argument('--results_dir', default='results/dummy/self_consistency')
        self.parser.add_argument('--plot_freq', type=int, default=40)
        self.parser.add_argument('--save_pcd', action='store_true')
        self.args = self.parser.parse_args()

class testKittiArguments():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--kitti_dir', default='data/kitti')
        self.parser.add_argument('--checkpoint', default='logs/dummy')
        self.parser.add_argument('--results_dir', default='results/dummy')
        self.parser.add_argument('--plot_freq', type=int, default=40)
        self.parser.add_argument('--save_pcd', action='store_true')
        self.args = self.parser.parse_args()

    def to_config(self):
        log_directory = self.args.checkpoint
        yaml_file_list = glob.glob(os.path.join(log_directory, '*.yaml'))
        assert len(yaml_file_list) == 1, "A config file must be in log directory."
        config_path = yaml_file_list[0]

        # read config info
        config = yaml.load(open(config_path), Loader=yaml.FullLoader)

        # save basic config info
        config['config_path'] = config_path
        config['checkpoint'] = self.args.checkpoint
        config['results_dir'] = self.args.results_dir

        # link directories for kitti dataset
        config['pcd_dir'] = os.path.join(self.args.kitti_dir, 'cars')
        config['bbox_dir'] = os.path.join(self.args.kitti_dir, 'bboxes')
        config['tracklet_dir'] = os.path.join(self.args.kitti_dir, 'tracklets')

        # num points
        config['dataset']['num_gt_points'] = 2048 if config['dataset']['type'] == 'topnet' else 16384
        config['model']['decoder']['num_decoder_points'] = config['dataset']['num_gt_points']

        # setting for testing
        config['test_setting'] = {}
        config['test_setting']['plot_freq'] = self.args.plot_freq
        config['test_setting']['save_pcd'] = self.args.save_pcd

        # use upsampling module or not
        config['model']['use_decoder_only'] = True if config['model']['upsampling_ratio'] == 0 else False

        return config
