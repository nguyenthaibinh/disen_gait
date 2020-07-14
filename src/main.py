import torch as th
from train import DlrAutoEncTrainer as Trainer
from pathlib import Path
from utils import get_current_time
from data import TripletDataset
from utils import ConfigLoader

import argparse

def read_arguments():
    parser = argparse.ArgumentParser(description="Parse the parameters for training the model.")
    parser.add_argument('--epoch', type=int, default=5000,
                        help='Number of epoch for training the model')
    parser.add_argument('--train-cfg', type=str, default='casia_train_001',
                        help='Train config id.')
    parser.add_argument('--env', type=str, default='local',
                        help='Environment to run the script.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU id to use.')
    parser.add_argument('--multi-gpu', action='store_true', default=False,
                        help='Use multi-gpu or not')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--channels', type=int, default=1,
                        help='the number of input channels.')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers used in data loader.')
    parser.add_argument('--checkpoint-freq', type=int, default=20,
                        help='Checkpoint frequency.')
    parser.add_argument('--neg-ratio', type=float, default=0.0,
                        help='Ratio of negative example in the siamese dataset.')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and th.cuda.is_available()
    return args

def main():
    args = read_arguments()
    device = th.device(f"cuda:{args.gpu_id}" if args.cuda else "cpu")
    print("device:", device)
    print("args:", args)
    env = args.env
    train_cfg = args.train_cfg
    dropout = args.dropout
    channels = args.channels
    multi_gpu = args.multi_gpu
    num_workers = args.num_workers
    checkpoint_freq = args.checkpoint_freq
    neg_ratio = args.neg_ratio
    lr = args.lr
    beta = 1

    width = 64
    height = 64

    ROOT_DIR = Path(__file__).resolve().parents[1]
    train_conf_file = Path(ROOT_DIR, f'configs/casia_train_cfg.yaml')
    train_conf = ConfigLoader(train_conf_file).config

    # data_root = Path(train_conf['data_path'].format(ROOT_DIR=ROOT_DIR))
    data_root = train_conf['data_path']

    train_conf = train_conf[train_cfg]

    if env == 'server':
        train_ids = [f'{i:03}' for i in range(1, 51)]
        val_ids = [f'{i:03}' for i in range(51, 75)]
    else:
        train_ids = [f'{i:03}' for i in range(1, 10)]
        val_ids = [f'{i:03}' for i in range(1, 10)]

    train_condition_set = train_conf['train_conditions']
    val_gallery_condition_set = train_conf['val_gallery_conditions']
    val_probe_condition_set = train_conf['val_prob_conditions']
    train_view_set = train_conf['train_views']
    val_gallery_view_set = train_conf['val_gallery_views']
    val_probe_view_set = train_conf['val_prob_views']

    print("train_views:", train_view_set)
    print("train_conditions:", train_condition_set)

    train_set = TripletDataset(data_root, sub_id_set=train_ids, condition_set=train_condition_set,
                                   view_set=train_view_set)
    val_set = TripletDataset(data_root, sub_id_set=val_ids, condition_set=val_gallery_condition_set,
                                 view_set=val_gallery_view_set)
    print("train_dataset.len:", len(train_set))
    print("val_dataset.len:  ", len(val_set))
    model_name = 'siamese_drl_gait_model'
    cur_time = get_current_time()
    checkpoint_dir = Path(ROOT_DIR, f'checkpoints/casia_gei/{model_name}/{train_cfg}/{cur_time}')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    feature_dir = Path(ROOT_DIR, f'feature_vectors/casia_gei/{model_name}/{train_cfg}/{cur_time}')
    feature_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(ROOT_DIR, f'tensor_log/casia_gei/{model_name}/{train_cfg}/{cur_time}')
    print("log_dir:", log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    in_size = (1, 64, 64)
    net = Trainer(in_size=in_size, beta=beta, checkpoint_dir=checkpoint_dir, checkpoint_freq=checkpoint_freq,
                  margin=2.0, feature_dir=feature_dir, device=device, multi_gpu=multi_gpu, neg_ratio=neg_ratio,
                  num_workers=num_workers, p_dropout=dropout, lr=lr, log_dir=log_dir)

    net.train(train_set=train_set, validation_set=val_set,
              num_epochs=args.epoch, verbose=True)

if __name__ == '__main__':
    main()
