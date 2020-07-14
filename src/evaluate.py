import numpy as np
import torch as th
from data import load_multi_images
from metrics import knn_evaluate
from pathlib import Path
from utils import ConfigLoader
from nets.encoders import Encoder
import argparse

def read_arguments():
    parser = argparse.ArgumentParser(description="Parse the parameters for training the model.")
    parser.add_argument('--epoch', type=int, default=2,
                        help='Number of epoch for training the model')
    parser.add_argument('--train-cfg', type=str, default='casia_train_105',
                        help='Train config id.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--env', type=str, default='local',
                        help='Environment to run the script.')
    parser.add_argument('--model-name', type=str, default='siamese_drl_gait_model',
                        help='Name of the model to be used.')
    parser.add_argument('--channels', nargs='+', type=int, default=[0, 1],
                        help='the input channels: 0: x, 1: y, 2: dx, 3: dy, 4: ax, 5: ay')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU id to use.')
    parser.add_argument('--graph-layout', type=str, default='alphapose',
                        help='Layout of the skeleton graph.')
    parser.add_argument('--gcn-strategy', type=str, default='spatial',
                        help='Strategy of the graph convolutional network.')
    parser.add_argument('--max-hop', type=int, default=1,
                        help='Max hop of the GCN.')
    parser.add_argument('--dilation', type=int, default=1,
                        help='Dilation step.')
    parser.add_argument('--timestamp', type=str, help='Timestamp of the model.')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and th.cuda.is_available()
    return args

def evaluate(net, device, data_root, probe_sub_id_set, gallery_sub_id_set, probe_condition_set,
             gallery_condition_set, result_out_file):
    f = open(result_out_file, "w")
    f.write("probe_view,gallery_view,hit_rate_1,hit_rate_5,pre_1,pre_5,recall_1,recall_5\n")
    net = net.to(device, dtype=th.float)
    for probe_view_idx in range(11):
        probe_view = f'{probe_view_idx * 18:03}'
        probe_info = load_multi_images(data_root=data_root, subject_ids=probe_sub_id_set,
                                       conditions=probe_condition_set, views=[probe_view])
        probe_images, probe_sub_ids, probe_conditions, probe_views = probe_info
        probe_images = th.from_numpy(probe_images).to(device, dtype=th.float)
        probe_features, _ = net(probe_images)
        probe_features = probe_features.detach().cpu().numpy()
        for gallery_view_idx in range(11):
            gallery_view = f'{gallery_view_idx * 18:03}'
            gallery_info = load_multi_images(data_root=data_root, subject_ids=gallery_sub_id_set,
                                             conditions=gallery_condition_set, views=[gallery_view])
            gallery_images, gallery_sub_ids, gallery_conditions, gallery_views = gallery_info

            gallery_images = th.from_numpy(gallery_images).to(device, dtype=th.float)
            gallery_features, _ = net(gallery_images)
            gallery_features = gallery_features.detach().cpu().numpy()
            hit_rate_1, pre_1, recall_1 = knn_evaluate(probe_features, probe_sub_ids, gallery_features,
                                                       gallery_sub_ids, k=1)
            hit_rate_5, pre_5, recall_5 = knn_evaluate(probe_features, probe_sub_ids, gallery_features,
                                                       gallery_sub_ids, k=5)
            f.write(f'{probe_view},{gallery_view},{hit_rate_1:.4f},{hit_rate_5:.4f},'
                    f'{pre_1:.4f},{pre_5:.4f},{recall_1:.4f},{recall_5:.4f}\n')
    f.close()

def main():
    args = read_arguments()
    device = th.device(f"cuda:{args.gpu_id}" if args.cuda else "cpu")
    print("device:", device)
    train_cfg = args.train_cfg
    epoch = args.epoch
    env = args.env
    timestamp = args.timestamp
    print("args:", args)

    if env == 'server':
        test_ids = [f'{i:03}' for i in range(75, 125)]
    else:
        test_ids = [f'{i:03}' for i in range(1, 10)]

    ROOT_DIR = Path(__file__).resolve().parents[1]
    model_name = args.model_name
    train_conf_file = Path(ROOT_DIR, f'configs/casia_train_cfg.yaml')
    train_conf = ConfigLoader(train_conf_file).config
    data_root = Path(train_conf['data_path'].format(ROOT_DIR=ROOT_DIR))
    if timestamp is not None:
        checkpoint_dir = Path(ROOT_DIR, f'checkpoints/casia_gei/{model_name}/{train_cfg}/{timestamp}')
    else:
        checkpoint_dir = Path(ROOT_DIR, f'checkpoints/casia_gei/{model_name}/{train_cfg}')
    net = Encoder()
    encoder_file = Path(checkpoint_dir, f'encoder_epoch_{epoch}.pth')
    print("snapshot_file:", encoder_file)
    if timestamp is not None:
        result_out_dir = Path(ROOT_DIR, f'results/casia/casia_gei/{model_name}/{train_cfg}/{timestamp}')
    else:
        result_out_dir = Path(ROOT_DIR, f'results/casia/casia_gei/{model_name}/{train_cfg}')
    result_out_dir.mkdir(parents=True, exist_ok=True)

    net.load_state_dict(th.load(encoder_file))
    net.eval()

    print("Loading model finished.")

    result_out_file = Path(result_out_dir, f'accuracy_overall_epoch_{epoch}.txt')
    evaluate(net=net, device=device, data_root=data_root, probe_sub_id_set=test_ids,
             probe_condition_set=['nm-05', 'nm-06', 'bg-01', 'bg-01', 'cl-01', 'cl-02'], gallery_sub_id_set=test_ids,
             gallery_condition_set=['nm-01', 'nm-02', 'nm-03', 'nm-04'], result_out_file=result_out_file)

    print(f"Done!! {result_out_file}")

    result_out_file = Path(result_out_dir, f'accuracy_nm_vs_nm_epoch_{epoch}.txt')
    evaluate(net=net, device=device, data_root=data_root, probe_sub_id_set=test_ids,
             probe_condition_set=['nm-05', 'nm-06'], gallery_sub_id_set=test_ids,
             gallery_condition_set=['nm-01', 'nm-02', 'nm-03', 'nm-04'], result_out_file=result_out_file)
    print(f"Done!! {result_out_file}")

    result_out_file = Path(result_out_dir, f'accuracy_bg_vs_nm_epoch_{epoch}.txt')
    evaluate(net=net, device=device, data_root=data_root, probe_sub_id_set=test_ids,
             probe_condition_set=['bg-01', 'bg-02'], gallery_sub_id_set=test_ids,
             gallery_condition_set=['nm-01', 'nm-02', 'nm-03', 'nm-04'], result_out_file=result_out_file)
    print(f"Done!! {result_out_file}")

    result_out_file = Path(result_out_dir, f'accuracy_cl_vs_nm_epoch_{epoch}.txt')
    evaluate(net=net, device=device, data_root=data_root, probe_sub_id_set=test_ids,
             probe_condition_set=['cl-01', 'cl-02'], gallery_sub_id_set=test_ids,
             gallery_condition_set=['nm-01', 'nm-02', 'nm-03', 'nm-04'], result_out_file=result_out_file)
    print(f"Done!! {result_out_file}")

    result_out_file = Path(result_out_dir, f'accuracy_cl_vs_bg_epoch_{epoch}.txt')
    evaluate(net=net, device=device, data_root=data_root, probe_sub_id_set=test_ids,
             probe_condition_set=['cl-01', 'cl-02'], gallery_sub_id_set=test_ids,
             gallery_condition_set=['bg-01', 'bg-02'], result_out_file=result_out_file)
    print(f"Done!! {result_out_file}")

    result_out_file = Path(result_out_dir, f'accuracy_bg_vs_cl_epoch_{epoch}.txt')
    evaluate(net=net, device=device, data_root=data_root, probe_sub_id_set=test_ids,
             probe_condition_set=['bg-01', 'bg-02'], gallery_sub_id_set=test_ids,
             gallery_condition_set=['cl-01', 'cl-02'], result_out_file=result_out_file)
    print(f"Done!! {result_out_file}")

if __name__ == '__main__':
    main()
