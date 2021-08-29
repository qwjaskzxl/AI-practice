# -*- coding: utf-8 -*-
import os, argparse, logging, random
import torch
import numpy as np
from optimizer.lr_scheduler import LR_SCHEDULER_REGISTRY

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu', help='')
    parser.add_argument('--g', type=str, default='3')
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--evaluate', action='store_true', help='evaluate the model on dev set')
    parser.add_argument('--train-and-eval', action='store_true', help='evaluate the model per epoch during training')
    parser.add_argument('--eval-n-epoch', type=int, default=1, help='')
    parser.add_argument('--load-pretrained-model', action='store_true', help='load pretrained model')
    parser.add_argument('--partial-load', action='store_true', help='load pretrained model for partial modules')
    parser.add_argument('--model-state-dict-path', default='ckpt/model-rid_chunk1Hinf_L3-best', type=str, help='model_state_dict_path')
    parser.add_argument('--model-saved-path', type=str, default='ckpt/', help='')
    parser.add_argument('--mn', type=str, default='', help='model_name')
    parser.add_argument('--dec', type=str, default='rnnt', help='decoder approach')

    parser.add_argument('--dataset', choices=['GRID'], default='GRID', help='')
    parser.add_argument('--model-name', choices=['simul_lr'], default='simul_lr', help='')
    parser.add_argument('--imgSize', default=224, type=int)
    parser.add_argument('--pretrain-dim', type=int, default=300, help='')
    parser.add_argument('--train-data', type=str, default='/home1/lihaoyuan/data/CV/PollutionSeg/data/train_list.json', help='')
    parser.add_argument('--val-data', type=str, default='/home1/lihaoyuan/data/CV/PollutionSeg/data/val_list.json', help='')
    parser.add_argument('--test-data', type=str, default='/home1/lihaoyuan/data/CV/PollutionSeg/data/test_data/', help='')
    parser.add_argument('--label-path', type=str, default='/home1/lihaoyuan/data/CV/PollutionSeg/data/id2cate.json', help='')
    parser.add_argument('--data-rate', type=float, default=0.3, help='')
    parser.add_argument('--cls-num', default=4, type=int)
    # parser.add_argument('--word2vec-path', type=str, default='glove_model.bin',help='')
    # parser.add_argument('--feature-path', type=str, default='data/activity-c3d',help='')
    # parser.add_argument('--max-num-words', type=int, default=20, help='')
    # parser.add_argument('--max-num-nodes', type=int, default=20,  help='')
    parser.add_argument('--d-model', type=int, default=512, help='')
    parser.add_argument('--num-heads', type=int, default=8, help='')
    parser.add_argument('--num-layers', type=int, default=6, help='')
    parser.add_argument('--batch-size', type=int, default=128, help='')
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    parser.add_argument('--d-ff', type=int, default=512, help='')
    parser.add_argument('--ffn-layer', type=str, default='fc', help='')
    parser.add_argument('--first-kernel-size', type=int, default=1, help='')
    # parser.add_argument('--word-dim', type=int, default=300, help='')
    # parser.add_argument('--frame-dim', type=int, default=500, help='')
    parser.add_argument('--num-gcn-layers', type=int, default=2, help='')
    # parser.add_argument('--num-attn-layers', type=int, default=2, help='')
    parser.add_argument('--display-n-batches', type=int, default=200, help='')
    parser.add_argument('--max-num-epochs', type=int, default=150, help='')
    parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD', help='weight decay')
    parser.add_argument('--lr-scheduler', default='cosine', choices=LR_SCHEDULER_REGISTRY.keys(), help='Learning Rate Scheduler')  # 'fixed', 'inverse_sqrt', 'inverse_linear', 'triangular', 'reduce_lr_on_plateau', 'cosine'
    # parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--lr', default=[1e-4, 2e-4], type=list, help='learning rate')  # interval
    parser.add_argument('--max-update', type=int, default=100, help='')
    parser.add_argument('--lr-shrink', type=float, default=0.999, help='')
    # self-training arguments
    parser.add_argument('--predict-steps', default=3, type=int, help='predict steps k')
    parser.add_argument('--n-negatives', default=8, type=int, help='number of negative samples')
    parser.add_argument('--batch-shuffle', action='store_true', help='whether to sample negatives from shuffled batches')
    # after self-training training main task
    parser.add_argument('--main-task', action='store_true', help='whether to train main task')

    # lipbert arguments
    # parser.add_argument('--mask-width', default=3, type=int, help='mask continuous width')
    # parser.add_argument('--mask-rate', default=0.15, type=float,  help='mask rate')
    # parser.add_argument('--anchor_width', default=3.0, type=float,   help='new_model3 anchor width')
    # from optimizer.lr_scheduler.inverse_square_root_schedule import InverseSquareRootSchedule
    # InverseSquareRootSchedule.add_args(parser)
    from optimizer.lr_scheduler.cosine_lr_scheduler import CosineSchedule
    CosineSchedule.add_args(parser)

    from optimizer.adam_optimizer import AdamOptimizer
    AdamOptimizer.add_args(parser)
    return parser.parse_args()


def main(args):
    args.train = 1
    # args.train_and_eval = 1
    # args.partial_load = 1
    # args.load_pretrained_model = 1
    # args.evaluate = 1

    print(args, end='\n\n')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.g)

    from runner_PS import Runner
    runner = Runner(args)
    if args.train:
        runner.train()
    if args.evaluate:
        runner.eval(args.dec)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args = parse_args()
    torch.autograd.set_detect_anomaly(True)
    main(args)
