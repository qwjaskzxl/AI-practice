import argparse, os, random
import torch
from torch.utils.data import DataLoader
from mosei_dataset import Mosei_Dataset
from meld_dataset import Meld_Dataset
from model_LA import Model_LA

from train import train
import numpy as np
from utils.compute_args import compute_args
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--g', type=str, default="0")
    # 模型选择
    parser.add_argument('--model', type=str, default='Model_LA')
    parser.add_argument('--mode', type=str, choices=['LA', 'L', 'A'], default='LA')
    # 定义层数(block)
    parser.add_argument('--layer', type=int, default=4)
    # 定义隐藏层神经元个数
    parser.add_argument('--hidden_size', type=int, default=512)
    # 定义dropout
    parser.add_argument('--dropout_r', type=float, default=0.1)
    # 定义head个数
    parser.add_argument('--multi_head', type=int, default=8)
    parser.add_argument('--ff_size', type=int, default=2048)
    parser.add_argument('--word_embed_size', type=int, default=300)

    # Data
    parser.add_argument('--lang_seq_len', type=int, default=60)
    parser.add_argument('--audio_seq_len', type=int, default=60)

    parser.add_argument('--audio_feat_size', type=int, default=80)

    # Training
    parser.add_argument('--output', type=str, default='ckpt/')
    parser.add_argument('--name', type=str, default='exp4_/')
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--opt', type=str, default="Adam")
    parser.add_argument('--opt_params', type=str, default="{'betas': '(0.9, 0.98)', 'eps': '1e-9'}")
    parser.add_argument('--lr_base', type=float, default=1e-4)  # 3e-4
    parser.add_argument('--lr_decay', type=float, default=0.8)
    parser.add_argument('--lr_decay_times', type=int, default=20)
    parser.add_argument('--warmup_epoch', type=float, default=0)
    parser.add_argument('--grad_norm_clip', type=float, default=-1)
    parser.add_argument('--eval_start', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999999))

    # Dataset and task
    parser.add_argument('--dataset', type=str, choices=['MELD', 'MOSEI'], default='MOSEI')
    parser.add_argument('--task', type=str, choices=['sentiment', 'emotion'], default='sentiment')
    parser.add_argument('--task_binary', type=bool, default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Base on args given, compute new args
    args = compute_args(parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = args.g
    while 1:
        args.seed = random.randint(0, 999999999)
        args.seed  = 43774074
        print(args.seed)

        # Seed
        # 设置种子，使得模型的每次初始化固定，torch框架本身内部写法，不用过于关注纠结
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # DataLoader
        train_dset = eval(args.dataloader)('train', args)
        eval_dset = eval(args.dataloader)('valid', args, train_dset.token_to_ix)

        train_loader = DataLoader(train_dset, args.batch_size, shuffle=True, num_workers=10, pin_memory=True)
        eval_loader = DataLoader(eval_dset, args.batch_size, num_workers=10, pin_memory=True)

        # Net
        net = eval(args.model)(args, train_dset.vocab_size, train_dset.pretrained_emb).cuda()

        # net = eval(args.model)(args, train_dset.vocab_size, train_dset.pretrained_emb).cpu()
        print("Total number of parameters : " + str(sum([p.numel() for p in net.parameters()]) / 1e6) + "M")

        # Create Checkpoint dir
        if not os.path.exists(os.path.join(args.output, args.name)):
            os.makedirs(os.path.join(args.output, args.name))

        # Run training
        eval_accuracies = train(net, train_loader, eval_loader, args)
        exit()
