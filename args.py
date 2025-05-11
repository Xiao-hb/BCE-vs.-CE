import os
import shutil
import datetime
import argparse

import torch
import numpy as np


def parse_train_args():
    parser = argparse.ArgumentParser()

    # parameters
    # Model Selection
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--no-bias', dest='bias', action='store_false')
    parser.add_argument('--ETF_fc', dest='ETF_fc', action='store_true')
    parser.add_argument('--fixdim', dest='fixdim', type=int, default=0)
    parser.add_argument('--SOTA', dest='SOTA', action='store_true')
    
    # MLP settings (only when using mlp and res_adapt(in which case only width has effect))
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=6)

    # Vision Transformer settings
    parser.add_argument('--img_size', type=int, default=32, help="input image size")
    parser.add_argument('--patch_size', default=4, type=int, help="patch for ViT")
    parser.add_argument('--head_dim', default=512, type=int)

    # Hardware Setting
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=6)
    parser.add_argument('--use_cudnn', type=bool, default=True)

    # Directory Setting
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'cifar100', 'cifar10_random'], default='mnist')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--uid', type=str, default=None)
    parser.add_argument('--force', action='store_true', help='force to override the given uid')

    # Learning Options
    parser.add_argument('--epochs', type=int, default=100, help='Max Epochs')
    parser.add_argument('--batch_size', type=int, default=128, choices = [16, 32, 64, 128, 256, 512, 1024, 2048], help='Batch size')
    parser.add_argument('--loss', type=str, default='CrossEntropy', help='loss function configuration')
    parser.add_argument('--sample_size', type=int, default=None, help='sample size PER CLASS')
    parser.add_argument('--epoch_save_step', type=int, default=5, help='epoch step to save the model parameters')


    # Optimization specifications
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--patience', type=int, default=30, help='learning rate decay per N epochs')
    parser.add_argument('--decay_type', type=str, default='cosine', help='learning rate decay type')
    parser.add_argument('--epochs_warmup', type=int, default=0, help='warmup epochs for cosine decay type')
    parser.add_argument('--gamma', type=float, default=0.2, help='learning rate decay factor for step decay')
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'AdamW'], help='optimizer to use')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--weight_decay_bias', type=float, default=0e-4, help='weight decay for the bias of the last full connection layer')
    # The following two should be specified when testing adding wd on Features
    parser.add_argument('--sep_decay', action='store_true', help='whether to separate weight decay to last feature and last weights')
    parser.add_argument('--feature_decay_rate', type=float, default=5e-4, help='weight decay for last layer feature')
    parser.add_argument('--history_size', type=int, default=10, help='history size for LBFGS')
    parser.add_argument('--ghost_batch', type=int, dest='ghost_batch', default=128, help='ghost size for LBFGS variants')

    parser.add_argument('--bias_init_mode', default = 0, type = int,
                        choices = [0, 1, 2, 3, 4, 5, 6, 7, -1],
                        help = 'the bias mode')
    parser.add_argument('--bias_init_mean', default = 0., type = int,
                        choices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, -1],
                        help = 'the bias initialized mean')

    parser.add_argument('--smoothing', type=float, default=0.0, help='label smoothing rate in the loss')
    # * Mixup params
    parser.add_argument('--MIX', dest='MIX', action='store_true')
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    args = parser.parse_args()

    if args.uid is None:
        unique_id = str(np.random.randint(0, 100000))
        print("revise the unique id to a random number " + str(unique_id))
        args.uid = unique_id
    else:
        args.uid = str(args.uid)

    args = parser.parse_args()

    if args.use_cudnn:
        print("cudnn is used")
        torch.backends.cudnn.benchmark = True
    else:
        print("cudnn is not used")
        torch.backends.cudnn.benchmark = False

    return args


def parse_eval_args():
    parser = argparse.ArgumentParser()

    # parameters
    # Model Selection
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--no-bias', dest='bias', action='store_false')
    parser.add_argument('--ETF_fc', dest='ETF_fc', action='store_true')
    parser.add_argument('--fixdim', dest='fixdim', type=int, default=0)
    parser.add_argument('--SOTA', dest='SOTA', action='store_true')
    
    # MLP settings (only when using mlp and res_adapt(in which case only width has effect))
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=6)

    # Vision Transformer settings
    parser.add_argument('--img_size', type=int, default=32, help="input image size")
    parser.add_argument('--patch_size', default=4, type=int, help="patch for ViT")
    parser.add_argument('--head_dim', default=512, type=int)

    # Hardware Setting
    parser.add_argument('--gpu_id', type=int, default=0)

    # Directory Setting
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'cifar100', 'cifar10_random'], default='mnist')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--optimizer', default='SGD', help='optimizer to use')
    parser.add_argument('--loss', default='CE', help='loss to use')

    # Learning Options
    parser.add_argument('--epochs', type=int, default=100, help='Max Epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--sample_size', type=int, default=None, help='sample size PER CLASS')

    args = parser.parse_args()

    return args
