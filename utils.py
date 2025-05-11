import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn.functional as F
from scheduler.cosine_lr import CosineLRScheduler

def set_seed(manualSeed=666):
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(manualSeed)

def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())
    
    if args.sep_decay:
        wd_term = 0
    else:
        wd_term = args.weight_decay

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9,
                  'lr': args.lr,
                  'weight_decay': wd_term#args.weight_decay
        }
    elif args.optimizer == 'Adam':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'lr': args.lr,
            'weight_decay': wd_term#args.weight_decay
        }
    elif args.optimizer == 'AdamW' or args.optimizer == 'adamw':
        optimizer_function = optim.AdamW
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'lr': args.lr,
            'weight_decay': wd_term#args.weight_decay
        }
    elif args.optimizer == 'LBFGS':
        optimizer_function = optim.LBFGS
        kwargs = {'lr': args.lr,
                  'history_size': args.history_size,
                  'line_search_fn': 'strong_wolfe'
        }

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            #step_size=args.patience,
            #milestones=[60, 120, 160],
            milestones = [30, 60, 80],
            gamma=args.gamma
        )
    elif args.decay_type == 'cosine':
        scheduler = CosineLRScheduler(
            my_optimizer,
            t_initial = args.epochs,
            t_mul=getattr(args, 'lr_cycle_mul', 1.),
            lr_min=getattr(args, 'min_lr', 1e-5),
            decay_rate=getattr(args, 'decay_rate', 0.1),
            warmup_lr_init=getattr(args, 'warmup_lr', 1e-6),
            warmup_t=getattr(args, 'epochs_warmup', 10),
            cycle_limit=getattr(args, 'lr_cycle_limit', 1),
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=getattr(args, 'lr_noise_pct', 0.67),
            noise_std=getattr(args, 'lr_noise_std', 1.),
            noise_seed=getattr(args, 'seed', 42),
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler


def make_criterion(args):
    if args.loss.lower() == 'crossentropy' or args.loss.lower() == 'ce':
        #criterion = nn.CrossEntropyLoss()
        criterion = LabelSmoothingCrossEntropy(smoothing = args.smoothing)
    elif args.loss.lower() == 'bce':
        criterion = BCE_Loss(smoothing = args.smoothing)
    elif args.loss.lower() == 'mse':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError
    return criterion


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_network_parameters(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in parameters])


def print_and_save(text_str, file_stream):
    print(text_str)
    print(text_str, file=file_stream)


def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



class BCE_Loss(nn.Module):
    def __init__(self, l = 1, smoothing = 0.1):
        super(BCE_Loss, self).__init__()
        self.s_max = 256
        self.lam = l
        self.confidence = 1. - smoothing
        self.smoothing = smoothing
        print('BCE loss in neural collapse')

    def forward(self, input, targets):
        p_loss = torch.log(1 + torch.exp(-input.clamp(min=-self.s_max, max=self.s_max)))
        n_loss = torch.log(1 + torch.exp(input.clamp(min=-self.s_max, max=self.s_max))) * self.lam
        batchsize, num_classes = input.size()
        if targets.size() == p_loss.size():
            loss = targets * p_loss + (1. - targets) * n_loss
        else:
            one_hot = torch.zeros((batchsize, num_classes), dtype=torch.bool, device = targets.device)
            one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
            #loss = one_hot * p_loss + (~one_hot) * n_loss
            loss = self.confidence * (one_hot * p_loss + (~one_hot) * n_loss) + self.smoothing * (p_loss + (num_classes - 1) * n_loss) / num_classes
        #loss /= num_classes
        return loss.sum(dim=1).mean()

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim = -1)
        if target.size() == logprobs.size():
            loss = torch.sum(-target * logprobs, dim = -1)
            loss = loss.mean()
        else:
            #print(f'logprobs size is {logprobs.size()}')
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            loss = loss.mean()
        return loss


