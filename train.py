import datetime
import sys

import torch

import models
from models.resnet import resnet18, resnet50
from models import vit, swin
from utils import *
from args import parse_train_args
from datasets import make_dataset
from mixup import Mixup

def loss_compute(args, model, criterion, outputs, targets):
    if args.loss.lower() == 'crossentropy' or args.loss.lower() == 'ce':
        loss = criterion(outputs[0], targets)
    elif args.loss.lower() == 'bce':
        loss = criterion(outputs[0], targets)
    elif args.loss.lower() == 'mse':
        loss = criterion(outputs[0], nn.functional.one_hot(targets).type(torch.FloatTensor).to(args.device))
    else:
        raise NotImplementedError

    # Now decide whether to add weight decay on last weights and last features
    if args.sep_decay:
        # Find features and weights
        features = outputs[1]
        w = model.fc.weight
        b = model.fc.bias
        lamb_weight = args.weight_decay / 2
        lamb_bias = args.weight_decay_bias / 2
        lamb_feature = args.feature_decay_rate / 2
        if b is not None:
            loss += lamb_weight * torch.sum(w ** 2) + lamb_bias * torch.sum(b ** 2) + lamb_feature * torch.sum(features ** 2)
        else:
            loss += lamb_weight * torch.sum(w ** 2) + lamb_feature * torch.sum(features ** 2)

    return loss

def trainer(args, model, trainloader, epoch_id, criterion, optimizer, scheduler, logfile, mixup_fn):

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print_and_save('\nTraining Epoch: [%d | %d] LR: %f -- %s -- %s' % (epoch_id + 1, args.epochs, optimizer.param_groups[0]["lr"], args.loss, args.decay_type), logfile)
    model.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)
        if mixup_fn is not None:
            inputs, targets_for_loss = mixup_fn(inputs, targets)
        else:
            targets_for_loss = targets

        outputs = model(inputs)
        
        if args.sep_decay:
            loss = loss_compute(args, model, criterion, outputs, targets_for_loss)
        else:
            if args.loss.lower() == 'crossentropy' or args.loss.lower() == 'ce':
                loss = criterion(outputs[0], targets_for_loss)
            elif args.loss.lower() == 'bce':
                loss = criterion(outputs[0], targets_for_loss)
            else:
                raise NotImplementedError

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        # model.eval()
        # outputs = model(inputs)
        prec1, prec5 = compute_accuracy(outputs[0].detach().data, targets.detach().data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if batch_idx % 10 == 0:
            time_now = datetime.datetime.now()
            print_and_save('[time: %s] | [epoch: %d] (%d/%d) | Loss: %.4f | top1: %.4f | top5: %.4f ' %
                           (time_now.strftime("%Y-%m-%d_%H-%M-%S"), epoch_id + 1, batch_idx + 1, len(trainloader), losses.avg, top1.avg, top5.avg), logfile)

            last_layer = model.fc
            if hasattr(last_layer, 'bias') and last_layer.bias is not None:
                print("Bias of the last layer:", last_layer.bias.data)
            else:
                print("The last layer does not have a bias term.")

    scheduler.step(epoch = epoch_id)

def tester(args, model, testloader, epoch_id, logfile, best_top1):

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print_and_save('\nTesting Epoch: [%d | %d] -- %s' % (epoch_id + 1, args.epochs, args.loss), logfile)
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(testloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        outputs = model(inputs)

        prec1, prec5 = compute_accuracy(outputs[0].detach().data, targets.detach().data, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        if batch_idx % 10 == 0:
            time_now = datetime.datetime.now()
            print_and_save('[time: %s] | [epoch: %d] (%d/%d) | top1: %.4f | top5: %.4f |' %
                           (time_now.strftime("%Y-%m-%d_%H-%M-%S"), epoch_id + 1, batch_idx + 1, len(testloader), top1.avg, top5.avg), logfile)

    time_now = datetime.datetime.now()
    if best_top1 < top1.avg:
        best_top1 = top1.avg
    print_and_save('[time: %s] | top1: %.4f | top5: %.4f | best_top1: %.4f' %
                   (time_now.strftime("%Y-%m-%d_%H-%M-%S"), top1.avg, top5.avg, best_top1), logfile)
    return best_top1



def train(args, model, trainloader, testloader, epoch_save_step = 1):

    criterion = make_criterion(args)
    args.lr  = args.batch_size / 128 * args.lr
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)

    logfile = open('%s/train_log.txt' % (args.save_path), 'w')
    print_and_save('# of model parameters: ' + str(count_network_parameters(model)), logfile)
    if 'cct' in args.model.lower():
        print_and_save('# the init bias of the model: {}'.format(model.classifier.fc.bias), logfile)
    else:
        print_and_save('# the init bias of the model: {}'.format(model.fc.bias), logfile)
    print_and_save('# the bias_init_mode: {}'.format(args.bias_init_mode), logfile)
    print_and_save('# the bias_init_mean: {}'.format(args.bias_init_mean), logfile)
    print_and_save('--------------------- Training -------------------------------', logfile)


    mixup_fn = None
    mixup_active = args.MIX
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    best_top1 = 0.
    for epoch_id in range(args.epochs):
        trainer(args, model, trainloader, epoch_id, criterion, optimizer, scheduler, logfile, mixup_fn)
        if (epoch_id + 1) % epoch_save_step == 0:
            torch.save(model.state_dict(), args.save_path + "/epoch_" + str(epoch_id + 1).zfill(3) + ".pth")
        #if epoch_id % 1 == 0:
        test_top1 = tester(args, model, testloader, epoch_id, logfile, best_top1)
        if test_top1 > best_top1:
            best_top1 = test_top1
            torch.save(model.state_dict(), args.save_path + "/best_model.pth")
    logfile.close()


def main():
    args = parse_train_args()

    if args.SOTA:
        args.bias_init_mode = 0
        if args.loss.lower() == 'ce' or args.loss.lower() == 'crossentropy':
            args.bias_init_mean = 0
        elif args.loss.lower() == 'bce':
            args.bias_init_mean = 6
        else:
            raise NotImplementedError

        args.sep_decay = False
        if args.optimizer.lower() == 'sgd':
            args.lr = 0.01
            args.weight_decay = 0.0005
            args.decay_type = 'step'
        elif args.optimizer.lower() == 'adamw':
            args.lr = 0.001
            args.weight_decay = 0.05
            args.decay_type = 'cosine'
        else:
            raise NotImplementedError
        if args.MIX:
            if args.smoothing == 0.0:
                args.smoothing = 0.1

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.SOTA:
        save_path = ('./model_weights/' + args.model + '_' + args.dataset + '_' + args.loss + '_' + args.optimizer + '_' + args.decay_type +
                     '_lr_' + str(args.lr) + '_smoothing_' + str(args.smoothing) + '_' + str(args.SOTA) + '_' + str(args.MIX) + '_' + timestamp)
    else:
        save_path = ('./model_weights/' + args.model + '_' + args.dataset + '_' + args.loss + '_' + args.optimizer + '_'
                     + args.decay_type + '_batch_size_'+ str(args.batch_size) + '_bias_mean_' + str(args.bias_init_mean) +
                     '_bias_decay_' + str(args.weight_decay_bias) + '_' + timestamp)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    args.save_path = save_path
    args.log = os.path.join(save_path, 'log.txt')
    args.arg = os.path.join(save_path, 'args.txt')

    with open(args.log, 'w') as f:
        f.close()
    with open(args.arg, 'w') as f:
        print(args)
        print(args, file=f)
        f.close()

    set_seed(manualSeed = args.seed)

    if args.optimizer == 'LBFGS':
        sys.exit('Support for training with 1st order methods!')

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    args.device = device

    trainloader, testloader, num_classes = make_dataset(args.dataset, args.data_dir, args.batch_size, args.sample_size, SOTA=args.SOTA)
    args.nb_classes = num_classes
    if args.model == "MLP":
        model = models.__dict__[args.model](hidden = args.width, depth = args.depth, fc_bias=args.bias, num_classes=num_classes).to(device)
    elif args.model == "ResNet18":
        model = resnet18(num_classes=num_classes, bias_init_mode=args.bias_init_mode, bias_init_mean=args.bias_init_mean).to(device)
    elif args.model == "ResNet50":
        model = resnet50(num_classes=num_classes, bias_init_mode=args.bias_init_mode, bias_init_mean=args.bias_init_mean).to(device)
    elif args.model == 'densenet121':
        model = models.densenet121(num_classes = num_classes, bias_init_mode = args.bias_init_mode, bias_init_mean = args.bias_init_mean).to(device)
    elif args.model == 'ViT':
        print('Using ViT! ')
        model = vit.ViT(image_size=args.img_size, patch_size=args.patch_size, dim=args.head_dim,
                        num_classes=num_classes,
                        bias_init_mode=args.bias_init_mode, bias_init_mean=args.bias_init_mean,
                        depth=6, heads=10, mlp_dim=512, dropout=0.2, emb_dropout=0.1).to(device)
    elif args.model == 'Swin':
        print('Using Swin-Transformer! ')
        model = swin.swin_t(window_size=args.patch_size, downscaling_factors=(2, 2, 2, 1),
                            num_classes=num_classes,
                            bias_init_mode=args.bias_init_mode, bias_init_mean=args.bias_init_mean).to(device)
    else:
        model = models.__dict__[args.model](num_classes=num_classes, fc_bias=args.bias, ETF_fc=args.ETF_fc, fixdim=args.fixdim, SOTA=args.SOTA,
                                            bias_init_mode = args.bias_init_mode, bias_init_mean = args.bias_init_mean).to(device)

    train(args, model, trainloader, testloader, epoch_save_step = args.epoch_save_step)


if __name__ == "__main__":
    main()
