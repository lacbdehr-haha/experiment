# coding=utf-8
from __future__ import print_function
from __future__ import division
import argparse
import torch.nn as nn
from code.distributions import log_normal_diag
from code.original.original_net import Led3Dnet_soft, FC, con_layer
from code.utils import standard_normal_logprob, get_logger
from code.Pointflowmodel.flow import get_latent_cnf
import os, shutil
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import time
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader_scu import SCU_loader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Led3Dnet_soft')

# /sort:normal images aug(pose generating, shape jittering, and shape scaling)
parser.add_argument('--root_path', default='/home/sanggaoli/znn_code/data/file/aug', type=str,
                    help='path to root path of images')
parser.add_argument('--modelpath', default='3Dpretrain/checkpoint_15.pth', type=str,
                    help='path to root path of images')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--num_class', default=338, type=int,
                    help='number of people(class) (default: 10572)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--step_size', type=list, default=[1500, 10000, 20000], metavar='SS',
                    help='lr decay step (default: [10,15,18])')  # [15000, 22000, 26000]
# arguments of networks
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0.0005,
                    metavar='W', help='weight decay (default: 0.0005)')

parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_path', default='finetune_ckpt_ODE/', type=str, metavar='PATH',
                    help='path to save checkpoint')
parser.add_argument('--workers', type=int, default=6, metavar='N',
                    help='how many workers to load data')
# arguments of flow
parser.add_argument('--zdim', type=int, default=960, metavar='N')   # 960
parser.add_argument('--context_dim', type=int, default=10, metavar='N')
parser.add_argument('--latent_num_blocks', type=int, default=1, metavar='N')
parser.add_argument('--num_blocks', type=int, default=1, metavar='N')
parser.add_argument('--latent_dims', type=str, default="960-960", metavar='N')
parser.add_argument('--layer_type', type=str, default="concat", metavar='N')
parser.add_argument('--nonlinearity', type=str, default="softplus", metavar='N')
parser.add_argument('--time_length', type=float, default=0.5, metavar='N')
parser.add_argument('--train_T', type=int, default=True, metavar='N')
parser.add_argument('--batch_norm', type=int, default=True, metavar='N')
parser.add_argument('--bn_lag', type=int, default=0, metavar='N')
parser.add_argument('--sync_bn', type=int, default=False, metavar='N')
args = parser.parse_args()

configurations1 = {
    1: dict(
        batch_size=256,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        epochs=50,
        lr=0.00001,
        exp_dir='./exp/',
        model_dir='../model/',
        pretrained='',
        print_freq=100,
        resume='',
        save_freq=1,
        start_epoch=0,
        # base_data(训练集474，测试集中200)
        root_dir='../data/original/base_data/',
        train_list='../data/original/file/train.txt',
        workers=8,
    )
}
cfg = configurations1[1]
logger = get_logger('original/training_log.log')

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    full_filename = os.path.join(cfg['model_dir'], filename)
    full_bestname = os.path.join(cfg['model_dir'], 'model_best.pth.tar')
    torch.save(state, full_filename)
    epoch_num = state['epoch']
    if epoch_num % cfg['save_freq'] == 0:
        torch.save(state, full_filename.replace('checkpoint', 'checkpoint_'+str(epoch_num)))
    if is_best:
        shutil.copyfile(full_filename, full_bestname)


def main():
    cfg = configurations1[1]
    # --------------------------------------model----------------------------------------
    # feature extract
    model = Led3Dnet_soft().to(cfg['device'])

    # CNF
    flow = get_latent_cnf(args).to(cfg['device'])

    # 10 dimensional vector
    cond_layer = con_layer(context_dim=args.context_dim).to(cfg['device'])

    MCP = FC(num_class=474).to(cfg['device'])

    # ------------------------------------load image---------------------------------------
    transform_train = transforms.Compose([
                      transforms.ToPILImage(),
                      transforms.Resize((128, 128)),
                      transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                      transforms.Normalize(mean=(0.5), std=(0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
                  ])

    # load train data
    train_dataset = SCU_loader(cfg['root_dir'], cfg['train_list'], transform_train)
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, pin_memory=True,
                              num_workers=cfg['workers'], drop_last=True)
    num_classes = train_dataset.class_num()

    print('length of train Dataset: ' + str(len(train_loader.dataset)))
    print('Number of Classses: ' + str(num_classes))
    criterion = nn.CrossEntropyLoss().cuda()
    params = [{'params': model.parameters()},
              {'params': flow.parameters()},
              {'params':cond_layer.parameters()},
              {'params':MCP.parameters()},
              ]
    optimizer = torch.optim.Adam(params, lr=cfg['lr'], weight_decay=0.00005)  # Adam梯度下降
    # ----------------------------------------train----------------------------------------
    for epoch in range(1, args.epochs + 1):

        # scheduler.step()
        train(train_loader, model, criterion, optimizer, epoch, MCP, flow, cond_layer)
        if epoch % 2 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'led': model.state_dict(),
                'metric': MCP.state_dict(),
                'flow': flow.state_dict(),
                'cond_layer': cond_layer.state_dict(),
                'optimizer': optimizer.state_dict()
            }, False)


def train(train_loader, model, criterion, optimizer, epoch, MCP, flow, cond_layer):
    model.train()
    MCP.train()
    cond_layer.train()
    flow.train()
    time_curr = time.time()
    loss_display, totalloss_display, kl = 0, 0, 0
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        iteration = (epoch - 1) * len(train_loader) + batch_idx
        if iteration % 3000 == 0:
            schedule_lr(optimizer)
            print(optimizer)
        data, target = data.cuda(), target.cuda()
        feat, sample, mean, var = model(data, std=True)

        cond_feat = cond_layer(feat).view(feat.size(0), -1)

        entropy = log_normal_diag(sample, mean=mean, log_var=var.log(), average=False, dim=-1)
        # x,logp(x)
        w, delta_log_pw = flow(sample, cond_feat, torch.zeros(mean.size(0), 1).to(mean))
        log_pw = standard_normal_logprob(w).sum(-1, keepdim=True)
        delta_log_pw = delta_log_pw.sum(-1, keepdim=True)
        log_pz = log_pw - delta_log_pw
        prior_loss = -log_pz.mean()
        entropy_loss = entropy.mean()
        # compute kl divergence
        kl_div = prior_loss + 1 * entropy_loss

        output = MCP(w)

        loss = criterion(output, target)
        totalloss = loss + 5e-3 * kl_div

        loss_display += loss.item()
        kl += kl_div.item()
        totalloss_display += totalloss.item()
        optimizer.zero_grad()
        totalloss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            time_used = time.time() - time_curr
            loss_display /= args.log_interval
            kl /= args.log_interval
            totalloss_display /= args.log_interval
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]{}, totalLoss:{:.5f}, Loss:{:.5f}, kl:{:.5f}, Time: {:.1f}s({} iters)'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    iteration, totalloss_display, loss_display, kl, time_used, args.log_interval))
            time_curr = time.time()
            loss_display, totalloss_display, kl = 0, 0, 0


def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.


if __name__ == '__main__':
    main()
