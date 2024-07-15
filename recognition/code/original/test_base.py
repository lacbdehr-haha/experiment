# coding=utf-8
import os, sys, shutil
from dataloader_scu import SCU_testloader
import torch
import torchvision.transforms as transforms
import numpy as np
from original_net import Led3Dnet_soft, con_layer
import argparse
from code.Pointflowmodel.flow import get_latent_cnf
from code.utils import get_logger


def get_top(probe, gallery_data):
    """从probe中提取图像，到gallery_data中寻找与其匹配分数最高的图像，得到其身份"""
    score_info = list()     # 将评分信息放在列表中
    probe_feat, probe_id = probe
    for path in gallery_data.keys():
        gallery_feat, gallery_id = gallery_data[path]

        probe_norm = np.linalg.norm(probe_feat)
        gallery_norm = np.linalg.norm(gallery_feat)
        score = np.dot(probe_feat, gallery_feat.T) / (probe_norm * gallery_norm)

        score_info.append((gallery_id, score))
    score_info = sorted(score_info, key=lambda a: a[1], reverse=True)
    top1_id = [item[0] for item in score_info[:1]]
    return top1_id


def eval_recog(probe_data, gallery_data):
    gallery_ids = set()
    for path in gallery_data.keys():
        gallery_ids.add(gallery_data[path][1])
    gallery_ids = list(gallery_ids)
    top1_num,  total_num = 0, 0
    for path in probe_data.keys():
        class_id = probe_data[path][1]
        if class_id not in gallery_ids:
            continue
        top1_id = get_top(probe_data[path], gallery_data)
        if class_id == top1_id[0]:
            top1_num += 1
        total_num += 1
    return top1_num / total_num, top1_num, total_num


@torch.no_grad()
def feature_extract(root_dir, model, protocl_file, flow, cond_layer):
    backbone.eval()

    TFS = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([128, 128]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    dataset = SCU_testloader(root_dir, protocl_file, TFS)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    feature_dict = dict()
    for j, (input, label, path) in enumerate(loader):
        input = input.to(torch.device(device))
        feat, mean = model(input, std=False)
        cond_feat = cond_layer(feat).view(feat.size(0), -1)
        w, _ = flow(mean, cond_feat, torch.zeros(mean.size(0), 1).to(mean))

        mean = w

        for index, m in enumerate(path):
            feature_dict[m] = (mean[index].cpu().data.numpy(), label[index].item())
    return feature_dict


def TestA(root_dir, backbone, gallery_file, nu_file, ps1_file, ps2_file, fe_file, flow, cond_layer):
    gallery_data = feature_extract(root_dir, backbone, gallery_file, flow, cond_layer)
    print("finish calc gallery_data")
    nu_data = feature_extract(root_dir, backbone, nu_file, flow, cond_layer)
    print("finish calc nu_data")
    ps1_data = feature_extract(root_dir, backbone, ps1_file, flow, cond_layer)
    print("finish calc ps1_data")
    ps2_data = feature_extract(root_dir, backbone, ps2_file, flow, cond_layer)
    print("finish calc ps2_data")
    fe_data = feature_extract(root_dir, backbone, fe_file, flow, cond_layer)
    print("finish calc fe_data")
    nu_acc, nu_top1, nu_total = eval_recog(nu_data, gallery_data)
    ps1_acc, ps1_top1, ps1_total = eval_recog(ps1_data, gallery_data)
    ps2_acc, ps2_top1, ps2_total = eval_recog(ps2_data, gallery_data)
    fe_acc, fe_top1, fe_total = eval_recog(fe_data, gallery_data)
    avg_acc = (nu_top1+ps1_top1+ps2_top1+fe_top1)/(nu_total+ps1_total+ps2_total+fe_total)
    logger.info('DatasetA_acc_nu:{}, fe:{}, ps1:{}, ps2:{}, avg:{}'.format(nu_acc, fe_acc, ps1_acc, ps2_acc, avg_acc))


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Led3Dnet_soft')

# DATA
parser.add_argument('--modelpath', default='3Dpretrain/checkpoint_15.pth', type=str,
                    help='path to root path of images')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--num_class', default=674, type=int,
                    help='number of people(class) (default: 10572)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--step_size', type=list, default=[1500, 10000, 20000], metavar='SS',
                    help='lr decay step (default: [10,15,18])')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0005,
                    metavar='W', help='weight decay (default: 0.0005)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_path', default='finetune_ckpt_ODE/', type=str, metavar='PATH',
                    help='path to save checkpoint')
parser.add_argument('--workers', type=int, default=6, metavar='N',
                    help='how many workers to load data')
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

if __name__ == '__main__':
    list_dir = '../data/original/file/'
    device = torch.device("cuda:0")
    gallery_A_file = list_dir + 'gallery_a.txt'
    nu_file = list_dir + 'brl_probe_nu.txt'
    ps1_file = list_dir + 'brl_probe_ps1.txt'
    ps2_file = list_dir + 'brl_probe_ps2.txt'
    fe_file = list_dir + 'brl_probe_fe.txt'
    logger = get_logger('./test_log.log')
    logger.info('start testing!')
    for m in range(1, 51, 2):
        model_dir = './model/checkpoint_' + str(m) + '.pth.tar'
        logger.info('----------------------------------------------')
        logger.info('model:{}'.format(model_dir))
        backbone = Led3Dnet_soft()
        backbone = backbone.to(device)
        checkpoint = torch.load(model_dir, device)
        backbone.load_state_dict(checkpoint['led'], strict=False)

        flow = get_latent_cnf(args).to(device)
        flow.load_state_dict(checkpoint['flow'], strict=False)
        cond_layer = con_layer(context_dim=args.context_dim).to(device)
        cond_layer.load_state_dict(checkpoint['cond_layer'], strict=False)

        root_dir = '../data/original/base_data/'
        TestA(root_dir, backbone, gallery_A_file, nu_file, ps1_file, ps2_file, fe_file, flow, cond_layer)