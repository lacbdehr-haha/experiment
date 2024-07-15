import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_cipl(low_image_features, high_image_features, pid, epsilon=1e-8):
    logit_scale = torch.ones([]) * (1 / 0.02)
    batch_size = low_image_features.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()
    # ��ȡͼ������
    low_image_norm = low_image_features / low_image_features.norm(dim=1, keepdim=True)
    high_image_norm = high_image_features / high_image_features.norm(dim=1, keepdim=True)
    # �����������ƶ�
    h2l_cosine_theta = high_image_norm @ low_image_norm.t()
    l2h_cosine_theta = h2l_cosine_theta.t()
    # �Լ���õ����������ƶȽ������Ŵ���
    high_proj_low = logit_scale * h2l_cosine_theta
    low_proj_high = logit_scale * l2h_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    # ��low_proj_high����Softmax�õ�ÿ��Ԥ�����ĸ���ֵ
    l2h_pred = F.softmax(low_proj_high, dim=1)
    # KLɢ��, l2h_pred��Ԥ��ķֲ��������¶�ϵ������
    l2h_loss = l2h_pred * (F.log_softmax(low_proj_high, dim=1) - torch.log(labels_distribute + epsilon))    # KL divergence
    h2l_pred = F.softmax(high_proj_low, dim=1)
    h2l_loss = h2l_pred * (F.log_softmax(high_proj_low, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(l2h_loss, dim=1)) + torch.mean(torch.sum(h2l_loss, dim=1))

    return loss