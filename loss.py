import numpy as np
import torch
import torch.nn as nn


def get_loss(args, images, texts, model):
    #1.前向传播,计算相似度矩阵
    image_features, text_features, logit_scale = model(images, texts)
    logit_scale = logit_scale.mean()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logit_scale * text_features @ image_features.t()

    #2.计算itc损失
    ground_truth = torch.arange(len(logits_per_image)).long().cuda(non_blocking=True)  #torch.Size([32])
    CE_loss = nn.CrossEntropyLoss().cuda()
    itc_loss = (CE_loss(logits_per_image, ground_truth)+ CE_loss(logits_per_text, ground_truth)) / 2

    return itc_loss