import numpy as np
import torch
from torch import nn
import clip_ori.clip as clip
from ipdb import set_trace


class CustomCLIP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.clip_model ,_ = clip.load(args.pretrain, device=args.device, context_length=args.context_length)
        self.logit_scale = self.clip_model.logit_scale

    def forward(self, images, texts):
        image_features, text_features, logit_scale = self.clip_model(images, texts)
        return image_features, text_features, logit_scale