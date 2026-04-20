from ipdb import set_trace
import json as jsonmod
import logging
import os
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from clip_ori.clip import tokenize


def _convert_to_rgb(image):
    return image.convert('RGB')


def build_transform(resolution=224):
    return Compose([
        Resize(resolution, interpolation=Image.BICUBIC),  # 扩大分辨率
        CenterCrop(resolution),  # 中心裁剪
        _convert_to_rgb,  # 转换成RGB三通道
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class JsonDataset(Dataset):
    def __init__(self, args, json_filename, img_filename):
        self.root = img_filename
        self.transform = build_transform()

        # 直接读入扁平化结构
        with open(json_filename, 'r', encoding='utf-8') as f:
            self.data_dict = jsonmod.load(f)
        # 提取所有 (filename, caption_id) 对
        self.ids = []
        for k in self.data_dict.keys():
            fname, cap_id_str = k.split('#')
            self.ids.append((fname, int(cap_id_str)))
        self.length = len(self.ids)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        fname, cap_id = self.ids[idx]
        image_path = os.path.join(self.root, fname)
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        # 获取对应的 caption
        key = f"{fname}#{cap_id}"
        caption = self.data_dict[key]
        text = tokenize([str(caption)], truncate=True)
        return image, text


# 定义dataset和dataload
def get_dataset(args, is_train, split=None):
    if split == "train":
        input_json = args.data_json + 'train.json'
    elif split == "test":
        input_json = args.data_json + 'test.json'

    img_filename = args.data_img
    dataset = JsonDataset(args, input_json, img_filename)

    logging.info("    " + split + "_data: " + str(len(dataset)))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=is_train,
                            num_workers=args.num_workers, pin_memory=True, drop_last=is_train)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader


# 将训练的验证的数据传入
def get_data(args):
    logging.info("Dataset:")
    data = {}
    data["train"] = get_dataset(args, is_train=True, split="train")
    data["test"] = get_dataset(args, is_train=False, split="test")
    data["val"] = data["test"]
    return data
