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
                    Resize(resolution, interpolation=Image.BICUBIC), #修改分辨率
                    CenterCrop(resolution),                          #中心裁剪
                    _convert_to_rgb,                                 #转换成RGB三通道
                    ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])

class JsonDataset(Dataset):
    def __init__(self, args, json_filename, img_filename, split="val"):
        json_dir = json_filename
        self.root = img_filename
        self.dataset = jsonmod.load(open(json_dir, 'r'))['images']
        self.transform = build_transform()

        self.ids = []
        for i, ds in enumerate(self.dataset):
            if ds['split'] == split:
                #五个句子
                if len(ds['sentences']) != 5:
                    logging.info(split)
                    logging.info(len(ds['sentences']))
                self.ids += [(i, x) for x in range(len(ds['sentences']))]

        self.split = split
        self.length = len(self.ids)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        root = self.root
        ann_id = self.ids[idx]
        img_id = ann_id[0]
        cap_id = ann_id[1]

        path = self.dataset[img_id]['filename']
        image = Image.open(os.path.join(root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        caption = self.dataset[img_id]['sentences'][cap_id]['raw']
        text = tokenize([str(caption)])
        return image, text

#定义dataset和dataload
def get_dataset(args, is_train, split=None):
    input_json = args.data_json
    img_filename = args.data_img
    dataset = JsonDataset(args, input_json, img_filename, split=split)

    logging.info("    "+split+"_data: "+str(len(dataset)))
    dataloader = DataLoader(dataset,batch_size=args.batch_size,shuffle=is_train,
                            num_workers=args.num_workers,pin_memory=True,drop_last=is_train)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader
    
#将训练的验证的数据传入
def get_data(args):
    logging.info("Dataset:")
    data = {}

    if args.data_json:
        data["train"] = get_dataset(args, is_train=True, split="train")
    # if args.val_data:
        data["val"] = get_dataset(args, is_train=False, split="val")
        data["test"] = get_dataset(args, is_train=False, split="test")
        if args.data_json.split("/")[-2] == "RSITMD":
            data["val"] = data["test"]

    return data
