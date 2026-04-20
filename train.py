from ipdb import set_trace
import logging
import os
import time

import numpy as np
import torch
from torch import optim
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler

from config import parse_args
from model import CustomCLIP
from train_process import train, eval

import clip_ori.clip as clip
from clip_ori.model import convert_weights
from clip_ori.scheduler import cosine_lr
from clip_ori.utils import setup_logging,seed_torch,convert_models_to_fp32,logging_params


### 总训练过程
def main(args):
    ##1.配置
    # 配置当前实验的日志文件夹
    if args.exp_name is None:
        args.exp_name = time.strftime(f"%Y-%m-%d-%H-%M-%S",time.localtime())
    args.log_file = os.path.join(args.train_log, args.exp_name)
    os.makedirs(args.log_file, exist_ok=True)
    # 配置日志
    args.log_path = os.path.join(args.log_file, "log.log")
    args.log_level = logging.INFO
    logging.getLogger().handlers.clear()
    setup_logging(args.log_path, args.log_level)
    # 其他配置
    seed_torch(seed=13)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)

    ##2.实例化
    # 将超参数输出日志
    logging_params(args)
    # 数据集
    data = get_data(args)
    # 模型
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CustomCLIP(args)
    if args.precision == "amp" or args.precision == "fp32":
        convert_models_to_fp32(model)
    if args.precision == "fp16":
        convert_weights(model)
    if not torch.cuda.is_available():
        model.float()
        logging.warning("using CPU, this will be slow")
    # 优化器和学习率策略
    exclude = lambda n : "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n
    include = lambda n : not exclude(n)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]
    optimizer = optim.AdamW([
                                    {"params": gain_or_bias_params, "weight_decay": 0.},
                                    {"params": rest_params, "weight_decay": args.wd},
                                    ],
                            lr=args.lr,
                            betas=(args.beta1, args.beta2),
                            eps=args.eps,
                           )
    total_steps = data["train"].num_batches * args.epoch_num
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    scaler = GradScaler() if args.precision == "amp" else None

    ##3.zeroshot、train、test
    logging.info("Start:")
    train_start_time = time.time()
    cudnn.benchmark = True
    cudnn.deterministic = False
    start_epoch = 0
    val_best_rsum, val_best_epoch = 0, 0
    test_best_rsum, test_best_epoch = 0, 0
    for epoch in range(start_epoch, args.epoch_num):
        logging.info(f"Epoch {epoch}: "+"="*100)
        if epoch == start_epoch:
            logging.info("<Zero-shot Val&Test>")
            val_best_rsum, val_best_epoch = eval(args,data,"val",model,optimizer,epoch,val_best_rsum,val_best_epoch)
            test_best_rsum, test_best_epoch = eval(args,data,"test",model,optimizer,epoch,test_best_rsum,test_best_epoch)
        logging.info("<Train>")
        train(args,data,model,optimizer,epoch,scheduler,scaler)
        logging.info("<Val&Test>")
        val_best_rsum, val_best_epoch = eval(args,data,"val",model,optimizer,epoch,val_best_rsum,val_best_epoch)
        test_best_rsum, test_best_epoch = eval(args,data,"test",model,optimizer,epoch,test_best_rsum,test_best_epoch)
    print("All_time: ", time.time()-train_start_time)

    final_log_file = f"{args.log_file}_{test_best_rsum:.3f}"
    os.rename(args.log_file, final_log_file)


if __name__ == "__main__":
    args = parse_args()

    # RSITR
    # from data_RSITR import get_data
    # args.data_json = "/data2/ly/data/igarss/RSITMD/dataset_RSITMD.json"
    # args.data_img = "/data2/ly/data/igarss/RSITMD/images/"
    # args.data_json = "/data2/ly/data/igarss/RSICD/dataset_rsicd.json"
    # args.data_img = "/data2/ly/data/igarss/RSICD/RSICD_images/"
    # args.data_json = "/data2/ly/data/igarss/UCM/dataset.json"
    # args.data_img = "/data2/ly/data/igarss/UCM/UCM_imgs/"
    # # CMITR
    from data_CMITR import get_data
    args.data_json = "/data1/lyh/clip_qie/Datas/flickr30k/"
    args.data_img = "/data3/ycb/Files/Datasets/Flickr30K/flickr30k-images"
    # args.data_json = "/data1/lyh/clip_qie/Datas/coco/"
    # args.data_img = "/data3/ycb/Files/Datasets/COCO2014"

    args.lr = 1.0e-5

    main(args)
