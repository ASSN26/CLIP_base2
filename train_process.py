from ipdb import set_trace
import logging
import os
import time

import numpy as np
import torch
from torch.cuda.amp import autocast

from loss import get_loss

##训练
def train(args, data, model, optimizer, epoch, scheduler, scaler):
    dataloader = data["train"]
    num_batches_per_epoch = dataloader.num_batches
    model.train()

    batch_start_time = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        optimizer.zero_grad()

        images, texts = batch
        texts = texts.squeeze()
        images = images.cuda(non_blocking=True)
        texts = texts.cuda(non_blocking=True)
        data_time = time.time() - batch_start_time

        if args.precision == "amp":
            with autocast():#混合精度，提升运行速度
                itc_loss = get_loss(args, images, texts, model)
                total_loss = itc_loss
                scaler.scale(total_loss).backward()#提升精度，放大损失来防止梯度的下溢
                scaler.step(optimizer)
            scaler.update()
        else:
            itc_loss = get_loss(args, images, texts, model)
            total_loss = itc_loss
            total_loss.backward()
            optimizer.step()
        model_time = time.time() - batch_start_time - data_time
        batch_start_time = time.time()

        if i==0:
            logging.info(f"Train Epoch: {epoch} logit_scale {model.logit_scale.data:.3f}")
        if (i % (dataloader.num_samples//args.print_freq//args.batch_size)) == 0:   #日志输出至控制台
            num_samples = i * len(images)
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            logging.info(
                        f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                        f"Time_d,m: {data_time:.3f},{model_time:.3f}\t"
                        f"Lr: {optimizer.param_groups[0]['lr']:6f}\tLoss_itc: {itc_loss.item():.6f}"
                        )

#测试
def test(args, data, split, model, epoch):
    dataloader = data[split]
    model.eval()

    all_image_features, all_text_features = [], []
    with torch.no_grad():
        for i,batch in enumerate(dataloader):
            images, texts = batch
            texts = texts.squeeze()
            images = images.cuda(non_blocking=True)
            texts = texts.cuda(non_blocking=True)
            image_features, text_features, logit_scale = model(images, texts)

            all_image_features.append(image_features.detach())
            all_text_features.append(text_features.detach())
        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)
        all_image_features = all_image_features[::5]

        sim_i2t = all_image_features @ all_text_features.t()
        _, all_i2t_idx = torch.topk(sim_i2t, k=16, dim=1, largest=True, sorted=True)
        _, all_t2i_idx = torch.topk(sim_i2t.t(), k=16, dim=1, largest=True, sorted=True)
    return all_i2t_idx.cpu().numpy(), all_t2i_idx.cpu().numpy()

## 评估
def eval(args, data, split, model, optimizer, epoch, best_rsum, best_epoch):
    all_i2t_idx, all_t2i_idx = test(args, data, split, model, epoch)
    (ir1, ir5, ir10) = i2t5(all_i2t_idx)
    (tr1, tr5, tr10) = t2i5(all_t2i_idx)
    rsum = ir1+ ir5+ ir10+ tr1+ tr5+ tr10

    logging.info(f"{split:<5} Image to text: {ir1:.2f}, {ir5:.2f}, {ir10:.2f}")
    logging.info(f"{split:<5} Text to image: {tr1:.2f}, {tr5:.2f}, {tr10:.2f}")

    # remember best R@ sum and save checkpoint
    is_best = rsum > best_rsum
    # is_best = False
    if is_best :
        best_rsum = rsum
        best_epoch = epoch
        if split=="test" and args.save_ckpt==True:
            torch.save({
                            "name": args.exp_name,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            },
                       os.path.join(args.log_file, f"best.pt"),
                       )
    logging.info(f"{split:<5} Now_epoch , rsum: {epoch:2d}, {rsum:.3f}")
    logging.info(f"{split:<5} Best_epoch, rsum: {best_epoch:2d}, {best_rsum:.3f}")

    return best_rsum, best_epoch

def i2t5(all_i2t_idx):
    n = all_i2t_idx.shape[0]
    ranks = np.zeros(n)

    for index in range(n):
        inds = all_i2t_idx[index]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0]
            if len(tmp) == 0:
                tmp = 16
            else:
                tmp = tmp[0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    return (r1, r5, r10)

def t2i5(all_t2i_idx):
    n = int(all_t2i_idx.shape[0]/5)
    ranks = np.zeros(all_t2i_idx.shape[0])

    # --> (5N(caption), N(image))
    for index in range(n):
        for i in range(5):
            inds = all_t2i_idx[5*index + i]
            tmp = np.where(inds == index)[0]
            if len(tmp) == 0:
                tmp = 16
            else:
                tmp = tmp[0]
            ranks[5*index + i] = tmp

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    return (r1, r5, r10)