import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # 配置参数
    parser.add_argument("--train_log",
                        type=str,
                        default="train_log/",
                        help="file of save train config-result",)
    parser.add_argument("--exp_name",
                        type=str,
                        default=None,
                        help="name like train_log/exp.0, otherwise use [current time_result Rsum]",)
    parser.add_argument("--precision",
                        choices=["amp", "fp16", "fp32"],
                        default="amp",
                        help="floating point precition")
    parser.add_argument("--print_freq",
                        type=int,
                        default=5,
                        help="print frequency in per batch")
    parser.add_argument("--save_ckpt",
                        type=bool,
                        default=False,
                        help="path of save ckpt")

    # 数据集参数
    parser.add_argument("--data_json",
                        type=str,
                        default="/data2/ly/data/igarss/RSITMD/dataset_RSITMD.json",
                        help=".json like RSITMD dataset json or Flickr30k train/test json")

    parser.add_argument("--data_img",
                        type=str,
                        default="/data2/ly/data/igarss/RSITMD/images/",
                        help="file of data image")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="train and test batch size")
    parser.add_argument("--num_workers",
                        type=int,
                        default=4,
                        help="set to 0 if debug")

    # 模型参数
    parser.add_argument("--pretrain",
                        default="/data2/ly/clip_pretrain/ViT-B-16.pt",
                        type=str,
                        help="ViT-B-16.pt/ViT-B-32.pt/...", )

    parser.add_argument("--context_length",
                        default=17,
                        type=int,
                        help="context max length")

    # 优化器和学习率策略参数
    parser.add_argument("--beta1",
                        type=float,
                        default=0.9,
                        help="Adam beta1 for ViTB 16/32")
    parser.add_argument("--beta2",
                        type=float,
                        default=0.98,
                        help="Adam beta2 for ViTB 16/32")
    parser.add_argument("--eps",
                        type=float,
                        default=1.0e-6,
                        help="Adam epsilon for ViTB 16/32")
    parser.add_argument("--wd",
                        type=float,
                        default=0.001,
                        help="weight decay.")
    parser.add_argument("--epoch_num",
                        type=int,
                        default=10,
                        help="train epoch num")
    parser.add_argument("--lr",
                        type=float,
                        default=1.0e-5,
                        help="learning rate")
    parser.add_argument("--warmup",
                        type=int,
                        default=100,
                        help="number of steps to warmup for")

    args = parser.parse_args()
    return args
