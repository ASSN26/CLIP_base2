# 项目简介

这是一个基于 CLIP 的基础代码，用于 RSITR（RSITMD、RSICD、UCM 数据集）或 CMITR（Flickr30k、MSCOCO 数据集）。

在之前的版本 CLIP_base（精简代码和清晰注释）的基础上，CLIP_base2 根据使用习惯有以下改进：

1. 整合了 RSITR（RSITMD、RSICD、UCM 数据集）和 CMITR（Flickr30k、MSCOCO 数据集）。
2. 提供 class CustomCLIP()，默认是基础的 CLIP 模型，以供更方便地开发。
3. 将 def test() 中的相似度矩阵计算由 CPU/numpy 改为 GPU/torch（这导致了轻微的指标差异），并将返回变量由相似度矩阵改为 topk 索引，以提高效率。
4. 训练日志文件夹名改为"日期_Rsum值"，非完整训练结果仅有"日期"，以便训练后查找相关结果。
5. 代码文件夹结构精简为：clip_ori（不常用文件夹）+ outputs（训练输出文件夹）+（常用文件）


# 使用方法

1. `config.py` 提供了一个基本配置文件，无需改动其中的内容。  
2. `train.py` 提供了一个训练文件，修改此处以选择相关数据集，例如：

```python
from data_RSITR import get_data
args.data_json = "/.json"
args.data_img = "images/"
```

3. `train.py` 提供了修改其它参数的示例，你可以通过修改这些参数进行模型调试、搜参，例如：

```python
args.lr = 1.0e-5
```

4. 使用 `python train.py` 或者 `nohup python train.py > exp_name.log 2>&1 &` 进行训练，训练后在 `outputs` 里会生成相应的训练日志。  
5. `model.py` 和 `loss.py` 提供了基础的模型和损失函数，以供开发。


# 训练结果

最后，在 RTX3090上（ly_clip_38，详细信息见 GPU_torch_info.txt 和 requirements.txt），提供了 CLIP_base2 的训练结果以供参考（详见 dataset.log）：

| Dataset  | time   | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 | Rsum    |
|----------|--------|---------|---------|----------|---------|---------|----------|---------|
| RSITMD   | 1384s  | 29.20   | 51.77   | 62.39    | 24.96   | 55.49   | 71.73    | 295.531 |
| RSICD    | 2744s  | 20.86   | 37.42   | 51.33    | 14.25   | 38.26   | 55.65    | 217.768 |
| UCM      | 559s   | 21.43   | 58.57   | 81.90    | 18.95   | 64.67   | 94.48    | 340.000 |
| Flickr30k| 8851s  | 88.80   | 98.00   | 99.40    | 76.18   | 93.54   | 96.76    | 552.680 |
| MSCOCO   | 32895s | 62.32   | 86.30   | 92.24    | 46.80   | 75.80   | 84.97    | 448.436 |

---
