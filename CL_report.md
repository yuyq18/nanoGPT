# 计算语言学大作业-实验报告

> 庾源清 王哲凡

## 0. 实验环境搭建

在nanoGPT代码基础上，运行命令

```
python train.py config/train_shakespeare_char.py
```

最后10个step的截图如下：

<img src="figs/0-shakespeare_char_train.jpg" style="zoom:40%;" />

运行命令

```
python sample.py --out_dir=out-shakespeare-char
```

生成样本截图如下：

<img src="figs/0-shakespeare_char_sample.jpg" style="zoom:40%;" />

## 1. 预训练
### 1.1 字符级别模型预训练
#### 1.1.1 数据处理
仿照nanoGPT⾥shakespeare_char对于歌词语料进行预处理，对字符进行编码。其中，训练集、验证集和测试集的划分采用原始数据中的划分方式，在全部数据上建立词表，对于训练集、验证集和测试集进行编码。数据的基本信息如下：

**词表大小**：7,409
|          | token数 |
| -------- | ------- |
| 训练集   |    34,637,186     |
| 验证集   |    864,927     |
| 测试集   |    1,731,006     |
| 全部数据 |    37,233,119     |


#### 1.1.2 模型训练
仿照nanoGPT⾥shakespeare_char，设置训练参数config/train_lyrics_char.py

运行命令

```
python train.py config/train_lyrics_char.py
```

训练5000轮后，train/loss=2.63339，val/loss=2.66636

Train Loss曲线和Validation Loss曲线如下:

<img src="figs/1-train_loss.png" style="zoom:20%;" />

<img src="figs/1-valid_loss.png" style="zoom:20%;" />


编写代码测试测试集上的[PPL](https://huggingface.co/docs/transformers/perplexity)，设置stride=64,运行命令

```
python evaluate.py config/eval_lyrics_char.py
```

得到结果

```
PPL = 74.62682342529297
```

#### 1.1.3 歌词续写

设置tempreture=0.8，以“如果离别是为了能再见一面”为第一句续写歌词，结果如下:
```
如果离别是为了能再见一面
我也不会再思念
就算在每一天
我也都会想起你一直陪着你
我不会再怀念
就算在每一天
我也都会想起你一直陪着你
我不会再怀念
...
就算在每一天
我也都会想起你一直陪着你
我不会再怀念
就算在每一天
我也都会想起你一直陪
```
有如下观察：
1. 模型具有一定的续写能力。观察前几句的生成，逻辑上和句法上没太大问题，句尾还有“押韵”现象。
2. 模型在生成长文本时出现问题。观察后续歌词的生成，发现出现几句歌词一直循环的现象。这个现象在多次实验中均有出现，推测原因在于训练语料不够多，训练不够充分，且模型对于前面的生成的token依赖性太强。

### 1.2 分词后语⾔模型预训练

TODO