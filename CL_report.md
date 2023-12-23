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

训练5000轮后，train/loss=2.6334，val/loss=2.6664

Train Loss曲线和Validation Loss曲线如下:

<img src="figs/1-train_loss.png" style="zoom:20%;" />

<img src="figs/1-valid_loss.png" style="zoom:20%;" />


编写代码测试测试集上的[PPL](https://huggingface.co/docs/transformers/perplexity)，设置stride=64,运行命令

```
python evaluate.py config/eval_lyrics_char.py
```

得到结果

```
PPL = 8.036737442016602
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

#### 1.2.1 数据处理
使用词表大小为30000的BPE算法进行分词。其中，训练集、验证集和测试集的划分采用原始数据中的划分方式，在全部数据上建立词表，对于训练集、验证集和测试集进行编码。数据的基本信息如下：

**词表大小**：30000
|          | token数 |
| -------- | ------- |
| 训练集   |    18,488,903     |
| 验证集   |    439,310     |
| 测试集   |    877,700     |
| 全部数据 |    19,805,913     |


#### 1.2.2 模型训练
设置训练参数config/train_lyrics.py

运行命令

```
python train.py config/train_lyrics.py
```

训练5000轮后，train/loss=4.3054，val/loss=4.6454

Train Loss曲线和Validation Loss曲线如下:

<img src="figs/2-train_loss.png" style="zoom:20%;" />

<img src="figs/2-valid_loss.png" style="zoom:20%;" />


编写代码测试测试集上的[PPL](https://huggingface.co/docs/transformers/perplexity)，设置stride=64,运行命令

```
python evaluate.py config/eval_lyrics.py
```

得到结果

```
PPL = 57.463218688964844
```


#### 1.2.3 歌词续写

设置tempreture=0.8，以“如果离别是为了能再见一面”为第一句续写歌词，结果如下:
```
如果离别是为了能再见一面
今生的爱情故事为何错过
回忆从心会碎
曾经那么的甜蜜
在回忆中永远没开始
那是一种美丽的承诺
还是无法改变的幸福
...
在回忆中永远没开始
那些年
一个背影
在记忆中
再起 再起
在记忆中
让每个清晨
都写在彼此的记忆
都曾偶尔相遇
从时间开始
那是一种美丽的承诺
...
```

有如下观察：
1. 模型具有一定的续写能力，逻辑上和句法上没太大问题。
2. 模型在生成长文本时表现比char-level模型更好。虽然也出现了一定程度上的歌词重复，但中间存在一些歌词跳出原有的重复循环，重复率比char-level模型低很多。


#### 1.2.4 char-level模型比较

与char-level模型相比，训练开始时的loss时⾼了。在第一个step中，（TODO：通过计算给出模型训练第⼀个step loss的⼤概取值随词表⼤⼩的估计）

### 1.3 调参

TODO
