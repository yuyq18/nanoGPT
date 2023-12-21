# 计算语言学大作业-实验报告

> 庾源清 王哲凡

## 0. 实验环境搭建

在nanoGPT代码基础上，运行命令
```
python train.py config/train_shakespeare_char.py
```
最后10个step的截图如下：
<img src="figs/0-shakespeare_char_train.jpg" style="zoom:50%;" />


运行命令
```
python sample.py --out_dir=out-shakespeare-char
```
生成样本截图如下：
<img src="figs/0-shakespeare_char_sample.jpg" style="zoom:50%;" />

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
得到如下Train Loss曲线以及Validation Loss曲线:

TODO

#### 1.1.3 歌词续写
TODO