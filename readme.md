# 基于Bert的微博评论情感分析（参考自https://gitee.com/chyl-100days/bert-weibo-emotional-analysis/blob/master）

# 摘要

## 本项目

本文采用了基于Bert架构进行微博评论情感分析，数据集来源于Github，采用其中100,000条数据，80000条作为训练集，10000条作为验证集和测试集。Bert的预训练模型开源模型来源于HuggingFace team的bert base chinese。该模型已针对中文进行了预训练。本项目基于pytorch进行编程。



## Bert

语言表示模型BERT，名字全称为 Bidirectional Encoder Representations from Transformer （来自于Transformer的双向编码器表示）

BERT是用于训练深的双向表示，用的是没有标签的数据，训练时会联合左右的上下文信息。

于是，我们预训练好的BERT模型只用加一个额外的输出层就可以得到一个不错的结果，再很多NLP任务上不需要对任务对架构进行大的改动（这里说明了和GPT不一样）ELMo（前人的研究）用的是一个基于RNN的架构，BERT用的是Transformer，BERT任务做起来比较简单。



# 项目

## 目录结构

| 文件夹             | 作用                   |
| ------------------ | ---------------------- |
| Bert-pretrain      | 存放bert预训练模型     |
| dataset            | 训练集、验证集、测试集 |
| pytorch_pretrained | 预训练相关脚本         |
| saved              | 存放训练出来的权重     |

| Python代码     | 作用             |
| -------------- | ---------------- |
| bert_module.py | Bert模型类       |
| config.py      | 配置项           |
| dataloader.py  | 数据加载实现代码 |
| **main.py**    | 程序运行主程     |
| train.py       | 训练部分代码     |



## 运行训练

由于bert模型过于大，Bert-pretrain和saved下的文件放在了百度网盘中

链接：https://pan.baidu.com/s/16aGJh4fBTFodZcI_nzN4ug 
提取码：syrj 



随后直接运行main.py即可进行训练





## 训练集

训练集来源于Github上某位作者搜集的NLP语料

[SophonPlus/ChineseNlpCorpus: 搜集、整理、发布 中文 自然语言处理 语料/数据集，与 有志之士 共同 促进 中文 自然语言处理 的 发展。 (github.com)](https://github.com/SophonPlus/ChineseNlpCorpus)

采用了其中 <weibo_senti_100k>10 万多条，带情感标注 新浪微博，正负向评论约各 5 万条

sample其中几条看看：

| id     | Label | Review                                                       |
| ------ | ----- | ------------------------------------------------------------ |
| 62050  | 0     | 太过分了@Rexzhenghao //@Janie_Zhang:招行最近负面新闻越来越多呀... |
| 68263  | 0     | 希望你?得好?我本＂?肥血?史＂[晕][哈哈]@Pete三姑父            |
| 81472  | 0     | 有点想参加????[偷?]想安排下时间再决定[抓狂]//@黑晶晶crystal: @细腿大羽... |
| 42021  | 1     | [给力]感谢所有支持雯婕的芝麻！[爱你]                         |
| 7777   | 1     | 2013最后一天，在新加坡开心度过，向所有的朋友们问声：新年快乐！2014年，我们会更好[调... |
| 100399 | 0     | 大中午出门办事找错路，曝晒中。要多杯具有多杯具。[泪][泪][汗] |
| 82398  | 0     | 马航还会否认吗？到底在隐瞒啥呢？[抓狂]//@头条新闻: 转发微博  |
| 106423 | 0     | 克罗地亚球迷很爱放烟火！球又没进，就硝烟四起。[晕]           |
| 24798  | 1     | [抱抱]福芦 TangRoulou 吉祥书 8.8折优惠 >>> http://t.cn/z...  |

采用其中100,000条数据，80000条作为训练集，10000条作为验证集和测试集



## 预训练

Bert的预训练模型开源模型来源于HuggingFace team的bert base chinese

来源地址为：[bert-base-chinese · Hugging Face](https://huggingface.co/bert-base-chinese#)

**Model Description:** This model has been pre-trained for Chinese, training and random input masking has been applied independently to word pieces (as in the original BERT paper).

**Developed by:** HuggingFace team

**Model Type:** Fill-Mask

**Language(s):** Chinese

**License:** [More Information needed]

**Parent Model:** See the [BERT base uncased model](https://huggingface.co/bert-base-uncased) for more information about the BERT base model.



## 训练参数

训练环境：AMD Ryzen7 5800H + GTX3070 8G

epoch：3

batchSize：128

padSize：32 （一句话的最大长度，少补多切）

learningRate：5e-5

隐藏层大小：768（原始论文的base版参数）



Bert原始论文的base版参数 12个Transformer块、768个隐藏层块、12个自注意力多头数



# 训练结果

**训练**

| 迭代次数（batch） | 训练损失 | 训练正确率 | 验证集损失 | 验证集正确率 | 使用时间(秒) |
| ----------------- | -------- | ---------- | ---------- | ------------ | ------------ |
| 0 epoch1          | 0.68     | 61.72%     | 0.69       | 53.83%       | 123          |
| 100               | 0.28     | 84.38%     | 0.3        | 86.02%       | 443          |
| 200               | 0.29     | 86.72%     | 0.31       | 85.93%       | 770          |
| 300               | 0.31     | 84.38%     | 0.28       | 87.04%       | 1109         |
| 400               | 0.22     | 88.28%     | 0.3        | 85.91%       | 1430         |
| 500               | 0.28     | 84.38%     | 0.28       | 87.50%       | 1766         |
| 600               | 0.36     | 81.25%     | 0.27       | 87.63%       | 2096         |
| 700 epoch2        | 0.27     | 87.50%     | 0.27       | 87.61%       | 2422         |
| 800               | 0.29     | 85.94%     | 0.28       | 87.28%       | 2740         |
| 900               | 0.28     | 85.94%     | 0.27       | 87.56%,      | 3058         |
| 1000              | 0.27     | 87.50%     | 0.27       | 87.84%       | 3376         |
| 1100              | 0.24     | 88.28%     | 0.27       | 87.31%       | 3700         |
| 1200              | 0.24     | 89.06%     | 0.3        | 86.66%       | 4025         |
| 1300 epoch3       | 0.18     | 94.53%     | 0.3        | 87.35%       | 4351         |
| 1400              | 0.22     | 90.62%     | 0.3        | 87.38%       | 4689         |
| 1500              | 0.21     | 92.97%     | 0.3        | 87.07%       | 5036         |
| 1600              | 0.2      | 92.19%     | 0.31       | 87.26%       | 5360         |
| 1700              | 0.26     | 86.72%     | 0.31       | 87.31%       | 5684         |
| 1800              | 0.15     | 92.97%     | 0.31       | 87.28%       | 6013         |



**测试**

|          | 正确率 | 召回率 | F1-score | 数量  |
| -------- | ------ | ------ | -------- | ----- |
| 负面标签 | 0.8631 | 0.8730 | 0.8680   | 5063  |
| 正面标签 | 0.8632 | 0.8580 | 0.8631   | 4937  |
| 准确度   |        |        | 0.8656   | 10000 |
| 宏平均   | 0.8657 | 0.8655 | 0.8656   | 10000 |
| 加权平均 | 0.8656 | 0.8656 | 0.8656   | 10000 |



# 参考

[1] 李宏毅 机器学习 春 Transformer 上 https://www.bilibili.com/video/BV1Wv411h7kN

[2] 李宏毅 机器学习 春 自监督式学习 Bert https://www.bilibili.com/video/BV1Wv411h7kN

[3] 李沐 BERT论文逐段精读 https://www.bilibili.com/video/BV1PL411M7eQ

[4] Github [649453932 / Bert-Chinese-Text-Classification-Pytorch](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch)

[5] Github [SophonPlus / ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)

[6] bert-base-chinese https://huggingface.co/bert-base-chinese#