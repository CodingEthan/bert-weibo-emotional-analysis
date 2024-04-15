import time
import torch
from pytorch_pretrained import BertModel, BertTokenizer

class Config(object):
    """配置参数"""
    def __init__(self):
        self.train_path = './dataset/train_data.txt'                                # 训练集
        self.dev_path = './dataset/validation_data.txt'                                    # 验证集
        self.test_path = './dataset/test_data.txt'                                  # 测试集
        self.class_list = ['Negative', 'Positive']                                # 类别名单
        self.save_path = './saved/' + str(int(time.time())) + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 192                                          # mini-batch大小128
        self.pad_size = 64                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 4e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 1024
        self.application_path = './dataset/application_data.txt'