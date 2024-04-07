from pytorch_pretrained import BertModel, BertTokenizer
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, bert_path, hidden_size, num_classes):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out