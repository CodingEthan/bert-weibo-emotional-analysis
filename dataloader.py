import torch
from tqdm import tqdm
import time
from datetime import timedelta

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import config

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

class Dataset(Dataset):
    def __init__(self, txt_path, transform=None):  # 加载所有文件到数组
        self.contents = []
        self.labels = []
        with open(txt_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')  # 切分为语句和标签
                self.contents.append(content)
                self.labels.append(label)
        print(txt_path)
        print("contentSize = ", len(self.contents))
        print("labelsSize = ", len(self.labels))


    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):  # 获取单个信息算法
        config_inf = config.Config()

        token = config_inf.tokenizer.tokenize(self.contents[idx])
        token = [CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = config_inf.tokenizer.convert_tokens_to_ids(token)
        padsize = config_inf.pad_size

        if config_inf.pad_size >= 1:
            if len(token) < padsize:
                mask = [1] * len(token_ids) + [0] * (padsize - len(token))
                token_ids += ([0] * (padsize - len(token)))
            else:
                mask = [1] * padsize
                token_ids = token_ids[:padsize]
                seq_len = padsize

        x = torch.LongTensor(token_ids).to(config_inf.device)
        y = torch.Tensor([1-int(self.labels[idx]), int(self.labels[idx])]).to(config_inf.device)
        seq_len = torch.LongTensor([seq_len]).to(config_inf.device)
        mask = torch.LongTensor(mask).to(config_inf.device)

        # print(seq_len)
        # print(len(x), len(seq_len), len(mask), len(y))
        return (x, seq_len, mask), y


def get_data_loader():  # return trainloader and testloader
    config_inf = config.Config()
    data_train = Dataset(config_inf.train_path)
    data_test = Dataset(config_inf.test_path)
    data_dev = Dataset(config_inf.dev_path)

    train_loader = DataLoader(data_train, batch_size=config_inf.batch_size, shuffle=True)  # , collate_fn=collate
    test_loader = DataLoader(data_test, batch_size=config_inf.batch_size, shuffle=True)  # , collate_fn=collate
    dev_loader = DataLoader(data_dev, batch_size=config_inf.batch_size, shuffle=True)
    return train_loader, dev_loader, test_loader

def create_dataloader():
    config_inf = config.Config()
    dataloader_application_dataset = Dataset(config_inf.application_path)
    dataloader_application = DataLoader(dataloader_application_dataset, batch_size=config_inf.batch_size, shuffle=True)
    return dataloader_application