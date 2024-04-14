import time
import train
import bert_model
import torch
import numpy as np
import dataloader
import config
import train
import pandas as pd
from tqdm import tqdm
import csv
import sys
import torch.nn.functional as F



if __name__ == '__main__':
    config_inf = config.Config()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    dataloader_application = dataloader.create_dataloader()
    print("Time usage:", time.time() - start_time)

    model = bert_model.Model(config_inf.bert_path, config_inf.hidden_size, config_inf.num_classes).to(torch.device('cuda'))
    model.load_state_dict(torch.load(config_inf.save_path, map_location='cuda'))
    model.eval()

    all_len = len(dataloader_application)
    probabilities_label_0 = []  # 存储标签为0的概率
    cot = 0
    for i,j in dataloader_application:
        outputs = model(i) 
        probabilities = F.softmax(outputs, dim=1)  # 应用 softmax 函数
        probability_label_0 = probabilities[:, 0]  # 提取标签为0的概率
        probabilities_label_0.extend(probability_label_0.cpu().numpy())
        cot+=1
        num = format(cot/all_len, '.2%')
        if cot%10 == 0:
            print(num)
    
    #将预测结果写入文件
    with open("./dataset/result.csv", 'w',encoding="utf-8") as w:
        writer = csv.writer(w)
        writer.writerow(["label_0_probability"])
        for i in probabilities_label_0:
            writer.writerow([i])