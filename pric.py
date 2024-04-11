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

    pre_list = []
    for i,j in dataloader_application:
        outputs = model(i) 
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()
        pre_list.extend(predic)
    
    #将预测结果写入文件
    with open("./dataset/result.csv", 'w',encoding="utf-8") as w:
        writer = csv.writer(w)
        writer.writerow(["label"])
        for i in pre_list:
            writer.writerow([i])