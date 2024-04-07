import time
import train
import bert_model
import torch
import numpy as np
import dataloader
import config
import train

if __name__ == '__main__':
    config_inf = config.Config()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data_loader, dev_data_loader, test_data_loader = dataloader.get_data_loader()
    print("Time usage:", time.time() - start_time)

    # train
    model = bert_model.Model(config_inf.bert_path, config_inf.hidden_size, config_inf.num_classes).to(config_inf.device)
    model.load_state_dict(torch.load("./saved/1669439524.ckpt"))
    train.train(model, train_data_loader, dev_data_loader, test_data_loader)
    # train.test(model, test_data_loader, "./saved/1669439524.ckpt")