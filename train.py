# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from pytorch_pretrained.optimization import BertAdam
import config


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):  # 选择数据初始化的方式
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(model, train_data_loader, dev_data_loader, test_data_loader):
    config_inf = config.Config()
    start_time = time.time()
    model.train()  # 设置为训练模式
    param_optimizer = list(model.named_parameters())  # 存放参数名
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config_inf.learning_rate,
                         warmup=0.05,
                         t_total=len(train_data_loader) * config_inf.num_epochs)

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    for epoch in range(config_inf.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config_inf.num_epochs))
        for i, (trains, labels) in enumerate(train_data_loader):
            outputs = model(trains)
            model.zero_grad()
            # print(outputs, labels)
            loss = F.cross_entropy(outputs, labels)  # 交叉熵损失
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()[:,1]
                predic = torch.max(outputs.data, 1)[1].cpu()
                # print(true, predic)
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(model, dev_data_loader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config_inf.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = int(time.time() - start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config_inf.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    # test(model, test_data_loader)


def test(model, test_data_loader, path):
    config_inf = config.Config()
    # test
    # model.load_state_dict(torch.load(config_inf.save_path))
    model.load_state_dict(torch.load(path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(model, test_data_loader, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = start_time - time.time()
    print("Time usage:", time_dif)


def evaluate(model, data_loader, test=False):
    config_inf = config.Config()
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_loader:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()[:,1]
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            # print(labels, predic)
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config_inf.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_loader), report, confusion
    return acc, loss_total / len(data_loader)