# train.py
import torch
from torch import nn
from d2l import torch as d2l
from download import load_data
import csv
import os
import time

# 加载数据
train_features, test_features, train_labels, test_data = load_data()
# 定义损失函数
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    """构建一个简单的线性回归网络"""
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net

def log_rmse(net, features, labels):
    """计算对数均方根误差(log RMSE)"""
    # 为防止取对数时出现负值，将预测值小于1的部分截断为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
    """训练模型，并返回每轮训练及（可选）验证误差列表"""
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    """返回第 i 折数据：训练数据和验证数据"""
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    """执行 k 折交叉验证，返回平均训练和验证误差"""
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print(f'折 {i + 1}，训练 log rmse: {float(train_ls[-1]):f}, 验证 log rmse: {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

def run_experiment_and_log(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size, results_file="../output/records.csv"):
    """
    执行一次交叉验证实验，
    并将超参数及结果（平均训练和验证误差）追加保存到 CSV 文件中
    """
    train_rmse, valid_rmse = k_fold(k, train_features, train_labels,
                                    num_epochs, lr, weight_decay, batch_size)
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "k": k,
        "num_epochs": num_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "train_rmse": train_rmse,
        "valid_rmse": valid_rmse
    }
    file_exists = os.path.isfile(results_file)
    with open(results_file, "a", newline="") as f:
        fieldnames = ["timestamp", "k", "num_epochs", "lr", "weight_decay", "batch_size", "train_rmse", "valid_rmse"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
    print("实验记录已保存：", record)
    return record

if __name__ == '__main__':
    # Hyperparameter which need to be adjusted
    k, num_epochs, lr, weight_decay, batch_size = 10, 400, 3, 0.01, 64
    record = run_experiment_and_log(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)