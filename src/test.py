# test.py
import torch
import pandas as pd
import csv
import os
from train import get_net, train
from download import load_data

# 加载数据
train_features, test_features, train_labels, test_data = load_data()

def load_best_hparams(results_file="../output/records.csv"):
    """
    从 results_file 中读取所有训练记录，
    找到 valid_rmse 最小的那条记录，
    返回最佳超参数（如果没有记录则返回 None）
    """
    best_valid_rmse = float("inf")
    best_params = None
    try:
        with open(results_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                valid_rmse = float(row["valid_rmse"])
                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    best_params = row
    except Exception as e:
        print("读取结果文件失败：", e)
    return best_params

def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    """
    使用给定超参数在全部训练数据上训练最终模型，
    对测试集生成预测，并将结果保存为 submission.csv
    """
    net = get_net()
    # 此处不使用验证集，直接在全部训练数据上训练
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    print(f'训练 log rmse：{float(train_ls[-1]):f}')

    preds = net(test_features).detach().numpy()
    # 将预测结果格式化为一维数组，并赋给 test_data 的 'SalePrice' 列
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv("../output/result.csv", index=False)

if __name__ == '__main__':
    best_params = load_best_hparams("../output/records.csv")
    if best_params is None:
        # 如果没有找到记录，则使用默认超参数
        num_epochs, lr, weight_decay, batch_size = 100, 5, 0, 64
        print("未找到最佳超参数记录，使用默认超参数。")
    else:
        num_epochs = int(best_params["num_epochs"])
        lr = float(best_params["lr"])
        weight_decay = float(best_params["weight_decay"])
        batch_size = int(best_params["batch_size"])
        print("使用最佳超参数：", best_params)

    train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size)
    print("预测完成，生成 result.csv 文件。")