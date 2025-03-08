# download.py
import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch

# 定义 DATA_HUB 和基础 URL
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

# 设置 Kaggle 房价预测数据集的下载链接及校验码
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce'
)
DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90'
)

def download(name, cache_dir=os.path.join('..', 'data')):
    """下载 DATA_HUB 中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)  # 每次读取 1MB
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从 {url} 下载 {fname} ...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    """下载并解压 zip/tar 文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有 zip/tar 文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
    """下载 DATA_HUB 中的所有文件"""
    for name in DATA_HUB:
        download(name)

def load_data():
    """下载并加载数据，返回处理后的 train_features, test_features, train_labels 和原始 test_data"""
    train_data = pd.read_csv(download('kaggle_house_train'))
    test_data = pd.read_csv(download('kaggle_house_test'))

    # print(train_data.shape)
    # print(test_data.shape)
    # print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

    # 合并训练集和测试集（去掉ID列及目标变量）
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

    # 对数值型特征进行标准化（零均值、单位方差），并填充缺失值
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    # 对类别型特征进行独热编码，并将所有数据转换为 float32 类型
    all_features = pd.get_dummies(all_features, dummy_na=True)
    all_features = all_features.astype(np.float32)

    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
    train_labels = torch.tensor(
        train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

    return train_features, test_features, train_labels, test_data

if __name__ == '__main__':
    # 用于测试数据加载
    load_data()
    print("数据已加载完成。")