# 算法模块
import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
import numpy as np
from tqdm import tqdm
from manage_feature import weather_data_df as feature_df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger
from analyze_data import data_df as target_df
from torch.utils.data import Dataset, DataLoader


finally_features = [
    'data_datetime', 'precipitation_diff', 'cloudrate_diff', 'dswrf_diff',
    'visibility_diff', 'precipitation_probability_diff', 'apparent_temperature_diff',
    'pressure_diff', 'wind_speed_diff', 'humidity_diff', 'minute_of_day', 'distance_max'
]
# finally_features = [
#     'data_datetime', 'precipitation_diff', 'cloudrate_diff', 'dswrf_diff',
#     'visibility_diff', 'apparent_temperature_diff',
#     'humidity_diff', 'minute_of_day', 'distance_max'
# ]
target = 'is_increase'
not_continuous_features = 'minute_of_day'
continuous_features = [
    'precipitation_diff', 'cloudrate_diff', 'dswrf_diff',
    'visibility_diff', 'precipitation_probability_diff', 'apparent_temperature_diff',
    'pressure_diff', 'wind_speed_diff', 'humidity_diff', 'distance_max'
]
# continuous_features = [
#     'precipitation_diff', 'cloudrate_diff', 'dswrf_diff',
#     'visibility_diff', 'apparent_temperature_diff',
#     'humidity_diff', 'distance_max'
# ]


class SolarPowerDataset(Dataset):
    def __init__(self, X, y):
        # 特征
        self.continuous = torch.tensor(X[continuous_features].values, dtype=torch.float32)

        # 类别特征
        self.min = torch.tensor(X[not_continuous_features].values, dtype=torch.long)

        # 附带一个日期时间对象用于观测性能而不用于训练
        # self.data_time = X['data_datetime'].tolist()

        # 目标值
        self.target = torch.tensor(y, dtype=torch.long).view(-1)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return (
            # (self.continuous[idx], self.min[idx], self.data_time[idx]),
            (self.continuous[idx], self.min[idx]),
            self.target[idx]
        )


class MLP(nn.Module):
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super(MLP, self).__init__()

        # 嵌入层
        self.minute_embedding = nn.Embedding(24, 3)  # 一天中的第几分钟嵌入层

        # 定义第1个全连接层
        self.fc1 = nn.Linear(13, 30)
        # 定义第2个全连接层
        self.fc2 = nn.Linear(30, 50)
        # 定义第3个全连接层
        self.fc3 = nn.Linear(50, 100)
        # 定义第4个全连接层
        self.fc4 = nn.Linear(100, 30)
        # 定义第5个全连接层
        self.fc5 = nn.Linear(30, 3)

        # 定义激活函数
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

    def forward(self, features, minute_feature):
        minute_feature = self.minute_embedding(minute_feature)

        # 合并特征
        x = torch.cat([features, minute_feature], dim=1)

        out = self.fc1(x)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc4(out)
        out = self.relu(out)

        out = self.fc5(out)

        return out


def split_dataset_by_day(df):
    text_X = df.loc[
        (df['data_datetime'] >= datetime.datetime(2025, 7, 25))
        &
        (df['data_datetime'] < datetime.datetime(2025, 7, 26)),
        finally_features
    ]
    train_X = df.loc[
        ~((df['data_datetime'] >= datetime.datetime(2025, 7, 25))
          &
          (df['data_datetime'] < datetime.datetime(2025, 7, 26))),
        finally_features
    ]

    text_y = df.loc[
        (df['data_datetime'] >= datetime.datetime(2025, 7, 25))
        &
        (df['data_datetime'] < datetime.datetime(2025, 7, 26)),
        target
    ]
    train_y = df.loc[
        ~((df['data_datetime'] >= datetime.datetime(2025, 7, 25))
          &
          (df['data_datetime'] < datetime.datetime(2025, 7, 26))),
        target
    ]

    logger.info(f'训练集条数：{len(train_y)}')
    logger.info(f'测试集条数：{len(text_y)}')

    return train_X, text_X, train_y, text_y


def manager_data():
    logger.info(f'目标值条数:{len(target_df)}')
    logger.info(f'特征值条数:{len(feature_df)}')


    combination_df = pd.merge(
        left=target_df, right=feature_df,
        how='inner', left_on='time', right_on='data_datetime'
    )

    # print(combination_df.loc[combination_df['is_increase'] == 1].shape[0])
    # print(combination_df.loc[combination_df['is_increase'] == 0].shape[0])
    # print(combination_df.loc[combination_df['is_increase'] == 2].shape[0])

    X = combination_df[finally_features]
    y = combination_df[target]

    # 分割特征和目标
    # 分割训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)
    X_train, X_test, y_train, y_test = split_dataset_by_day(combination_df)

    # 初始化标准化器
    scaler = StandardScaler()
    scaler.fit(X_train[continuous_features])  # 训练标准化器

    # 标准化特征
    X_train[continuous_features] = scaler.transform(X_train[continuous_features])
    X_test[continuous_features] = scaler.transform(X_test[continuous_features])

    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)

    # 创建数据集
    train_dataset = SolarPowerDataset(X_train, y_train.tolist())
    test_dataset = SolarPowerDataset(X_test, y_test.tolist())

    # 创建数据加载器
    batch_size = 240
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_model_save_model():
    train_loader, test_loader = manager_data()

    # 初始化MLP
    model = MLP()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model.to(device)

    # 定义损失函数
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    learning_rate = 0.001  # 学习率
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    # 训练网络
    num_epochs = 1200
    for epoch in tqdm(range(num_epochs)):
        i = 0
        # for (features, minute_feature, date_time), labels in train_loader:
        for (features, minute_feature), labels in train_loader:
            # print(f'第 `{i + 1}` 次训练')
            if torch.isnan(features).any() or torch.isnan(labels).any():
                # print(f"第 `{i + 1}` 次<训练时> 发现NaN值!")
                i = i + 1

                continue

            # 移动数据到设备
            features = features.to(device)
            minute_feature = minute_feature.to(device)
            labels = labels.to(device)

            # 将数据送到网络中
            # print('看看输入模型的参数：', features[0])

            outputs = model(features, minute_feature)

            # print('模型输出是个啥？', outputs)
            # print('模型输出是个啥？', type(outputs))
            # 计算损失
            loss = criterion(outputs, labels)

            # 首先将梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            i = i + 1

    def evaluate_model(model, data_loader, device, is_course=False):
        model.eval()

        with torch.no_grad():
            correct = 0

            # for (features, minute_feature, date_time_evaluate), labels in data_loader:
            for (features, minute_feature), labels in data_loader:
                sumerate_features = features.numpy().tolist()
                sumerate_minute_feature = minute_feature.numpy().tolist()
                sumerate_labels = labels.numpy().tolist()
                # sumerate_date_time = date_time_evaluate.numpy().tolist()

                features = features.to(device)
                minute_feature = minute_feature.to(device)

                outputs = model(features, minute_feature)

                pred = outputs.argmax(dim=1, keepdim=True).cpu()  # 获取概率最大的类别

                if is_course:
                    sumerate_pred = pred.numpy().tolist()
                    for index, item in enumerate(sumerate_labels):
                        if sumerate_pred[index][0] == sumerate_labels[index]:
                            # logger.success(f'日期：{sumerate_date_time[index]} - 时间为：{sumerate_minute_feature[index]} - 预测为：{sumerate_pred[index]} - 实际值为：{sumerate_labels[index]} - 特征：{sumerate_features[index]}')
                            logger.success(f'时间为：{sumerate_minute_feature[index]} - 预测为：{sumerate_pred[index]} - 实际值为：{sumerate_labels[index]} - 特征：{sumerate_features[index]}')
                        else:
                            # logger.error(f'日期：{sumerate_date_time[index]} - 时间为：{sumerate_minute_feature[index]} - 预测为：{sumerate_pred[index]} - 实际值为：{sumerate_labels[index]} - 特征：{sumerate_features[index]}')
                            logger.error(f'时间为：{sumerate_minute_feature[index]} - 预测为：{sumerate_pred[index]} - 实际值为：{sumerate_labels[index]} - 特征：{sumerate_features[index]}')

                # labels.cpu().numpy()
                correct += pred.eq(labels.view_as(pred)).sum().item()

        # logger.info(f'测试集：\n{list(data_loader.dataset)}')

        return correct / len(data_loader.dataset), len(data_loader.dataset)


    # 获取预测结果
    train_accuracy, train_total_count = evaluate_model(model, train_loader, device)
    test_accuracy, text_total_count = evaluate_model(model, test_loader, device, True)

    print(f'训练集准确率：{train_accuracy}，总记录条数：{train_total_count}')
    print(f'测试集准确率：{test_accuracy}，总记录条数：{text_total_count}')


if __name__ == '__main__':
    train_model_save_model()
    # manager_data()

