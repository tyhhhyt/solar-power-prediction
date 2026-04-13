# RNN算法模块 - 测试集准确率最高达到了0.775891
import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger
from manage_feature import weather_data_df as feature_df
from analyze_data import data_df as target_df
from torch.utils.data import Dataset, DataLoader


# finally_features = [
#     'data_datetime', 'precipitation', 'cloudrate', 'dswrf',
#     'visibility', 'precipitation_probability', 'apparent_temperature',
#     'pressure', 'wind_speed', 'humidity', 'minute_of_day', 'distance_max'
# ]
# finally_features = [
#     'data_datetime', 'precipitation_diff', 'cloudrate_diff', 'dswrf_diff',
#     'visibility_diff', 'apparent_temperature_diff',
#     'humidity_diff', 'minute_of_day', 'distance_max'
# ]
target = 'is_increase'
not_continuous_features = 'minute_of_day'
# continuous_features_minute = [
#     'precipitation', 'cloudrate', 'dswrf',
#     'visibility', 'precipitation_probability', 'apparent_temperature',
#     'pressure', 'wind_speed', 'humidity', 'minute_of_day', 'distance_max'
# ]
# continuous_features = [
#     'precipitation', 'cloudrate', 'dswrf',
#     'visibility', 'precipitation_probability', 'apparent_temperature',
#     'pressure', 'wind_speed', 'humidity', 'distance_max'
# ]
finally_features = [
    'data_datetime', 'precipitation', 'cloudrate', 'dswrf',
    'visibility', 'apparent_temperature',
    'wind_speed', 'humidity', 'minute_of_day', 'distance_max'
]
continuous_features_minute = [
    'precipitation', 'cloudrate', 'dswrf',
    'visibility', 'apparent_temperature',
    'wind_speed', 'humidity', 'minute_of_day', 'distance_max'
]
continuous_features = [
    'precipitation', 'cloudrate', 'dswrf',
    'visibility', 'apparent_temperature',
    'wind_speed', 'humidity', 'distance_max'
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


class RNNMLP(nn.Module):
    # 初始化方法
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        # 调用父类的初始化方法
        super(RNNMLP, self).__init__()

        # 嵌入层
        self.minute_embedding = nn.Embedding(24, 4)  # 一天中的第几分钟嵌入层

        self.rnn = nn.RNN(
        # self.rnn = nn.LSTM(
        # self.rnn = nn.GRU(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False     # 单向RNN
        )
        self.fc1 = nn.Linear(hidden_size, output_size)
        # self.fc1 = nn.Linear(hidden_size, 64)
        # self.fc2 = nn.Linear(64, output_size)
        # self.fc3 = nn.Linear(128, output_size)

        # 定义激活函数
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.2)

    def forward(self, features, minute_feature):
        minute_feature = self.minute_embedding(minute_feature)

        # 合并特征
        x = torch.cat([features, minute_feature], dim=-1)

        out, _ = self.rnn(x)

        logger.info(f'kk看看这个out形状：{out.shape}')

        out = self.fc1(out[:, -1, :])

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


# 一个时间窗口包含前24条数据
def manager_data():
    # logger.info(f'目标值条数:{target_df.columns}')
    # logger.info(f'特征值条数:{len(feature_df)}')

    merge_data = pd.merge(target_df[[target, 'time']], feature_df[finally_features], left_on='time', right_on='data_datetime', how='inner')

    logger.info(f'merge列：{merge_data.columns}')
    logger.info(f'merge形状：{merge_data.shape}')

    scaler = StandardScaler()
    merge_data[continuous_features] = scaler.fit_transform(merge_data[continuous_features])

    X, X_minute, y = [], [], []
    feature_date_list = merge_data['data_datetime'].dt.strftime('%Y-%m-%d').unique().tolist()
    for date_item in feature_date_list:
        start_loop_datetime = datetime.datetime.combine(
            datetime.datetime.strptime(date_item, '%Y-%m-%d'),
            datetime.datetime.min.time()
        )
        end_loop_datetime = datetime.datetime.combine(
            datetime.datetime.strptime(date_item, '%Y-%m-%d'),
            datetime.datetime.max.time()
        )

        logger.info(rf'当前日期为：`{start_loop_datetime}`')

        x_temp = merge_data.loc[
            (merge_data['data_datetime'] >= start_loop_datetime)
            &
            (merge_data['data_datetime'] < end_loop_datetime),
            continuous_features
        ].values

        if len(x_temp) != 24:
            logger.warning(f'`{start_loop_datetime}` 日的数据为 `{len(x_temp)}` 条，不为24条不加入数据集中')
            continue

        X.append(
            x_temp
        )

        X_minute.append(
            list(map(lambda xx: int(xx), merge_data.loc[
                (feature_df['data_datetime'] >= start_loop_datetime)
                &
                (feature_df['data_datetime'] < end_loop_datetime),
                'minute_of_day'
            ].values.tolist()))
        )

        y.extend(
            merge_data.loc[
                (feature_df['data_datetime'] >= start_loop_datetime)
                &
                (feature_df['data_datetime'] < end_loop_datetime),
                target
            ].values.tolist()
        )
    logger.info(rf'特征批次长度为：{len(X)}')
    logger.info(rf'目标值批次长度为：{len(y)}')
    logger.info(rf'目标值批次长度为：{X_minute[:10]}')
    logger.info(rf'目标值批次长度为：{y[:10]}')
    # for i in X:
    #     logger.success(f'检测数据条数：{len(i)}')
    #     logger.warning(f'检测数据条数：{len(i[0])}')

    return X, X_minute, y


# 一个时间窗口包含前五条数据
def manager_data_5():
    merge_data = pd.merge(target_df[[target, 'time']], feature_df[finally_features], left_on='time',
                          right_on='data_datetime', how='inner')

    logger.info(f'merge列：{merge_data.columns}')
    logger.info(f'merge形状：{merge_data.shape}')

    scaler = StandardScaler()
    merge_data[continuous_features] = scaler.fit_transform(merge_data[continuous_features])

    seq_length = 2  # 时间窗口大小（用多少条历史数据预测未来的一条数据）

    X, X_minute, y = [], [], []
    for data_index in range(merge_data.shape[0] - seq_length):
        x_temp = merge_data.iloc[
            data_index:data_index + seq_length
        ][continuous_features].values

        X.append(
            x_temp
        )

        X_minute.append(
            list(map(lambda xx: int(xx), merge_data.iloc[
                data_index:data_index + seq_length
            ]['minute_of_day'].values.tolist()))
        )

        y.append(
            merge_data.iloc[
                data_index + seq_length
            ][target]
        )
    logger.info(rf'特征批次长度为：{len(X)}')
    logger.info(rf'目标值批次长度为：{len(y)}')
    logger.info(rf'X前十个预览：{X[:10]}')
    # logger.info(rf'X_minute前十个预览：{X_minute[:10]}')
    logger.info(rf'X_minute前十个预览：{X_minute}')
    logger.info(rf'y前十个预览：{y[:10]}')
    # for i in X:
    #     logger.success(f'检测数据条数：{len(i)}')
    #     logger.warning(f'检测数据条数：{len(i[0])}')

    # raise Exception('终止掉程序')

    return X, X_minute, y


def train_model_save_model():
    # X, X_minute, y = manager_data()
    X, X_minute, y = manager_data_5()

    X = torch.FloatTensor(X)
    X_minute = torch.tensor(X_minute, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)

    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    X_minute_train, X_minute_test = X_minute[:train_size], X_minute[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 初始化MLP
    model = RNNMLP(input_size=12, hidden_size=64, output_size=3, dropout=0.5, num_layers=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model.to(device)

    # 定义损失函数
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    learning_rate = 0.001  # 学习率
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    X_train = X_train.to(device)
    X_minute_train = X_minute_train.to(device)
    y_train = y_train.to(device)

    epochs = 150
    tqdm_obj = tqdm(range(epochs))
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train, X_minute_train)
        loss = criterion(outputs.squeeze(), y_train)  # 去掉多余的维度
        loss.backward()
        optimizer.step()
        tqdm_obj.update(1)
        tqdm_obj.set_description(f'Epoch {epoch}, Loss: {loss.item():.4f}')
        # logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():

        predictions = model(X_train, X_minute_train).argmax(dim=1)

        train_count = 0
        train_correct = 0
        for p, c in zip(predictions, y_train):
            train_count = train_count + 1
            if p == c:
                train_correct = train_correct + 1
        logger.warning(f'训练集准确率为：{train_correct / train_count}')

        # logger.info(f'看看这个所谓的测试集和对应的minute：\n{X_test.numpy().shape}\n{X_minute_test.numpy().shape}\n')
        minute_list = X_minute_test.numpy().tolist()

        X_test = X_test.to(device)
        X_minute_test = X_minute_test.to(device)
        y_test = y_test.to(device)
        predictions = model(X_test, X_minute_test).argmax(dim=1)
        test_correct = 0
        test_count = 0
        for p, c, m in zip(predictions, y_test, minute_list):
            test_count = test_count + 1
            if p == c:
                test_correct = test_correct + 1
                logger.success(f'{m}时间：预测与实际值：{p} - {c}')
            else:
                logger.error(f'{m}时间：预测与实际值：{p} - {c}')

        logger.warning(f'测试集准确率：{test_correct / test_count}')


if __name__ == '__main__':
    train_model_save_model()
    # manager_data()
    # logger.info(f'目标值条数:{len(target_df)}')
    # logger.info(f'特征值条数:{len(feature_df)}')

    # print(
    #     feature_df[
    #         (feature_df['data_datetime'] >= '2025-07-20 00:00:00')
    #         &
    #         (feature_df['data_datetime'] < '2025-07-21 00:00:00')
    #     ]
    # )
