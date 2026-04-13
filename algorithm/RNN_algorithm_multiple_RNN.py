# 双头RNN算法模块 - 测试集准确率最高达到了0.861091954（测试集中不包含结果为 0 的数据条）
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from manage_feature import weather_data_df as feature_df
from analyze_data import data_df as target_df

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
        self.minute_embedding_1 = nn.Embedding(24, 4)  # 一天中的第几分钟嵌入层
        self.minute_embedding_2 = nn.Embedding(24, 4)  # 一天中的第几分钟嵌入层

        # self.rnn_1 = nn.RNN(
        # self.rnn = nn.LSTM(
        self.rnn_1 = nn.GRU(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False  # 单向RNN
        )

        # self.rnn_2 = nn.RNN(
        # self.rnn_2 = nn.LSTM(
        self.rnn_2 = nn.GRU(
            input_size + 2, hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False  # 单向RNN
        )

        self.fc1_1 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc1_2 = nn.Linear(hidden_size, hidden_size * 2)

        self.fc2_1 = nn.Linear(hidden_size * 2, output_size)
        self.fc2_2 = nn.Linear(hidden_size * 2, output_size)

        # self.fc3_1 = nn.Linear(hidden_size, output_size)
        # self.fc3_2 = nn.Linear(hidden_size, output_size)

        # self.fc1 = nn.Linear(2 * hidden_size, 128)
        self.fc1 = nn.Linear(output_size * 2, 256)
        # self.fc2 = nn.Linear(256, 64)
        # self.fc3 = nn.Linear(64, output_size)
        self.fc3 = nn.Linear(256, output_size)

        # 定义激活函数
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.2)

    def forward(self, features_1, minute_feature_1, features_2, minute_feature_2):
        # minute_feature_1 = self.minute_embedding_1(minute_feature_1)
        minute_feature_1 = self.encode_time(minute_feature_1, 24)
        # 合并特征
        x_1 = torch.cat([features_1, minute_feature_1], dim=-1)

        minute_feature_2 = self.minute_embedding_2(minute_feature_2)
        # minute_feature_2 = self.encode_time(minute_feature_2, 24)
        # 合并特征
        x_2 = torch.cat([features_2, minute_feature_2], dim=-1)

        out_1, _ = self.rnn_1(x_1)
        out_2, _ = self.rnn_2(x_2)
        out_1 = out_1[:, -1, :]  # 取最后时刻
        out_2 = out_2[:, -1, :]

        out_1_1 = self.fc1_1(out_1)
        out_1_1 = self.relu(out_1_1)
        out_1_1 = self.dropout(out_1_1)

        out_1_2 = self.fc1_2(out_2)
        out_1_2 = self.relu(out_1_2)
        out_1_2 = self.dropout(out_1_2)

        out_2_1 = self.fc2_1(out_1_1)
        out_2_1 = self.relu(out_2_1)
        out_2_1 = self.dropout(out_2_1)

        out_2_2 = self.fc2_2(out_1_2)
        out_2_2 = self.relu(out_2_2)
        out_2_2 = self.dropout(out_2_2)

        # out_3_1 = self.fc3_1(out_2_1)
        # out_3_1 = self.relu(out_3_1)
        # out_3_1 = self.dropout(out_3_1)
        #
        # out_3_2 = self.fc3_2(out_2_2)
        # out_3_2 = self.relu(out_3_2)
        # out_3_2 = self.dropout(out_3_2)

        # out = torch.cat([out_1, out_2], dim=-1)
        # out = torch.cat([out_3_1, out_3_2], dim=-1)
        out = torch.cat([out_2_1, out_2_2], dim=-1)

        # logger.info(f'kk看看这个out形状：{out.shape}')
        # logger.info(f'kk看看这个out形状：{out}')

        # out = self.fc1(out[:, -1, :])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)

        # out = self.fc2(out)
        # out = self.relu(out)
        # out = self.dropout(out)

        out = self.fc3(out)

        return out

    def encode_time(self, t_value, T):
        """
        t: [B, T] 时间索引 (int or float)
        T: 周期长度，例如 24小时=24，分钟=1440
        """
        t = t_value.float()
        sin_t = torch.sin(2 * torch.pi * t / T)
        cos_t = torch.cos(2 * torch.pi * t / T)
        return torch.stack([sin_t, cos_t], dim=-1)  # [B, T, 2]


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

    merge_data = pd.merge(target_df[[target, 'time']], feature_df[finally_features], left_on='time',
                          right_on='data_datetime', how='inner')

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


def second_rnn_data():
    target_df['hour_minute'] = target_df['time'].dt.time
    target_df['date'] = target_df['time'].dt.date
    target_df_days_rnn = target_df.set_index(['hour_minute', 'date'], drop=True).sort_index()

    feature_df['hour_minute'] = feature_df['data_datetime'].dt.time
    feature_df['date'] = feature_df['data_datetime'].dt.date
    feature_df_days_rnn = feature_df.set_index(['hour_minute', 'date'], drop=True).sort_index()

    scaler = StandardScaler()
    feature_df_days_rnn[continuous_features] = scaler.fit_transform(feature_df_days_rnn[continuous_features])

    manager_df = pd.merge(feature_df_days_rnn, target_df_days_rnn, left_index=True, right_index=True, how='inner')

    return manager_df


# 一个时间窗口包含前五条数据
def manager_data_5():
    merge_data = pd.merge(target_df[[target, 'time']], feature_df[finally_features], left_on='time',
                          right_on='data_datetime', how='inner')

    logger.info(f'merge列：{merge_data.columns}')
    logger.info(f'merge形状：{merge_data.shape}')

    scaler = StandardScaler()
    merge_data[continuous_features] = scaler.fit_transform(merge_data[continuous_features])

    second_data = second_rnn_data()  # 第二RNN的输入数据

    seq_length = 2  # 时间窗口大小（用多少条历史数据预测未来的一条数据）

    X, X_minute, y = [], [], []
    X_2, X_minute_2, y_2 = [], [], []

    X_test, X_minute_test, y_test = [], [], []
    X_2_test, X_minute_2_test, y_2_test = [], [], []

    # merge_data.to_csv('kk看看这个merge_data是怎么个事.csv')

    for data_index in range(merge_data.shape[0] - seq_length):
        # if data_index < seq_length:
        #     continue

        loop_date = merge_data.iloc[data_index + seq_length]["time"].date()
        loop_time = merge_data.iloc[data_index + seq_length]["time"].time()
        # logger.warning(f'看看当前的时间：{loop_date} - {loop_time}')

        # if (loop_date <= datetime.datetime(2025, 6, 23).date()):
        if (loop_date <= datetime.datetime(2025, 6, 23).date()) or (
            loop_date >= datetime.datetime(2025, 8, 1).date() and
            loop_date <= datetime.datetime(2025, 8, 15).date()
        ):
            logger.warning(f'<{loop_date}> 日期跳过')
            continue

        # if data_index % 8 == 0 or data_index % 7 == 0:
        y__ = int(merge_data.iloc[
                      data_index + seq_length
                      ][target])
        test_set_flag = set()
        # if y__ != 0 and (data_index % 4 == 0 or data_index % 6 == 0) and (len(X_test) <= 45):
        if y__ != 0 and (data_index >= 1500 and data_index <= 1700):
            # logger.warning(f'当前数据loop_date为：{loop_date}')
            if len(second_data.loc[
                       (loop_time, [loop_date - datetime.timedelta(days=1), loop_date - datetime.timedelta(days=2)]),
                       'minute_of_day'
                   ].values.tolist()
                   ) != 2:
                continue

            X_2_test.append(
                second_data.loc[
                (loop_time, [loop_date - datetime.timedelta(days=1), loop_date - datetime.timedelta(days=2)]), :
                ][continuous_features].values
            )

            X_minute_2_test.append(
                list(
                    map(
                        lambda xx: int(xx),
                        second_data.loc[
                            (loop_time,
                             [loop_date - datetime.timedelta(days=1), loop_date - datetime.timedelta(days=2)]),
                            'minute_of_day'
                        ].values.tolist()
                    )
                )
            )

            y_2_test.append(
                second_data.loc[
                    (loop_time, loop_date), target
                ]
            )

            x_temp = merge_data.iloc[
                     data_index:data_index + seq_length
                     ][continuous_features].values

            X_test.append(
                x_temp
            )

            X_minute_test.append(
                list(map(lambda xx: int(xx), merge_data.iloc[
                                             data_index:data_index + seq_length
                                             ]['minute_of_day'].values.tolist()))
            )

            y_test.append(
                merge_data.iloc[
                    data_index + seq_length
                    ][target]
            )
        else:
            if len(second_data.loc[
                       (loop_time, [loop_date - datetime.timedelta(days=1), loop_date - datetime.timedelta(days=2)]),
                       'minute_of_day'
                   ].values.tolist()
                   ) != 2:
                continue

            X_2.append(
                second_data.loc[
                (loop_time, [loop_date - datetime.timedelta(days=1), loop_date - datetime.timedelta(days=2)]), :
                ][continuous_features].values
            )

            X_minute_2.append(
                list(
                    map(
                        lambda xx: int(xx),
                        second_data.loc[
                            (loop_time,
                             [loop_date - datetime.timedelta(days=1), loop_date - datetime.timedelta(days=2)]),
                            'minute_of_day'
                        ].values.tolist()
                    )
                )
            )

            y_2.append(
                second_data.loc[
                    (loop_time, loop_date), target
                ]
            )

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
    logger.info(rf'X_minute前十个预览：{X_minute[:10]}')
    # logger.info(rf'X_minute前十个预览：{X_minute}')
    logger.info(rf'y前十个预览：{y[:10]}')

    logger.info(rf'X_2前十个预览{X_2[:10]}')
    logger.info(rf'X_minute_2前十个预览{X_minute_2[:10]}')
    logger.info(rf'y_2前十个预览{y_2[:10]}')

    # for i in X:
    #     logger.success(f'检测数据条数：{len(i)}')
    #     logger.warning(f'检测数据条数：{len(i[0])}')

    # raise Exception('终止掉程序')

    return (
        X, X_minute, y,
        X_test, X_minute_test, y_test,
        X_2, X_minute_2, y_2,
        X_2_test, X_minute_2_test, y_2_test
    )


def train_model_save_model():
    # X, X_minute, y = manager_data()
    # X, X_minute, y = manager_data_5()
    (X_train, X_minute_train, y_train,
     X_test, X_minute_test, y_test,
     X_2_train, X_minute_2_train, y_2_train,
     X_2_test, X_minute_2_test, y_2_test) = manager_data_5()

    X_train = torch.FloatTensor(X_train)
    logger.info(f'看看正常的情况：{X_minute_train[:10]}')
    X_minute_train = torch.tensor(X_minute_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_test = torch.FloatTensor(X_test)
    X_minute_test = torch.tensor(X_minute_test, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    X_2_train = torch.FloatTensor(X_2_train)
    logger.info(f'看看什么情况：{X_minute_2_train}')
    X_minute_2_train = torch.tensor(X_minute_2_train, dtype=torch.long)
    y_2_train = torch.tensor(y_2_train, dtype=torch.long)

    X_2_test = torch.FloatTensor(X_2_test)
    X_minute_2_test = torch.tensor(X_minute_2_test, dtype=torch.long)
    y_2_test = torch.tensor(y_2_test, dtype=torch.long)

    # X = torch.FloatTensor(X)
    # X_minute = torch.tensor(X_minute, dtype=torch.long)
    # y = torch.tensor(y, dtype=torch.long)

    # train_size = int(0.8 * len(X))
    # X_train, X_test = X[:train_size], X[train_size:]
    # X_minute_train, X_minute_test = X_minute[:train_size], X_minute[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]

    # 初始化MLP
    model = RNNMLP(input_size=10, hidden_size=200, output_size=3, dropout=0.5, num_layers=2)
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

    X_2_train = X_2_train.to(device)
    X_minute_2_train = X_minute_2_train.to(device)
    # y_train = y_train.to(device)

    epochs = 1200
    tqdm_obj = tqdm(range(epochs))
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train, X_minute_train, X_2_train, X_minute_2_train)

        l2_regularization = 0
        for param in model.parameters():
            l2_regularization += torch.norm(param, 2)

        loss = criterion(outputs.squeeze(), y_train) + 0.01 * l2_regularization  # 去掉多余的维度
        loss.backward()
        optimizer.step()
        tqdm_obj.update(1)
        tqdm_obj.set_description(f'Epoch {epoch}, Loss: {loss.item():.4f}')
        # logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():

        predictions = model(X_train, X_minute_train, X_2_train, X_minute_2_train).argmax(dim=1)

        train_count = 0
        train_correct = 0
        for p, c in zip(predictions, y_train):
            train_count = train_count + 1
            if p == c:
                train_correct = train_correct + 1
        logger.warning(f'训练集准确率为：{train_correct / train_count}')

        # logger.info(f'看看这个所谓的测试集和对应的minute：\n{X_test.numpy().shape}\n{X_minute_test.numpy().shape}\n')
        minute_list = X_minute_test.numpy().tolist()
        minute_2_list = X_minute_2_test.numpy().tolist()

        X_test = X_test.to(device)
        X_minute_test = X_minute_test.to(device)
        X_2_test = X_2_test.to(device)
        X_minute_2_test = X_minute_2_test.to(device)
        y_test = y_test.to(device)
        predictions = model(X_test, X_minute_test, X_2_test, X_minute_2_test).argmax(dim=1)
        test_correct = 0
        test_count = 0
        for p, c, m, m2 in zip(predictions, y_test, minute_list, minute_2_list):
            test_count = test_count + 1
            if p == c:
                test_correct = test_correct + 1
                logger.success(f'{m} - {m2}时间：预测与实际值：{p} - {c}')
            else:
                logger.error(f'{m} - {m2}时间：预测与实际值：{p} - {c}')

        logger.warning(f'测试集准确率：{test_correct / test_count}')

    return test_correct / test_count


if __name__ == '__main__':
    result_list = []

    for i in range(3):
        logger.info(f'------------------------ 第 `{i + 1}` 次训练开始 ------------------------')
        rate = train_model_save_model()
        result_list.append(rate)
        logger.info(f'------------------------ 第 `{i + 1}` 次训练结束 ---------')

    logger.warning(f'3次训练的测试集准确率情况如下：\n{result_list}')
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
