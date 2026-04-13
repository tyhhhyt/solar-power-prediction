import datetime
from loguru import logger
import pandas as pd
from manage_feature import weather_data_df as feature_df
from analyze_data import data_df as target_df
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


target = 'is_increase'
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


# print(feature_df.head(24))
# print(target_df[(target_df['time'] >= '2025-06-22') & (target_df['time'] < '2025-06-23')])
# print('*-*-' * 20)
# print(target_df[(target_df['time'] >= '2025-06-23') & (target_df['time'] < '2025-06-24')])
# print(target_df[24:48])
target_df.loc[target_df['time'].dt.hour <= 4, 'real_load'] = 0
target_df.loc[target_df['time'].dt.hour <= 4, 'diff'] = 0
target_df.loc[target_df['time'].dt.hour <= 4, 'is_increase'] = 0
# print(target_df[target_df['time'].dt.hour <= 4].head(50))
# target_df['hour_minute'] = target_df['time'].dt.strftime('%H:%M:%S')
target_df['hour_minute'] = target_df['time'].dt.time
target_df['date'] = target_df['time'].dt.date


target_df_days_rnn = target_df.set_index(['hour_minute', 'date'], drop=True).sort_index()
# print(target_df_days_rnn)
start_date_index = datetime.datetime.strptime('2025-06-22', '%Y-%m-%d').date()
end_date_index = datetime.datetime.strptime('2025-08-01', '%Y-%m-%d').date()
# print(target_df_days_rnn.loc[(slice(None), slice(start_date_index, end_date_index)), :])

feature_df['hour_minute'] = feature_df['data_datetime'].dt.time
feature_df['date'] = feature_df['data_datetime'].dt.date
feature_df_days_rnn = feature_df.set_index(['hour_minute', 'date'], drop=True).sort_index()
# print(feature_df_days_rnn.index.get_level_values(0).nunique())
# print(feature_df_days_rnn.index.get_level_values(1).nunique())
# print(type(feature_df_days_rnn.index))


def manager_data_5():
    scaler = StandardScaler()
    feature_df_days_rnn[continuous_features] = scaler.fit_transform(feature_df_days_rnn[continuous_features])

    manager_df = pd.merge(feature_df_days_rnn, target_df_days_rnn, left_index=True, right_index=True, how='inner')

    # manager_df.to_csv('看看数据.csv')
    print('看看manager_df', manager_df.head(10))
    seq_length = 2  # 时间窗口大小（用多少条历史数据预测未来的一条数据）

    X, X_minute, y = [], [], []
    X_test, X_minute_test, y_test = [], [], []
    hour_count = manager_df.index.get_level_values(0).nunique()  # 第一层索引长度
    date_count = manager_df.index.get_level_values(1).nunique()  # 第二层索引长度
    # for hour_index in range(hour_count + date_count - seq_length):
    for item_index, hour_index in enumerate(range(manager_df.shape[0] - seq_length)):
        x_temp = manager_df.iloc[hour_index:hour_index + seq_length][continuous_features].values

        if item_index % 7 != 0 and item_index % 8 != 0 and item_index % 9 != 0:
            X.append(
                x_temp
            )

            X_minute.append(
                list(map(lambda xx: int(xx),
                    manager_df.iloc[
                     hour_index:hour_index + seq_length
                    ]['minute_of_day'].values.tolist())
                )
            )

            y.append(
                manager_df.iloc[
                    hour_index + seq_length
                ][target]
            )
        else:
            X_test.append(
                x_temp
            )

            X_minute_test.append(
                list(map(lambda xx: int(xx),
                         manager_df.iloc[
                         hour_index:hour_index + seq_length
                         ]['minute_of_day'].values.tolist())
                     )
            )

            y_test.append(
                manager_df.iloc[
                    hour_index + seq_length
                    ][target]
            )
    logger.info(rf'特征批次长度为：{len(X)}')
    logger.info(rf'目标值批次长度为：{len(y)}')
    logger.info(rf'X前十个预览：{X[:10]}')
    logger.info(rf'X_minute前十个预览：{X_minute[:10]}')
    logger.info(rf'y前十个预览：{y}')
    # for i in X:
    #     logger.success(f'检测数据条数：{len(i)}')
    #     logger.warning(f'检测数据条数：{len(i[0])}')

    return X, X_minute, y, X_test, X_minute_test, y_test


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

        out = self.fc1(out[:, -1, :])

        return out


def train_test_set_split(x, x_minute, y):
    train_size_1 = 20
    test_size_1 = 40
    X_train_1, X_test_1 = x[:train_size_1], x[train_size_1:test_size_1]

    train_size_2 = 30
    test_size_2 = 80
    X_train_2, X_test_2 = x[test_size_1:test_size_1 + train_size_2], x[test_size_1 + train_size_2:test_size_2]


    X_train_2, X_test_2 = x[test_size_1:test_size_1 + train_size_2], x[test_size_1 + train_size_2:test_size_2]





def train_model_save_model():
    # X, X_minute, y = manager_data()
    # X, X_minute, y, X_test, X_minute_test, y_test = manager_data_5()
    X_train, X_minute_train, y_train, X_test, X_minute_test, y_test = manager_data_5()

    X_train = torch.FloatTensor(X_train)
    X_minute_train = torch.tensor(X_minute_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_test = torch.FloatTensor(X_test)
    X_minute_test = torch.tensor(X_minute_test, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # train_size = int(0.8 * len(X))
    # X_train, X_test = X[:train_size], X[train_size:]
    # X_minute_train, X_minute_test = X_minute[:train_size], X_minute[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]

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

    epochs = 500
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
    # manager_data_5()
    # print(aa)
