# 本文件得到了目标值列：is_increase

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
import pandas as pd


engine_url = URL.create(
    drivername="mysql+pymysql",
    username=f"root",
    password=f'abc123',
    host=f"127.0.0.1",
    port=3306,
    database='yunqi'
)
conn = create_engine(engine_url)

fetch_his_data_sql = rf'''
    select * from pv_real_load where station_id = 1 
    
'''
# and time >= '2025-09-01 00:00:00' and time < '2025-10-31 23:59:59'

data_df = pd.read_sql(fetch_his_data_sql, conn)

data_df.sort_values('time', inplace=True)

# print(data_df.shape)
# print(data_df['time'].max(), data_df['time'].min())

data_df.loc[data_df['real_load'] < 1, 'real_load'] = 0  # 发电量小于1就置为 0

# data_df['last_step'] = data_df['real_load'].shift(1)
data_df['diff'] = data_df['real_load'].diff(2)
data_df['diff'] = data_df['diff'].fillna(0)  # 差值为NaN的替换为0

# 目标值编码，0：不升不降，1：上升，2：下降
# data_df['is_increase'] = np.where((data_df['diff'] > 0) & (data_df['diff'] != 0), 1, 2)
data_df['is_increase'] = np.where((data_df['diff'] > 0), 1, 2)
data_df['is_increase'] = np.where(data_df['diff'] == 0, 0, data_df['is_increase'])

data_df = data_df.loc[data_df['time'].dt.minute != 30]

data_df.loc[data_df['time'].dt.hour <= 4, 'real_load'] = 0
data_df.loc[data_df['time'].dt.hour <= 4, 'diff'] = 0
data_df.loc[data_df['time'].dt.hour <= 4, 'is_increase'] = 0

data_df.loc[data_df['time'].dt.hour >= 20, 'real_load'] = 0
data_df.loc[data_df['time'].dt.hour >= 20, 'diff'] = 0
data_df.loc[data_df['time'].dt.hour >= 20, 'is_increase'] = 0

# print(data_df[['time', 'real_load', 'diff', 'is_increase']].head(50))
#
# print('增长的时间段有：', data_df.loc[data_df['is_increase'] == 1].shape)
# print('为0的时间段有：', data_df.loc[data_df['is_increase'] == 0].shape)
# print('下降的时间段有：', data_df.loc[data_df['is_increase'] == -1].shape)


