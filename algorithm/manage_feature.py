# 整理出特征数据
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

fetch_weather_data_sql = rf'''
    select * from ems_caiyun_weather where station_id = 1 
    
'''
# and data_datetime >= '2025-09-01 00:00:00' and data_datetime < '2025-10-31 23:59:59'

feature_list = [
    'data_datetime', 'precipitation', 'cloudrate', 'dswrf',
    'visibility', 'precipitation_probability', 'apparent_temperature',
    'pressure', 'wind_speed', 'humidity'
]
weather_data_df = pd.read_sql(fetch_weather_data_sql, conn)[feature_list]

weather_data_df.sort_values('data_datetime', inplace=True)  # 按索引时间排序

# 取得差分值
weather_data_df['precipitation_diff'] = weather_data_df['precipitation'].diff()  # 降雨量
weather_data_df['cloudrate_diff'] = weather_data_df['cloudrate'].diff()  # 云层覆盖率
weather_data_df['dswrf_diff'] = weather_data_df['dswrf'].diff()  # 太阳辐射地面值
weather_data_df['visibility_diff'] = weather_data_df['visibility'].diff()  # 可见度
weather_data_df['precipitation_probability_diff'] = weather_data_df['precipitation_probability'].diff()  # 降雨概率
weather_data_df['apparent_temperature_diff'] = weather_data_df['apparent_temperature'].diff()  # 体感温度
weather_data_df['pressure_diff'] = weather_data_df['pressure'].diff()  # 气压
weather_data_df['wind_speed_diff'] = weather_data_df['wind_speed'].diff()  # 风速
weather_data_df['humidity_diff'] = weather_data_df['humidity'].diff()  # 湿度
weather_data_df['minute_of_day'] = weather_data_df['data_datetime'].apply(lambda x: x.hour * 60 + x.minute) / 60

weather_data_df.fillna(0, inplace=True)  # 差分的缺失值填充为0

# region 计算获取时间上与每天的地面辐射最大值相差的分钟数
dwrf_diff_df = weather_data_df[['data_datetime', 'dswrf']].copy()
dwrf_diff_df.set_index('data_datetime', inplace=True, drop=False)
dwrf_diff_df['data_date'] = pd.to_datetime(dwrf_diff_df['data_datetime'].dt.strftime('%Y-%m-%d'))
dwrf_diff_df['minute_of_day'] = dwrf_diff_df['data_datetime'].apply(lambda x: x.minute + x.hour * 60)

max_dswrf_datetime = dwrf_diff_df[['data_date', 'dswrf']].groupby(by='data_date')['dswrf'].idxmax()
result = pd.DataFrame(pd.to_datetime(max_dswrf_datetime.values), index=max_dswrf_datetime.index, columns=['max_dswrf_datetime'])
result['minute_of_day'] = result['max_dswrf_datetime'].apply(lambda x: x.minute + x.hour * 60)


def diff_fun(row):
    target_minute = result.loc[row['data_date'], 'minute_of_day']

    return target_minute - row['minute_of_day']

dwrf_diff_df['distance_max'] = dwrf_diff_df.apply(diff_fun, axis=1)
dwrf_diff_df = dwrf_diff_df['distance_max'].copy()

# endregion

weather_data_df = pd.merge(weather_data_df, dwrf_diff_df, left_on='data_datetime', right_index=True, how='inner')


if __name__ == '__main__':
    print(weather_data_df.shape)
    print(weather_data_df.columns)
    print(weather_data_df.head(50))
    print(weather_data_df.count())
    print(weather_data_df['data_datetime'].min(), weather_data_df['data_datetime'].max())

    continute_datetime = pd.date_range(
        weather_data_df['data_datetime'].min(),
        weather_data_df['data_datetime'].max(),
        freq='1h'
    )

    result = list(set(continute_datetime.to_list()) - set(weather_data_df['data_datetime'].to_list()))
    result.sort()
    result = list(map(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'), result))
    from pprint import pprint
    pprint(result)

