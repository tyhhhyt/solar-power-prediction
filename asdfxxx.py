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

fetch_sql = rf'''
    select data_datetime, dswrf from ems_caiyun_weather where station_id = 1
'''

skycon_df = pd.read_sql(fetch_sql, conn)
skycon_df['data_date'] = pd.to_datetime(skycon_df['data_datetime'].dt.strftime('%Y-%m-%d'))

skycon_df.set_index(keys='data_datetime', drop=False, inplace=True)
skycon_df['minute_of_day'] = skycon_df['data_datetime'].apply(lambda x: x.minute + x.hour * 60)
print(skycon_df.head(30))
print(skycon_df.shape)
# print(skycon_df[['data_datetime', 'dswrf']].resample('d').max().head(30))
max_dswrf_datetime = skycon_df[['data_date', 'dswrf', 'data_datetime']].groupby(by='data_date')['dswrf'].idxmax()
result = pd.DataFrame(pd.to_datetime(max_dswrf_datetime.values), index=max_dswrf_datetime.index, columns=['max_dswrf_datetime'])
result['minute_of_day'] = result['max_dswrf_datetime'].apply(lambda x: x.minute + x.hour * 60)
# print(result)

print(skycon_df.head())

def diff_fun(row):
    target_minute = result.loc[row['data_date'], 'minute_of_day']

    return target_minute - row['minute_of_day']

skycon_df['distance_max'] = skycon_df.apply(diff_fun, axis=1)

print(skycon_df.head(50))


