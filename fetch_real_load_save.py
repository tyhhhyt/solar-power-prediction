import datetime
import traceback
from contextlib import contextmanager
import pymysql
import pymysql.cursors
import requests
from loguru import logger

# start_datetime = datetime.datetime(2025, 5, 1)
# end_datetime = datetime.datetime(2025, 7, 31)
start_datetime = datetime.datetime(2025, 9, 2)
end_datetime = datetime.datetime(2025, 10, 31)

base_url = 'http://118.178.18.56:8200/yunqi/e-server-device-service/index/runningChart/'

head = {
    'e-server-user': 'eyJhbGciOiJSUzI1NiJ9.eyJlLXNlcnZlci11c2VyIjoie1wiZ3JvdXBJZFwiOjEsXCJncm91cE5hbWVcIjpcIui2hee6p-euoeeQhuWRmFwiLFwiaGVhZFBpY1wiOlwiaHR0cDovLzExOC4xNzguMTguNTY6ODI1My91cGxvYWQvaW1hZ2VzL2Y4ODQwODA5LTUzMDgtNDhkOS1iMTNkLWMzNzAxMmE5OTdiYi5wbmdcIixcImlkXCI6MSxcImlwXCI6XCIxMjcuMC4wLjFcIixcImlzQWRtaW5cIjoxLFwidHJ1ZW5hbWVcIjpcIui2hee6p-euoeeQhuWRmFwifSIsImp0aSI6ImEyZjA5MzgyLTM2MmUtNGEyOC04YTM2LWVlZDVkOTJlOGQ5OSIsImV4cCI6MTc2MjYxNzYwMH0.XXII9dwaXeDOa3p2ta9ErUvW9V6K2I671kv6pB7p1OGuxIXI9vvy7yA-q8WMyItnujJvjmmnTGNhTA8kyTzHE6DTt7LcRtg6iVMLtYA2kwv5ZoXRbH5gVxvHc6bNGTtDSavoAUSgH0RD7gT5-htXqtbSf5to27JE1yZ-2GV0bugtW9wJg5O0M8Ka_rUXWTDBsXnGFyOvdowl-FnGJHBxDr9a4OtxPONety005QkxC6P3T12mch3WxZ47wDJ-ed6Gmn0i3ayTPf9MWwERhSbbGOvXWptDKTKVVoDGNn07BMB6G5S7ZZUr0oz4HU8gW6mVcygo3E0VBZUV5QJOujCxfw',
    'token': 'yunqi'
}


@contextmanager
def get_conn_to_mysql():
    mysql_conn = None
    for retry_conn_mysql in range(10):
        try:
            mysql_conn = pymysql.connect(
                user=f"root",
                password="abc123",
                host=f"127.0.0.1",
                port=3306,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,  # 不使用返回元组  使用返回字典(传输效率降低)
                db="yunqi"
            )
            if mysql_conn:
                break
            else:
                continue
        except:
            logger.error(f"数据库连接第 `{retry_conn_mysql + 1}` 次失败")
            continue

    if not mysql_conn:
        logger.error(f'尝试连接数据库 10 次后失败')
        return None

    try:
        yield mysql_conn
    finally:
        mysql_conn.close()


for i in range((end_datetime - start_datetime).days):  # 不包括结束日期
    loop_datetime = (start_datetime + datetime.timedelta(i)).strftime('%Y-%m-%d')
    loop_url = base_url + f'BDNv3mmWDaDniBMzof8M4Q==?startTime={loop_datetime}'

    for retry in range(5):
        try:
            result_json = requests.get(loop_url, headers=head).json()
        except:
            logger.error('*** 请求接口报错 ***')
            logger.error(traceback.format_exc())

    real_load_list = result_json['data']['pvPowerCharts']

    save_data = []
    for item in real_load_list:
        save_inner_list = []
        save_inner_list.append(item['ts'])
        save_inner_list.append(1)
        if item['value'] is None:
            save_inner_list.append(None)
        else:
            save_value = float(item['value'])
            if save_value < 0 or save_value > 589:  # 过滤掉异常值
                save_inner_list.append(None)
            else:
                save_inner_list.append(save_value)

        save_data.append(save_inner_list)

    save_sql = rf'''
        insert into pv_real_load (time, station_id, real_load) values (%s, %s, %s)
    '''

    with get_conn_to_mysql() as mysql_conn:
        with mysql_conn.cursor() as cursor:
            logger.info(save_data)
            cursor.executemany(save_sql, save_data)

        mysql_conn.commit()

    logger.success(f'保存 {loop_datetime} 的数据成功')

