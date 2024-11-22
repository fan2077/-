import pandas as pd

# 读取数据
df = pd.read_csv('./Data/11-3/温湿度数据.csv')

# 假设你的时间戳列名是 'timestamp'，并且已经转换成了 datetime 格式
df['timestamp'] = pd.to_datetime(df['Datetime'], format='%Y/%m/%d %H:%M')  # 根据你的时间戳格式调整

# 筛选出3月31日之前的数据（包括3月31日）
df = df[df['timestamp'] <= '2024-03-31']

# 通过时间戳列提取日期
df['date'] = df['timestamp'].dt.date

# 按日期分组处理数据
grouped = df.groupby('date')

# 用于存储处理后的结果
processed_data = []

# 遍历每一组数据（按日期分组）
for date, group in grouped:
    tem = round(group['TEM'].mean(), 3)
    rhu = round(group['RHU'].mean(), 3)



    # 将结果添加到列表中
    processed_data.append({
        'date': date,
        'tem': tem,
        'rhu': rhu,
    })

# 将处理后的数据转化为 DataFrame
result_df = pd.DataFrame(processed_data)

# 保存结果到新的 Excel 文件
result_df.to_excel('./Data/11-3/processed_weather_data.xlsx', index=False)

print("Data has been processed and saved to 'processed_weather_data.xlsx'.")
