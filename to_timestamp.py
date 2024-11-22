import pandas as pd

# 假设你的数据文件名为 'data.csv'
df = pd.read_excel('./Data/11-3/combine_2.xlsx')

# 将时间戳列转换为字符串格式
# 假设时间戳列名为 'timestamp'
df['timestamp'] = df['142__eDA'].astype(str)

# 使用 pd.to_datetime() 转换为 datetime 格式，指定格式
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d%H%M%S')

# 可选：将时间戳设置为索引
df.set_index('timestamp', inplace=True)

# 将修改后的 DataFrame 保存到新的 Excel 文件
df.to_excel('./Data/11-3/modified_combine.xlsx', index=True)

# 显示数据类型以确认转换成功
print(df.dtypes)
