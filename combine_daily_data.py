import pandas as pd

# 读取 Excel 文件
df = pd.read_excel('./Data/11-3/modified_combine.xlsx')


# 筛选出时间为每天 00:00:00 的数据
df = df[df['timestamp'].dt.time != pd.to_datetime('00:00:00').time()]

# 通过时间戳列提取日期
df['date'] = df['timestamp'].dt.date

# 按日期分组处理数据
grouped = df.groupby('date')

# 用于存储处理后的结果
processed_data = []

# 遍历每一组数据（按日期分组）
for date, group in grouped:
    # 过滤出辐照度大于0的数据
    non_zero_group = group[group['38__eA'] > 0]

    # 计算辐照度数据的平均值
    avg_irradiance = round(non_zero_group['38__eA'].mean(), 3) if not non_zero_group.empty else 0

    # 计算日照时长（即非0辐照度数据的最大时间减去最小时间）
    if len(non_zero_group) > 0:  # 确保有非零辐照度数据
        duration = round((non_zero_group['timestamp'].max() - non_zero_group[
            'timestamp'].min()).total_seconds() / 3600, 3)  # 时长以小时为单位
    else:
        duration = 0  # 如果当天没有非零辐照度数据，日照时长为0

    # 获取辐照度的最大值
    max_irradiance = group['38__eA'].max()

    # 获取日总辐射曝辐量最大值
    daily_global_irradiance = group['47__eJ'].max()

    # 过滤出反射辐射辐照度大于0的数据
    reflect_non_zero_group = group[group['50__eM'] > 0]

    # 计算反射辐照度数据的平均值
    reflect_avg_irradiance = round(reflect_non_zero_group['50__eM'].mean(), 3) if not reflect_non_zero_group.empty else 0

    # 获取反射辐照度的最大值
    max_reflect_irradiance = group['54__eQ'].max()

    # 获取日反射曝辐量最大值
    daily_global_reflect_irradiance = group['59__eV'].max()

    # 过滤出直接辐射辐照度大于0的数据
    direct_non_zero_group = group[group['62__eY'] > 0]

    # 计算直接辐照度数据的平均值
    direct_avg_irradiance = round(direct_non_zero_group['62__eY'].mean(),
                                   3) if not direct_non_zero_group.empty else 0

    # 获取直接辐照度的最大值
    max_direct_irradiance = group['62__eY'].max()
    # 获取日直接曝辐量最大值
    daily_global_direct_irradiance = group['71__eAH'].max()



    # 过滤出散射辐射辐照度大于0的数据
    scattered_non_zero_group = group[group['75__eAL'] > 0]
    # 计算散射辐照度数据的平均值
    scattered_avg_irradiance = round(scattered_non_zero_group['75__eAL'].mean(),
                                  3) if not scattered_non_zero_group.empty else 0
    # 获取散射辐照度的最大值
    max_scattered_irradiance = group['75__eAL'].max()
    # 获取日散射曝辐量最大值
    daily_global_scattered_irradiance = group['84__eAU'].max()



    # 计算净辐照度数据的平均值
    net_avg_irradiance = round(group['87__eAX'].mean(), 3)
    # 获取净辐照度的最大值
    max_net_irradiance = group['87__eAX'].max()
    min_net_irradiance = group['87__eAX'].min()
    # 获取日净曝辐量最大值
    daily_global_net_irradiance = group['97__eBH'].max()



    #  过滤出紫外a辐射辐照度大于0的数据
    uva_non_zero_group = group[group['98__eBI'] > 0]
    # 计算紫外a辐照度数据的平均值
    uva_avg_irradiance = round(uva_non_zero_group['98__eBI'].mean(),
                                     3) if not uva_non_zero_group.empty else 0
    # 获取紫外a辐照度的最大值
    max_uva_irradiance = group['98__eBI'].max()
    #  过滤出紫外b辐射辐照度大于0的数据
    uvb_non_zero_group = group[group['105__eBP'] > 0]
    # 计算紫外b辐照度数据的平均值
    uvb_avg_irradiance = round(uva_non_zero_group['105__eBP'].mean(),
                               3) if not uvb_non_zero_group.empty else 0
    # 获取紫外b辐照度的最大值
    max_uvb_irradiance = group['105__eBP'].max()



    #  过滤出光合有效辐射辐照度大于0的数据
    ph_non_zero_group = group[group['113__eBX'] > 0]
    # 计算光合有效辐照度数据的平均值
    ph_avg_irradiance = round(ph_non_zero_group['113__eBX'].mean(),
                               3) if not ph_non_zero_group.empty else 0
    # 获取光合有效辐照度的最大值
    max_ph_irradiance = group['113__eBX'].max()



    # 计算大气长波辐照度数据的平均值
    lw_avg_irradiance = round(group['121__eCF'].mean(),
                              3)
    # 获取大气长波辐照度的最大值
    max_lw_irradiance = group['121__eCF'].max()



    # 计算地面长波辐照度数据的平均值
    glw_avg_irradiance = round(group['131__eCP'].mean(),
                              3)
    # 获取地面长波辐照度的最大值
    max_glw_irradiance = group['131__eCP'].max()


    # 将结果添加到列表中
    processed_data.append({
        'date': date,
        'average_irradiance': avg_irradiance,
        'sunshine_duration': duration,
        'max_irradiance': max_irradiance,
        'daily_global_irradiance': daily_global_irradiance,
        'reflect_avg_irradiance': reflect_avg_irradiance,
        'max_reflect_irradiance': max_reflect_irradiance,
        'daily_global_reflect_irradiance': daily_global_reflect_irradiance,
        'direct_avg_irradiance': direct_avg_irradiance,
        'max_direct_irradiance': max_direct_irradiance,
        'daily_global_direct_irradiance': daily_global_direct_irradiance,
        'scattered_avg_irradiance': scattered_avg_irradiance,
        'max_scattered_irradiance': max_scattered_irradiance,
        'daily_global_scattered_irradiance': daily_global_scattered_irradiance,
        'net_avg_irradiance': net_avg_irradiance,
        'max_net_irradiance': max_net_irradiance,
        'min_net_irradiance': min_net_irradiance,
        'daily_global_net_irradiance': daily_global_net_irradiance,
        'uva_avg_irradiance': uva_avg_irradiance,
        'max_uva_irradiance': max_uva_irradiance,
        'uvb_avg_irradiance': uvb_avg_irradiance,
        'max_uvb_irradiance': max_uvb_irradiance,
        'ph_avg_irradiance': ph_avg_irradiance,
        'max_ph_irradiance': max_ph_irradiance,
        'lw_avg_irradiance': lw_avg_irradiance,
        'max_lw_irradiance': max_lw_irradiance,
        'glw_avg_irradiance': glw_avg_irradiance,
        'max_glw_irradiance': max_glw_irradiance

    })

# 将处理后的数据转化为 DataFrame
result_df = pd.DataFrame(processed_data)

# 保存结果到新的 Excel 文件
result_df.to_excel('./Data/11-3/processed_data_new.xlsx', index=False)

print("Data has been processed and saved to 'processed_data.xlsx'.")
