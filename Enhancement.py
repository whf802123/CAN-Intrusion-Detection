
import pandas as pd
import numpy as np
from scipy.stats import entropy as shannon_entropy

# ========== 配置参数 ==========
csv_in  = "C:\\Users\\Administrator\\Desktop\\Car-Hacking.csv"
csv_out = "C:\\Users\\Administrator\\Desktop\\enhanced-Car-Hacking.csv"
win_size_dt = 50    # Δt 滑窗大小
win_size_id = 100   # ID 熵滑窗大小

# ========== 读取数据 ==========
df = pd.read_csv(csv_in)

# ========== 基础特征 ==========
# 帧间到达时间 Δt 和瞬时频率
df['delta_time'] = df['Timestamp'].diff()
df['frequency'] = 1.0 / df['delta_time']

# 拆分 Data 字节，补足到 8 字节
def split_bytes(hex_str):
    if not isinstance(hex_str, str) or len(hex_str) % 2 != 0:
        return ['00'] * 8
    return [hex_str[i:i+2] for i in range(0, min(len(hex_str), 16), 2)] + ['00'] * (8 - len(hex_str) // 2)

bytes_df = df['Data'].apply(split_bytes).apply(pd.Series)
bytes_df.columns = [f'byte_{i}' for i in range(8)]

# 将十六进制字符串转换为整数
for col in bytes_df.columns:
    bytes_df[col] = bytes_df[col].map(lambda x: int(x, 16))

# 基本统计
df['data_mean'] = bytes_df.mean(axis=1)
df['data_std']  = bytes_df.std(axis=1)
df['data_max']  = bytes_df.max(axis=1)
df['data_min']  = bytes_df.min(axis=1)

# 熵 (byte 分布)
df['entropy'] = bytes_df.apply(
    lambda row: shannon_entropy(np.bincount(row, minlength=256) / 8.0),
    axis=1
)

# 是否全 0
df['is_all_zero'] = bytes_df.apply(lambda row: int((row == 0).all()), axis=1)

# ========== 新增特征 ==========
# 1. Hamming 权重
df['hamming_weight'] = bytes_df.apply(
    lambda row: sum(bin(b).count('1') for b in row),
    axis=1
)

# 2. Arbitration_ID 全局统计
# df['id_total_count'] = df['Arbitration_ID'].map(df['Arbitration_ID'].value_counts())
id_group = df.groupby('Arbitration_ID')['delta_time']
df['id_mean_period'] = id_group.transform('mean')
df['id_std_period']  = id_group.transform('std')

# 3. 滑动窗口 Δt 统计
df['rolling_dt_mean'] = df['delta_time'].rolling(win_size_dt, min_periods=1).mean()
df['rolling_dt_std']  = df['delta_time'].rolling(win_size_dt, min_periods=1).std()
# df['rolling_dt_skew'] = df['delta_time'].rolling(win_size_dt, min_periods=1).skew()
# df['rolling_dt_kurt'] = df['delta_time'].rolling(win_size_dt, min_periods=1).kurt()

# 4. 滑动窗口 Arbitration_ID 熵
df['id_code'] = pd.Categorical(df['Arbitration_ID']).codes
n_cat = int(df['id_code'].max()) + 1
df['rolling_id_entropy'] = (
    df['id_code']
      .rolling(win_size_id, min_periods=1)
      .apply(lambda arr: shannon_entropy(np.bincount(arr.astype(int), minlength=n_cat) / len(arr)), raw=False)
)

# 5. ID 切换标志
df['prev_id'] = df['Arbitration_ID'].shift()
df['id_switch'] = (df['Arbitration_ID'] != df['prev_id']).astype(int)
df.drop(columns=['prev_id'], inplace=True)

# 6. ID-载荷交互统计
payload_stats = bytes_df.join(df['Arbitration_ID']).groupby('Arbitration_ID').agg({
    **{f'byte_{i}': ['mean','std'] for i in range(8)}
})
payload_stats.columns = [f'{col[0]}_{col[1]}' for col in payload_stats.columns]
df = df.join(payload_stats, on='Arbitration_ID')

# 7. 累计出现次数
# df['id_cumcount'] = df.groupby('Arbitration_ID').cumcount()

# 删除辅助列
df.drop(columns=['id_code'], inplace=True)

df.to_csv(csv_out, index=False)
csv_out
