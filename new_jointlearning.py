import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# 设置设备为GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后时间步的输出
        return out

# 2. 定义 BP 神经网络模型
class BPNeuralNet(nn.Module):
    def __init__(self):
        super(BPNeuralNet, self).__init__()
        self.fc1 = nn.Linear(22, 64)  # 输入层特征数量
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNNModel(nn.Module):
    def __init__(self, input_channels):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x

# 3. 训练函数
def train_model(model, criterion, optimizer, train_loader, num_epochs=300):
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs=inputs.to(device)
            targets=targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# 读取数据
data = pd.read_excel('./Data/11-3/processed_data_new.xlsx')  # 替换为你的数据文件路径

# 选择特征和目标
features = data[['average_irradiance', 'sunshine_duration', 'max_irradiance','daily_global_irradiance',
                 'reflect_avg_irradiance','daily_global_reflect_irradiance','scattered_avg_irradiance',
                 'max_scattered_irradiance','daily_global_scattered_irradiance','net_avg_irradiance',
                 'max_net_irradiance','min_net_irradiance','daily_global_net_irradiance','uva_avg_irradiance',
                 'max_uva_irradiance','uvb_avg_irradiance','max_uvb_irradiance','ph_avg_irradiance',
                 'max_ph_irradiance','lw_avg_irradiance','max_glw_irradiance','rhu']]
target = data['PV_power']

# 数据归一化
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
target_scaled = scaler.fit_transform(target.values.reshape(-1, 1))

# 设置时间步长
time_steps = 5 # 这里时间步长设置为5
X, y = [], []

# 构造时间序列数据集
for i in range(len(features_scaled) - time_steps + 1):
    X.append(features_scaled[i:(i + time_steps)])
    y.append(target_scaled[i + time_steps - 1])

X, y = np.array(X), np.array(y)

# 数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)


# print(X_test)
# print(y_test)

# 6. 数据加载
train_data = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=False)

# 7. 训练 LSTM
lstm_model = LSTMModel(input_size=22, hidden_size=64, num_layers=1).to(device)
lstm_criterion = nn.MSELoss().to(device)
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
train_model(lstm_model, lstm_criterion, lstm_optimizer, train_loader)

# 8. 训练 BP 神经网络
# 提取最后一个时间步的特征用于 BP 神经网络
X_train_bp = X_train[:, -1, :]  # 取最后一个时间步的特征
bp_model = BPNeuralNet().to(device)
bp_criterion = nn.MSELoss().to(device)
bp_optimizer = optim.Adam(bp_model.parameters(), lr=0.001)

# 创建 BP 训练数据集
bp_train_data = TensorDataset(torch.tensor(X_train_bp).float(), torch.tensor(y_train).float())
bp_train_loader = DataLoader(dataset=bp_train_data, batch_size=32, shuffle=False)

train_model(bp_model, bp_criterion, bp_optimizer, bp_train_loader)

#训练 CNN
X_train_cnn = X_train.transpose(0,2,1)
y_train_cnn = y_train
cnn_model = CNNModel(input_channels=22).to(device)  # 3个输入特征
cnn_criterion = nn.MSELoss().to(device)
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# CNN数据加载
cnn_train_data = TensorDataset(torch.tensor(X_train_cnn).float(), torch.tensor(y_train_cnn).float())
cnn_train_loader = DataLoader(dataset=cnn_train_data, batch_size=32, shuffle=False)

train_model(cnn_model, cnn_criterion, cnn_optimizer, cnn_train_loader)

# 9. 预测
lstm_model.eval()
bp_model.eval()
cnn_model.eval()

with torch.no_grad():
    lstm_pred = lstm_model(torch.tensor(X_test).to(device).float()).cpu()
    X_test_bp = X_test[:, -1, :]  # 对测试数据也做相同处理
    bp_pred = bp_model(torch.tensor(X_test_bp).to(device).float()).cpu()
    X_test_cnn = X_test.transpose(0,2,1)
    cnn_pred = cnn_model(torch.tensor(X_test_cnn).to(device).float()).cpu()  # CNN预测


# 10. 计算 RMSE
lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_pred.numpy()))
bp_rmse = np.sqrt(mean_squared_error(y_test, bp_pred.numpy()))
cnn_rmse = np.sqrt(mean_squared_error(y_test, cnn_pred.numpy()))
#
# 11. 计算总 RMSE 和权重
total_rmse = lstm_rmse + bp_rmse \
             + cnn_rmse
lstm_weight = lstm_rmse / total_rmse if total_rmse > 0 else 0
bp_weight = bp_rmse / total_rmse if total_rmse > 0 else 0
cnn_weight = cnn_rmse / total_rmse if total_rmse > 0 else 0
print(f'lstm weight: {lstm_weight:.4f}')
print(f'bp weight: {bp_weight:.4f}')
print(f'cnn weight: {cnn_weight:.4f}')



# 12. 加权预测
final_prediction = lstm_weight * lstm_pred.numpy() + bp_weight * bp_pred.numpy() + cnn_weight * cnn_pred.numpy()

# 13. 反归一化
final_prediction_inverse = scaler.inverse_transform(final_prediction.reshape(-1, 1))
y_test_inverse = scaler.inverse_transform(y_test)
# X_test_inverse = scaler.inverse_transform(X_test_bp)
print(final_prediction_inverse)
print(y_test_inverse)

# 14. 计算最终 RMSE
final_rmse = np.sqrt(mean_squared_error(y_test_inverse, final_prediction_inverse))
mse = mean_squared_error(y_test_inverse, final_prediction_inverse)
r2 = r2_score(y_test_inverse, final_prediction_inverse)
print(f'Final RMSE: {final_rmse:.4f}')
print(f'Final MSE: {mse:.4f}')
print(f'Final r2_score: {r2:.4f}')


# 绘制预测结果
import matplotlib.pyplot as plt

plt.plot(y_test_inverse, label='True Power')
plt.plot(final_prediction_inverse, label='Predicted Power')
plt.xlabel('Sample')
plt.ylabel('Solar Power')
plt.title('Solar Power Prediction')
plt.legend()
plt.show()