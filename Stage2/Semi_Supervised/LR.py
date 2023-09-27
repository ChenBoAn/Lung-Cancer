from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd

#! 一元線性回歸
'''
start = hu[coordinate[1], coordinate[0]] # 起始值
average = -173.34 + 0.71 * start # 平均值
std = 166.08 + 0.1 * average # 標準差
'''

pred = pd.read_excel('E:/Lung_Cancer/Lung_Nodule_Segmentation/final3/result/threshold1.xlsx')
pred = pred["std"].to_numpy().ravel()

true = pd.read_csv('E:/Lung_Cancer/Lung_Nodule_Segmentation/labels.csv').to_numpy().ravel()

# 計算均誤差（Mean absolute Error）、均方誤差（Mean Squared Error）和決定係數（R-squared）
mae = mean_absolute_error(true, pred)
mse = mean_squared_error(true, pred)
r2 = r2_score(true, pred)

print("均誤差（MAE）：", mae)  # 數字越小的模型，預測效果更好
print("均方誤差（MSE）：", mse)  # 數字越小的模型，預測效果更好
print("決定係數（R-squared）：", r2)  # 數字越接近1的模型，預測效果更好