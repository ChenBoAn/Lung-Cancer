# 讀取檔案，載入資料
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
import pandas as pd
import pickle

# 分成訓練資料與測試資料
trainData = pd.read_csv('E:/Lung_Cancer/Lung_Nodule_Segmentation/lung.csv').to_numpy()  # 訓練資料
train_label = pd.read_csv('E:/Lung_Cancer/Lung_Nodule_Segmentation/labels.csv').to_numpy().ravel()  # 訓練資料類別標籤
testData = pd.read_csv('E:/Lung_Cancer/Lung_Nodule_Segmentation/lung.csv').to_numpy()  # 測試資料
test_label = pd.read_csv('E:/Lung_Cancer/Lung_Nodule_Segmentation/labels.csv').to_numpy().ravel()  # 測試資料類別標籤

# .to_numpy()=>把資料型態從表格(dataframe)轉換成陣列(array)
# .ravel()=>把二維陣列攤平成一維陣列

print(trainData)
print(train_label)
print(testData)
print(test_label)


# 匯入需要的套件

# 定義 Regression 模型
model = Ridge(alpha=1.0)

# 訓練模型
model.fit(trainData, train_label)

# 用模型進行預測
pred = model.predict(testData)
print(pred)

# 對模型進行評分
# 匯入需要的套件

# 計算均誤差（Mean absolute Error）、均方誤差（Mean Squared Error）和決定係數（R-squared）
mae = mean_absolute_error(test_label, pred)
mse = mean_squared_error(test_label, pred)
r2 = r2_score(test_label, pred)

print("均誤差（MAE）：", mae)  # 數字越小的模型，預測效果更好
print("均方誤差（MSE）：", mse)  # 數字越小的模型，預測效果更好
print("決定係數（R-squared）：", r2)  # 數字越接近1的模型，預測效果更好

excel = pd.read_excel("E:/Lung_Cancer/Lung_Nodule_Segmentation/MLR/analysis.xlsx")
excel['pred_std'] = pred
excel.to_excel("E:/Lung_Cancer/Lung_Nodule_Segmentation/MLR/analysis.xlsx", index=False)