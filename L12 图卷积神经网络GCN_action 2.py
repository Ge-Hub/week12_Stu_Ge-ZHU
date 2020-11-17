"""
二手车价格预测
数据集：
used_car_train_20200313.csv
used_car_testA_20200313.csv
数据来自某交易平台的二手车交易记录
ToDo：给你一辆车的各个属性（除了price字段），预测它的价格

"""
# 导入
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') # 忽略版本问题


# 数据加载
train_data = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test_data = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
print(train_data.info)
print(test_data.info)

# 查看数据缺失值
print(train_data.isnull().any())
print(train_data.isnull().sum())

# 缺失值可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
missing = train_data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
plt.show()

# sample随机抽取
import missingno as msno
sample = train_data.sample(1000)
msno.matrix(sample)
plt.show()
msno.bar(sample)
plt.show()
msno.heatmap(sample)
plt.show()
msno.heatmap(sample)
plt.show()

## 输出数据的大小信息
print('训练集大小:',train_data.shape)
print('测试集大小:',test_data.shape)
# 显示notRepairedDamage的分布情况
print(train_data['notRepairedDamage'].value_counts())

# Price分布
import seaborn as sns
import scipy.stats as st
# 查看price的分布
y = train_data['price']

plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)

plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)

plt.title('log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)

sns.distplot(y)
print('Skewness:', y.skew())
print('Skewness:', y.kurt())

# price 直方图
plt.title('price histogram')
plt.hist(y, color='red')

import numpy as np
plt.hist(np.log(y), color='red')

#-----------------------------------------------
# regDate转换为汽车注册时间（即时间diff）
dates = pd.to_datetime(train_data['regDate'], format='%Y%m%d', errors='coerce')
min_date = pd.to_datetime('19910101', format='%Y%m%d')

train_data['regTime'] = (dates - min_date).dt.days
test_data['regTime'] = (pd.to_datetime(train_data['regDate'], format='%Y%m%d', errors='coerce') - min_date).dt.days

# createDate转换为汽车售卖时间（即时间diff）
train_data['creatTime'] = (pd.to_datetime(train_data['creatDate'], format='%Y%m%d', errors='coerce') - min_date).dt.days
test_data['creatTime'] = (pd.to_datetime(test_data['creatDate'], format='%Y%m%d', errors='coerce') - min_date).dt.days

# 定义汽车使用时间 
train_data['usedTime'] = train_data['creatTime'] - train_data['regTime']
test_data['usedTime'] = test_data['creatTime'] - test_data['regTime']

#修改异常数据
train_data['power'][train_data['power']>600]=600
test_data['power'][test_data['power']>600]=600

# 众数补全
train_data['notRepairedDamage'].replace('-', '0.0', inplace=True)
train_data['notRepairedDamage'] = train_data['notRepairedDamage'].astype('float64')
test_data['notRepairedDamage'].replace('-', '0.0', inplace=True)
test_data['notRepairedDamage'] = test_data['notRepairedDamage'].astype('float64')

# 提取数值类型特征列名
numerical_cols = train_data.select_dtypes(exclude='object').columns
# 选择特征列
feature_cols = [col for col in numerical_cols if col not in ['SaleID','name','price', 'seller']]

# 提取特征列
X_data = train_data[feature_cols]
Y_data = train_data['price']
X_test = test_data[feature_cols]

# 定义统计函数，方便后续信息统计
def show_stats(data):
    print('min :', np.min(data))
    print('max :', np.max(data))
    print('ptp :', np.ptp(data))
    print('mean:', np.mean(data))
    print('std :', np.std(data))
    print('var :', np.var(data))

# 缺省值用-1填补
X_data = X_data.fillna(-1)
X_test = X_test.fillna(-1)

# 神经网络实现预测
import tensorflow
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X = np.array(X_data)
y = np.array(Y_data).reshape(-1,1)
X_ = np.array(X_test)
X.shape, y.shape, X_.shape

# 数据规范化
ss = MinMaxScaler()
X = ss.fit_transform(X)
X_ = ss.transform(X_)

#切分数据集
x_train,x_test,y_train,y_test = train_test_split(X, y,test_size = 0.3)

model = keras.Sequential([
        keras.layers.Dense(200,activation='relu',input_shape=[X.shape[1]]), 
        keras.layers.Dense(300,activation='relu'), 
        keras.layers.Dense(200,activation='relu'), 
        keras.layers.Dense(1)])
model.compile(loss='mean_absolute_error', optimizer='Adam')


model.fit(x_train,y_train,batch_size = 2048,epochs=250)

from sklearn.metrics import mean_absolute_error

#比较训练集和测试集效果
mean_absolute_error(y_train,model.predict(x_train))
mean_absolute_error(y_test,model.predict(x_test))
y_=model.predict(X_)
show_stats(y_)

#结果输出
data_test_price = pd.DataFrame(y_,columns = ['price'])
results = pd.concat([test_data['SaleID'],data_test_price],axis = 1)
results.to_csv('results_MLP.csv',sep = ',',index = None)