import numpy as np
from numpy import exp
from scipy.optimize import leastsq

# 原始数据
data = np.array([80, 85, 88, 90, 95])

# 1. 数据预处理：一次累加（1-AGO）
data_cumsum = np.cumsum(data)

# 2. 建立GM(1,1)模型的数据准备
# 计算累加序列的均值序列，用于构造微分方程的系数矩阵
z = 0.5 * (data_cumsum[:-1] + data_cumsum[1:])

# 构建B矩阵和Y矩阵
B = np.vstack([-z, np.ones(len(z))]).T
Y = data[1:]

# 3. 参数估计：最小二乘法求解a和b
def residuals(p, y, x):
    a, b = p
    return y - (a*x + b)

# 初始参数猜测
p0 = [0.1, 20]

# 最小二乘法求解
plsq = leastsq(residuals, p0, args=(Y, z))

a, b = plsq[0]

# 4. 使用模型参数进行预测
# 构造预测函数
def predict(x0, a, b, n):
    return (x0 - b/a) * exp(-a*n) + b/a

# 预测未来一年的值，n=6表示从基期到预测期的时间长度
x0 = data[0]
n = 6
future_value = predict(x0, a, b, n)

print(future_value)
