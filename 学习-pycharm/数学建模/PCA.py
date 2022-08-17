# 数据处理
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# 绘图
import seaborn as sns
import matplotlib.pyplot as plt
# Bartlett's球状检验
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
# KMO检验
# 检查变量间的相关性和偏相关性，取值在0-1之间；KOM统计量越接近1，变量间的相关性越强，偏相关性越弱，因子分析的效果越好。
# 通常取值从0.6开始进行因子分析
from factor_analyzer.factor_analyzer import calculate_kmo

df = pd.read_csv(r"20_f.csv", index_col=0, header=None)
a = np.array(df)
a = a.T
df = a
df = pd.DataFrame(df)

chi_square_value, p_value = calculate_bartlett_sphericity(df)
print(chi_square_value, p_value)


kmo_all, kmo_model = calculate_kmo(df)
print(kmo_all)
print(kmo_model)
print(df)
sc = StandardScaler().fit_transform(df)
pca = PCA(n_components=0.8,svd_solver='full').fit(df)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
k1_spss = pca.components_ / np.sqrt(pca.explained_variance_.reshape(pca.n_components_, 1))
print(k1_spss)
