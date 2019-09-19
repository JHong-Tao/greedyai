from sklearn import datasets, linear_model # 引用 sklearn库，主要为了使用其中的线性回归模块

# 创建数据集，把数据写入到numpy数组
import numpy as np  # 引用numpy库，主要用来做科学计算
import matplotlib.pyplot as plt   # 引用matplotlib库，主要用来画图

# 这里我们通过numpy库创建了一个二维的数组，其中第一列是特征（身高），第二列是（体重）
# 我们可以看出这是（10，2）的大小，其中10代表的是数组的行数，也就是样本的个数， 2代表的是列数。
data = np.array([[152,51],[156,53],[160,54],[164,55],
                 [168,57],[172,60],[176,62],[180,65],
                 [184,69],[188,72]])

# 打印出数组的大小
print(data.shape)

# 这里我们分别读出x和y,其中x里存储的是特征，y中存储的是标签。
# 需要注意的一点是data[:,0]中添加了一个reshape的函数，主要的原因是在之后调用fit函数的时候对特征矩阵x是要求是矩阵的形式。
x,y = data[:,0].reshape(-1,1), data[:,1]

# 把生成的数据在二位空间里面做了可视化，这里调用的是matplotlib的库 从图里可以看出这些点具有一定的线性的关系

# TODO 1. 实例化一个线性回归的模型
regr = linear_model.LinearRegression()

# TODO 2. 在x,y上训练一个线性回归模型。 如果训练顺利，则regr会存储训练完成之后的结果模型
regr.fit(x, y)

# TODO 3. 画出身高与体重之间的关系
plt.scatter(x, y, color='red')

# 画出已训练好的线条
plt.plot(x, regr.predict(x), color='blue')

# 画x,y轴的标题
plt.xlabel('height (cm)')
plt.ylabel('weight (kg)')
plt.show() # 展示

# 利用已经训练好的模型去预测身高为163的人的体重
print ("Standard weight for person with 163 is %.2f"% regr.predict([[163]]))