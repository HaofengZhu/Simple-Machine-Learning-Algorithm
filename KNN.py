import numpy as np
from math import sqrt
import operator as opt

class KNN:
    def __init__(self,K=5):
        self.K=K
        pass

    def fit(self,dataSet,labels):
        self.dataSet=np.array(dataSet)
        self.labels=np.array(labels)

    def predict(self,X):
        X=np.array(X)
        predict_label=[]
        for i in range(len(X)):
            distSquareMat=[[] for j in range(len(self.dataSet))]  # 计算差值的平方
            for j in range(len(distSquareMat)):
                distSquareMat[j]= (self.dataSet[j] - X[i]) ** 2
            distSquareMat=np.array(distSquareMat)
            distSquareSums = distSquareMat.sum(axis=1)  # 求每一行的差值平方和
            distances = distSquareSums ** 0.5  # 开根号，得出每个样本到测试点的距离

            sortedIndices = distances.argsort()  # 排序，得到排序后的下标
            indices = sortedIndices[:self.K]  # 取最小的k个
            labelCount = {}  # 存储每个label的出现次数
            for i in indices:
                label = self.labels[i]
                labelCount[label] = labelCount.get(label, 0) + 1  # 次数加一
            sortedCount = sorted(labelCount.items(), key=opt.itemgetter(1), reverse=True)
            # 对label出现的次数从大到小进行排序
            predict_label.append(sortedCount[0][0]) # 返回出现次数最大的label
        return predict_label
    def setParams(self,K=None):
        if K!=None:
            self.K=K