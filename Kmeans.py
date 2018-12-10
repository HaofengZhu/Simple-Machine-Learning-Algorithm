import numpy as np


class Kmeans:
    def __init__(self):
        pass

    def clustering(self,dataSet,K,max_iter=50):
        self.dataSet = dataSet
        self.K = K
        self.max_iter=max_iter
        m = self.dataSet.shape[0]
        #初始化中心点

        centroids = self.__createCenter()
        label = np.zeros(m, dtype=np.int)
        assement = np.zeros(m)
        changed = False
        iter_counter=0
        # 终止条件为中心点是否改变
        while changed and iter_counter<self.max_iter:
            old_centroids = np.copy(centroids)
            for i in range(m):
                min_dist, min_index = np.inf, -1
                for j in range(self.K):
                    dist = self.__distEclud(self.dataSet[i], centroids[j])
                    if dist < min_dist:
                        min_dist, min_index = dist, j
                        label[i] = j
                assement[i] = self.__distEclud(self.dataSet[i], centroids[label[i]]) ** 2

            # 更新中心点
            for m in range(self.K):
                centroids[m] = np.mean(self.dataSet[label == m], axis=0)
            #计算终止条件
            changed = self.__centerChange(old_centroids, centroids)
            iter_counter+=1
        return centroids, label

    def __centerChange(self, centroids1, centroids2):
        set1 = set([tuple(c) for c in centroids1])
        set2 = set([tuple(c) for c in centroids2])
        return not (set1 == set2)

    # 计算欧式距离
    def __distEclud(self, vecA, vecB):
        vecA = np.array(vecA)
        vecB = np.array(vecB)
        return np.sqrt(sum(np.power(vecA - vecB, 2)))

    # 随机生成初始的质心
    def __createCenter(self):
        n = self.dataSet.shape[1]
        centroids = np.zeros((self.K, n))
        #对dataSet的每一维计算一个在该纬度最大最小值之间的随机数
        for i in range(n):
            dmin, dmax = np.min(self.dataSet[:, i]), np.max(self.dataSet[:, i])
            centroids[:, i] = dmin + (dmax - dmin) * np.random.rand(self.K)
        return centroids




