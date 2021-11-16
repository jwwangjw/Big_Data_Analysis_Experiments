class KMeans:
    def __init__(self, n_clusters: int = 8, init: str = 'k-means++', max_iter: int = 300, tol: float = 0.0001):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol

        self.cluster_centers_ = None
        self.dist = None
        self.labels_ = None

    def __init_centroids(self, X_train):
        n_sample, n_feature = X_train.shape

        if self.init == 'random':
            f_mean = np.mean(X_train, axis=0)
            f_std = np.std(X_train, axis=0)
            self.cluster_centers_ = f_mean + np.random.randn(self.n_clusters, n_feature) * f_std

        elif self.init == 'k-means++':
            idx = np.random.randint(0, n_sample)
            self.cluster_centers_ = [X_train[idx, :]]
            for i in range(1, self.n_clusters):
                dist = np.zeros((n_sample, len(self.cluster_centers_)))
                for cent_idx in range(len(self.cluster_centers_)):
                    dist[:, cent_idx] = np.linalg.norm(
                        X_train - self.cluster_centers_[cent_idx], axis=1)
                dist = np.min(dist, axis=1)
                p = dist / np.sum(dist)

                next_cent_idx = np.random.choice(n_sample, p=p)
                self.cluster_centers_.append(X_train[next_cent_idx])
            self.cluster_centers_ = np.array(self.cluster_centers_)

    def fit(self, X_train):
        n_sample, n_feature = X_train.shape
        self.__init_centroids(X_train)
        self.dist = np.zeros((n_sample, self.n_clusters))
        cent_pre = np.zeros(self.cluster_centers_.shape)
        cent_move = np.linalg.norm(self.cluster_centers_ - cent_pre)
        epoch = 0
        from copy import deepcopy
        while epoch < self.max_iter and cent_move > self.tol:
            epoch += 1
            for i in range(self.n_clusters):
                self.dist[:, i] = np.linalg.norm(X_train - self.cluster_centers_[i], axis=1)
            self.labels_ = np.argmin(self.dist, axis=1)
            cent_pre = deepcopy(self.cluster_centers_)
            for i in range(self.n_clusters):
                self.cluster_centers_[i] = np.mean(X_train[self.labels_ == i], axis=0)

            cent_move = np.linalg.norm(self.cluster_centers_ - cent_pre)

    def fit_predict(self, X_test):
        n_sample = X_test.shape[0]
        dist_test = np.zeros((n_sample, self.n_clusters))

        for i in range(self.n_clusters):
            dist_test[:, i] = np.linalg.norm(X_test - self.cluster_centers_[i], axis=1)
        clus_pred = np.argmin(dist_test, axis=1)

        return clus_pred

    def score(self, y_test, clus_pred):
        count=0
        for i in range(len(y_test)):
            if clus_pred[i]==y_test[i]:
                count=count+1
        return count/len(y_test)




if __name__ == '__main__':
    import numpy as np
    from pandas import read_csv
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    df = read_csv('C:/Users/lenovo/Desktop/Iris.csv')
    data1 = np.asarray(list(df['PetalLengthCm']))
    data2 = np.asarray(list(df['PetalWidthCm']))
    data3 = np.asarray(list(df['SepalLengthCm']))
    data4 = np.asarray(list(df['SepalWidthCm']))
    data5 = np.asarray(list(df['Species']))
    dat_l = []
    speices=[]
    for i in range(len(data1)):
        data = []
        data.append(data3[i])
        data.append(data4[i])
        data = np.asarray(list(data))
        dat_l.append(data)
        if data5[i] == 'Iris-setosa':
            speices.append(0)
        elif data5[i] == 'Iris-versicolor':
            speices.append(1)
        else:
            speices.append(2)
    data = np.asarray(dat_l)
    speices=np.asarray(speices)
    X_train, X_test,y_train,y_test = train_test_split(data,speices, random_state=1,test_size=0.3)
    print(y_test)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_train)

    clus_pred = kmeans.fit_predict(X_test)
    print(clus_pred)
    print(kmeans.score(y_test,clus_pred))
