import numpy as np
import sample_data_create
import data
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class answer():
    def __init__(self, X= data.data):
        if X == [[]]:
            print ('Testing data is being used now')
            self.X = sample_data_create.X
        else:
            self.X= np.array(X)

    def plotting_the_org_data(self):
        plt.scatter(
            self.X[:, 0], self.X[:, 1],c='white', marker='o',edgecolor='black', s=50)
        plt.show()

    def auto_K(self,distortions):
        des_distortions = []
        for i in range(0, len(distortions) - 1):
            if i == 0:
                des_distortions.append(0)
                continue
            des_distortions.append((distortions[i] - distortions[i - 1]) / distortions[i - 1])
        return des_distortions.index(min(des_distortions)) + 1

    def data_preprocessing(self):
        self.scaler = StandardScaler()
        self.X_std = self.scaler.fit_transform(self.X)

        distortions = []
        for i in range(1, 11):
            km = KMeans(
                n_clusters=i, init='random',
                n_init=10, max_iter=300,
                tol=1e-04, random_state=0
            )
            km.fit(self.X_std)
            distortions.append(km.inertia_)

        # plot
        plt.plot(range(1, 11), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.show()

        try:
            self.num_of_K = int(input('Please choose number of K based on the elbow method and the graph is shown\n'))
        except:
            print('number of K will be determined automatically')
            self.num_of_K = self.auto_K(distortions)

        print ('number of K= %d'% self.num_of_K)



    def k_mean_model(self, print_review= False):
        km = KMeans(
            n_clusters=self.num_of_K, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        self.y_km = km.fit_predict(self.X_std)
        self.X_org = self.scaler.inverse_transform(self.X_std)
        self.Y = self.scaler.inverse_transform(km.cluster_centers_)

        if print_review == True:
            print('1. cluster id')
            for x, y in zip(self.X_org, self.y_km):
                print(x, '-------->', y)

            print('\n')

            print('2. coordinates of centroids')
            for i in range(0, self.num_of_K):
                print('cluster %d:\t' % (i + 1), self.Y[i])

        return self.y_km, self.Y

    def visual_result(self):
        # plot the X and clusters
        for i in range(0, self.num_of_K):
            plt.scatter(
                self.X_org[self.y_km == i, 0], self.X_org[self.y_km == i, 1],
                s=50,
                marker='p', edgecolor='black',
                label='cluster %d' % (i+1))

        # plot the centroids
        plt.scatter(
            self.Y[:, 0], self.Y[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='centroids'
        )
        plt.legend(scatterpoints=1)
        plt.grid()
        plt.show()

if __name__ == '__main__':
    project = answer()
    project.plotting_the_org_data()
    project.data_preprocessing()
    answer_1, answer_2 = project.k_mean_model()
    project.visual_result()

    print ('answer_1: ', answer_1)
    print ('answer 2: ', answer_2)