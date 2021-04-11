from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set() # for plot styling

#X0 = np.loadtxt("../data/x_s.dat")
#X0 = np.loadtxt("../data/time_s.dat")
X0 = np.loadtxt("../data/Temp.dat")
X1 = np.loadtxt("../data/RDm.dat")
X2 = np.loadtxt("../data/RDa.dat")
X3 = np.loadtxt("../data/RVTm.dat")
X4 = np.loadtxt("../data/RVTa.dat")
X5 = np.loadtxt("../data/RVV.dat")

print(X0.shape)
print(X1.shape)
print(X2.shape)
print(X3.shape)
print(X4.shape)
print(X5.shape)

#plt.scatter(X0, X1[:,1], s=5)
#plt.scatter(X0, X2[:,1], s=5)
#plt.scatter(X0, X3[:,1], s=5)
#plt.scatter(X0, X4[:,1], s=5)
#plt.scatter(X0, X5[:,1], s=5)
#plt.show()

#X = np.array([ [X0, X1[:,1]], [X0, X2[:,1]], [X0, X2[:,1]], [X0, X3[:,1]], [X0, X4[:,1]], [X0, X5[:,1]] ])
X = np.array([X0, X1[:,1], X2[:,1], X3[:,1], X4[:,1], X5[:,1]])

#wcss = []for i in range(1, 11):
#    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#    kmeans.fit(X)
#    wcss.append(kmeans.inertia_)
#plt.plot(range(1, 11), wcss)
#plt.title('Elbow Method')
#plt.xlabel('Number of clusters')
#plt.ylabel('WCSS')
#plt.show()

kmeans = KMeans(n_clusters=5, random_state=666).fit(X)

plt.scatter(X0, X1[:,1], s=5)
plt.scatter(X0, X2[:,1], s=5)
plt.scatter(X0, X3[:,1], s=5)
plt.scatter(X0, X4[:,1], s=5)
plt.scatter(X0, X5[:,1], s=5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=10, c='red')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 2], s=10, c='green')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 3], s=10, c='blue')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 4], s=10, c='magenta')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 5], s=10, c='yellow')
#plt.show()

print(kmeans.labels_.shape)
print(kmeans.cluster_centers_.shape)

#kmeans.predict([[0, 0], [12, 3]])
