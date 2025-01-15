import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr
# import torch_geometric
import matplotlib.pyplot as plt
from torchvision import transforms

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, name=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.name = name

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # if self.name is None:
        #     image, label = self.dataset[self.idxs[item]]
        # elif 'femnist' in self.name:
        #     image = torch.reshape(torch.tensor(self.dataset['x'][item]), (1, 28, 28))
        #     label = torch.tensor(self.dataset['y'][item])
        # elif 'sent140' in self.name:
        #     image = self.dataset['x'][item]
        #     label = self.dataset['y'][item]
        # else:
        #     image, label = self.dataset[self.idxs[item]]
        return image, label


def euclidDistance(x1, x2, sqrt_flag=True):
    res = np.sum((x1-x2)**2)
    if sqrt_flag:
        res = np.sqrt(res)
    # res = torch.cosine_similarity(x1, x2, dim=1)
    return res

def calEuclidDistanceMatrix(X):
    X = np.array(X)
    S = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            r, p = pearsonr(X[i], X[j])
            S[i][j] = 1.0 * euclidDistance(X[i], X[j])#abs(r)#np.sqrt(np.sum(np.square(X[i][0]-X[j][0])))#
            S[j][i] = S[i][j]
    return S

def myKNN(S, k, sigma=1.0):
    N = len(S)
    A = np.zeros((N,N))
    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        #对距离进行排序
        dist_with_index = sorted(dist_with_index, reverse=False, key=lambda x:x[0])#欧式升序，皮尔逊降序  tuple(dist,编号)
        neighbours_id = [dist_with_index[m][1] for m in range(k+1)] # xi's k nearest neighbours
        #构建邻接矩阵
        for j in neighbours_id: # xj is xi's neighbour
            if i!=j:
                A[i][j] = 1#np.exp(-S[i][j]/2/sigma/sigma)
                A[j][i] = A[i][j] # mutually

    return A

def adjacent(dict_users_train, dataset_train, args):
    for i in range(len(dict_users_train)):
        ldr_train = DataLoader(DatasetSplit(dataset_train, dict_users_train[i]), batch_size=args.local_bs, shuffle=False)
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            images = images.to(args.device).numpy()
            images = np.reshape(images,(images.shape[0],-1))
            # labels = labels.to(args.device).numpy()
            # print("shape:")#(1000, 784)(1000,)
            # print(images.shape)
            # print(labels.shape)
        S = calEuclidDistanceMatrix(images)
        A = myKNN(S, k=6)# 100  它是对称矩阵吗
        np.save('./adjacent/user{}.npy'.format(i), A)


def adjacent_to_edge(args):
    for user in range(args.num_users):
        A = np.load('./adjacent/user{}.npy'.format(user))
        print(A.shape)
        edge = []
        N = len(A)
        for i in range(N):
            for j in range(N):
                if A[i][j] == 1:
                    temp = [i, j]
                    edge.append(temp)
        # print(array)
        # array = torch.Tensor(array)
        edge = np.array(edge)
        print(edge.shape)
        np.save('./edge/user{}.npy'.format(user), edge)


def plot(C1,labels,idx,giter):
    image = C1
    for i in range(len(image)):
        for j in range(i + 1, len(image)):
            if labels[i] > labels[j]:
                image[[i, j], :] = image[[j, i], :]
                image[:, [i, j]] = image[:, [j, i]]

    plt.matshow(image, cmap=plt.cm.gray)
    plt.xlim((0, 500))
    plt.ylim((0, 500))
    x_sticks = np.arange(0, 500, 50)
    # y_sticks = np.arange(500,0,50)
    plt.xticks(x_sticks)
    plt.yticks(x_sticks)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    plt.savefig('./image/{}-{}.jpg'.format(idx, giter))

