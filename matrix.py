import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr
# import torch_geometric
import matplotlib.pyplot as plt
from torchvision import transforms

idxs = [ 5 ,13,  3,7 ,16]
giter = 14
a = 0#np.load('./matrix/user{}-{}.npy'.format(0, giter))
b = 0
aa = 0
# if a == 0:
for i in range(5):
    idx = idxs[i]
    if giter == 14:
    # for giter in range(20):
        image = np.load('./matrix/user{}-{}.npy'.format(idx, giter))
        labels = np.load('./matrix/userlabel{}-{}.npy'.format(idx, giter))
        if aa==0:
            aa = 1
            a = image
            b = labels
        else:
            a += image
        for i in range(len(image)):
            for j in range(i + 1, len(image)):
                if labels[i] > labels[j]:
                    image[[i, j], :] = image[[j, i], :]
                    image[:, [i, j]] = image[:, [j, i]]
                    temp = labels[i]
                    labels[i] = labels[j]
                    labels[j] = temp
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        # 彩色范围
        fig, ax = plt.subplots()
        # plt.xlim((0, 500))
        # plt.ylim((0, 500))
        # x_sticks = np.arange(0, 500, 50)
        plt.xticks(np.arange(0, 500, 50))
        plt.yticks(np.arange(0, 500, 50))
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        plt.imshow(image, cmap='rainbow', origin='upper', aspect="auto")
        plt.colorbar()
        plt.savefig('./image/{}-{}.jpg'.format(idx, giter))
        plt.savefig('./image/{}-{}.pdf'.format(idx, giter))
        plt.clf()


