import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
from torchvision import transforms

import torch.nn.functional as F
import post_clustering
import knn

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

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=False)#True

        if indd is not None:
            self.indd = indd
        else:
            self.indd = None

        self.dataset = dataset
        self.idxs = idxs
        #print("local idxs={}".format(len(idxs)))

    # def train(self, image, net, w_glob_keys, w_local_keys,  client_local, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.1):

    def dscn_train(self, model, w_glob_keys, idx, giter,  # type: DSCNet
              lr=1e-5):#1e-5    1e-4 1e-3    , show=10
        weight_coef = self.args.weight_coef
        weight_selfExp = self.args.weight_selfExp
        alpha = self.args.alpha
        dim_subspace = self.args.dim_subspace
        ro = self.args.ro


        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        local_eps = self.args.local_ep
        epoch_loss = []
        epoch_acc = []
        epoch_nmi = []
        epoch_ari = []
        epoch_ami = []
        epoch_fen = []

        num_updates = 0
        for iter in range(local_eps):

            batch_loss = []
            batch_acc = []
            batch_nmi = []
            correct = 0

            # 不确定用不用
            # for name, param in model.named_parameters():
            #     param.requires_grad = True
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)


                labels = labels.to('cpu').numpy()#'cpu'
                K = len(np.unique(labels))

                edge = np.load('./edge/user{}.npy'.format(idx))
                edge = torch.tensor(edge , dtype=torch.long)
                A = np.load('./adjacent/user{}.npy'.format(idx))
                # A = 0.5 * (A + A.T)
                A = torch.tensor(A)

                x_recon, z, z_recon = model(images, edge)#image:1000*1*28*28  , z_out

                C = model.self_expression.Coefficient.detach().to('cpu').numpy()
                # C = 0.5 * (C + C.T)
                # print((C >= 0).all())
                # print(np.all(np.abs(C - C.T) < 1e-08))
                C = torch.tensor(C)
                C = torch.clamp(C,min=0.0)

                # print(C)

                loss = model.loss_fn(idx, iter, images, x_recon, z, z_recon, weight_coef=weight_coef, weight_selfExp=weight_selfExp)

                a = 1
                c = 1
                loss1 = 1000000*torch.norm(a*A - c*C)
                # print("idx={},iter:{},loss1:{}".format(idx, iter, loss1))
                # print("loss:{}".format(loss))
                loss += loss1

                # print("loss1:{}".format(loss1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iter == local_eps - 1:


                    C1 = model.self_expression.Coefficient.detach().to('cpu').numpy()

                    np.save('./matrix/user{}-{}.npy'.format(idx, giter), C1)
                    np.save('./matrix/userlabel{}-{}.npy'.format(idx, giter), labels)

                    y_pred = post_clustering.spectral_clustering(C1, K, dim_subspace, alpha, ro)
                    epoch_loss.append(loss.item() / y_pred.shape[0])
                    epoch_acc.append(post_clustering.acc(labels, y_pred))
                    epoch_nmi.append(post_clustering.nmi(labels, y_pred))
                    epoch_ami.append(post_clustering.ami(labels, y_pred))
                    epoch_ari.append(post_clustering.ari(labels, y_pred))

                    # if idx == 1:
                    print("idx1--ACC={}".format(post_clustering.acc(labels, y_pred)))



        a = sum(epoch_loss) / len(epoch_loss)
        b = sum(epoch_acc) / len(epoch_acc)
        c = sum(epoch_nmi) / len(epoch_nmi)
        d = sum(epoch_ami) / len(epoch_ami)
        e = sum(epoch_ari) / len(epoch_ari)

        # print('User %02d: loss=%.4f, acc=%.4f, nmi=%.4f' %
        #       (idx, a, b, c))
        return model.state_dict(), a, b, c, d, e, self.indd



class DSCN_LocalUpdate(object):
    def __init__(self, args, orlx_users, orly_users):
        self.args = args
        self.orlx_users = orlx_users
        self.orly_users = orly_users

    # def train(self, image, net, w_glob_keys, w_local_keys,  client_local, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.1):

    def dscn_train(self, model, w_glob_keys, idx,  # type: DSCNet
              lr=1e-3):#, show=10
        weight_coef = self.args.weight_coef
        weight_selfExp = self.args.weight_selfExp
        alpha = self.args.alpha
        dim_subspace = self.args.dim_subspace
        ro = self.args.ro

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        x = self.orlx_users
        y = self.orly_users
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.args.device)
        x = x.to(self.args.device)
        if isinstance(y, torch.Tensor):
            y = y.to('cpu').numpy()
        K = len(np.unique(y))

        local_eps = self.args.local_ep
        epoch_loss = []
        num_updates = 0

        acc = 0

        for iter in range(local_eps):
            for name, param in model.named_parameters():
                param.requires_grad = True
            batch_loss = []

            x_recon, z, z_recon = model(x)
            loss = model.loss_fn(x, x_recon, z, z_recon, weight_coef=weight_coef, weight_selfExp=weight_selfExp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
            # if iter == local_eps-1:
            C = model.self_expression.Coefficient.detach().to('cpu').numpy()
            y_pred = post_clustering.spectral_clustering(C, K, dim_subspace, alpha, ro)
            print('User: %2d, Epoch %02d: loss=%.4f, acc=%.4f, nmi=%.4f' %
                  (idx, iter, loss.item() / y_pred.shape[0], post_clustering.acc(y, y_pred), post_clustering.nmi(y, y_pred)))
            acc = post_clustering.acc(y, y_pred)

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc