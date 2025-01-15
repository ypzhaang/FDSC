import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time

import post_clustering

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        d = int(self.idxs[item])
        image, label = self.dataset[d]
        return image, label

def test_img_local(net_g, dataset, args, idx=None, indd=None, user_idx=-1, idxs=None):#, show=10
    weight_coef = args.weight_coef
    weight_selfExp = args.weight_selfExp
    alpha = args.alpha
    dim_subspace = args.dim_subspace
    ro = args.ro

    net_g.eval()
    test_loss = 0
    correct = 0
    mni = 0
    # print("ids={}".format(len(idxs)))
    data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=False)
    count = 1
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        # if data[0].shape != 100:
        #     break;
        target = target.to(args.device).numpy()  # 'cpu'
        K = len(np.unique(target))
        # print("data = {}".format(data.shape))
        x_recon, z, z_recon = net_g(data)
        # sum up batch loss
        loss = net_g.loss_fn(data, x_recon, z, z_recon, weight_coef=weight_coef, weight_selfExp=weight_selfExp)
        C = net_g.self_expression.Coefficient.detach().to(args.device).numpy()
        #print("C.shape={}".format(C.shape))
        y_pred = post_clustering.spectral_clustering(C, K, dim_subspace, alpha, ro)

        test_loss1 = loss.item() / y_pred.shape[0]
        test_loss += test_loss1
        accu = post_clustering.acc(target, y_pred)
        mnimni = post_clustering.nmi(target, y_pred)
        mni += mnimni
        correct += accu
        count += 1


    test_loss /= count
    accuracy = correct / count
    mni = mni / count
    return accuracy, test_loss, mni

def test_img_local_all(net, args, dataset_test, dict_users_test, w_locals=None, w_glob_keys=None, indd=None,
                       dataset_train=None, dict_users_train=None, return_all=False):
    tot = 0
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    mni_test_local = np.zeros(num_idxxs)
    for idx in range(num_idxxs):
        net_local = copy.deepcopy(net)#此时的Net是平均后的服务器模型
        if w_locals is not None:
            w_local = net_local.state_dict()
            for k in w_locals[idx].keys():
                if k not in w_glob_keys:#自己加的，调试
                    w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)
        net_local.eval()

        a, b, c = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx][:args.m_te])

        tot += len(dict_users_test[idx])
        acc_test_local[idx] = a * len(dict_users_test[idx])
        loss_test_local[idx] = b * len(dict_users_test[idx])
        mni_test_local[idx] = c * len(dict_users_test[idx])

        del net_local

    return sum(acc_test_local) / tot, sum(loss_test_local) / tot, sum(mni_test_local) / tot
