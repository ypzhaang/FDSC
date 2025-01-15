# 针对coil dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from post_clustering import spectral_clustering, acc, nmi
import scipy.io as sio
import math
import random
import warnings
import os
import time
import copy
from torchvision import datasets, transforms

import net
import configure
from update import LocalUpdate,DSCN_LocalUpdate
import accuracy

import torch._utils

args = configure.args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
print(args)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # train test
orl = {}
if args.db == 'coil20':
    data = sio.loadmat('datasets/COIL20.mat')
    x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
    y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]



    # seed = 100
    # random.seed(seed)
    # random.shuffle(x)
    # random.seed(seed)
    # random.shuffle(y)
    # train:1000  test:440
    # train_x = x[:1440]
    # train_y = y[:1440]
    train_x = x[:1000]
    train_y = y[:1000]

    tr_step = 500#500
    # te_step = 44
    tr_tempx = [train_x[i:i + tr_step] for i in range(0,len(train_x),tr_step) ]
    tr_tempy = [train_y[i:i + tr_step] for i in range(0, len(train_y), tr_step)]
elif args.db == 'coil100':
    # load data
    data = sio.loadmat('datasets/COIL100.mat')
    x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']#7200*1*32*32  7200*1
    y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]  100类
    orl[0] = x
    orl[1] = y
    idxs_dict = {}
    count = 0
    for i in range(7200):
        label = y[i]
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)  # 字典，类别：[数据集属于该类的ID]
        count += 1  # count=60000
    # for i in range(10):
    #     print(len(idxs_dict[i]))
    dict_users = {i: np.array([], dtype='int64') for i in range(10)}
    num_imgs = 7
    for i in range(10):
        for j in range(100):
            a = np.array(idxs_dict[j][i * num_imgs:(i + 1) * num_imgs])
            # print(i,j,a.shape)
            # print(a)
            dict_users[i] = np.concatenate((dict_users[i], a), axis=0)
    orlx_users = {i: np.array([], dtype='int64') for i in range(10)}
    orly_users = {i: np.array([], dtype='int64') for i in range(10)}
    for i in range(10):
        a1 = []
        a2 = []
        for h in dict_users[i]:
            hh = x[h]
            hh = np.expand_dims(hh,axis=0)
            if len(a1)==0 :
                a1 = hh
            else:
                a1 = np.concatenate((a1, hh),axis=0)
            a2.append(y[h])
        orlx_users[i] =  np.array(a1)
        orly_users[i] = np.array(a2)


lens = np.ones(10)

# build net
db = args.db
# channels = [1, 15]
# kernels = [3]
# epochs = args.epochs#40
# weight_coef = 1.0
# weight_selfExp = 75

# post clustering parameters
# alpha = 0.04  # threshold of C
# dim_subspace = 12  # dimension of each subspace
# ro = 8  #
# warnings.warn("You can uncomment line#64 in post_clustering.py to get better result for this dataset!")

net_glob = net.DSCNet(num_sample=700, channels=args.channels, kernels=args.kernels)
net_glob.to(device)

# load the pretrained weights which are provided by the original author in
# https://github.com/panji1990/Deep-subspace-clustering-networks
ae_state_dict = torch.load('pretrained_weights_original/%s.pkl' % db)
net_glob.ae.load_state_dict(ae_state_dict)
print("Pretrained ae weights are loaded successfully.")


total_num_layers = len(net_glob.state_dict().keys())#state_dict 是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系.(如model的每一层的weights及偏置等等)
print("total_num_layers:")
print(total_num_layers)
print("net_glob.state_dict().keys():")
print(net_glob.state_dict().keys())
net_keys = [*net_glob.state_dict().keys()]
print("net_keys:")
print(net_keys)
# 自表达不上传
w_glob_keys = [net_keys[i] for i in [0,1]]
print("w_glob_keys:")
print(w_glob_keys)



num_param_glob = 0
num_param_local = 0
for key in net_glob.state_dict().keys():
    num_param_local += net_glob.state_dict()[key].numel()
    print("num_param_local={}".format(num_param_local))
    if key in w_glob_keys:
        num_param_glob += net_glob.state_dict()[key].numel()
percentage_param = 100 * float(num_param_glob) / num_param_local
print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
        num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
print("batch size: {}".format(args.local_bs))

# generate list of local models for each user
net_local_list = []
w_locals = {}
for user in range(10):
    w_local_dict = {}
    for key in net_glob.state_dict().keys():
        w_local_dict[key] =net_glob.state_dict()[key]
            #print(w_local_dict[key].shape)
    w_locals[user] = w_local_dict   #id:{各层：参数}

# training
indd = None      # indices of embedding for sent140
loss_train = []
accs = []
times = []
accs10 = 0
accs10_glob = 0
start = time.time()
accc = {}
for i in range(10):
    accc[i] = []
for iter in range(args.epochs+1):#每一轮
    w_glob = {}
    loss_locals = []
    m = 5
    idxs_users = np.random.choice(range(10), 5, replace=False)
    print("idxs_users:{}".format(idxs_users))
    w_keys_epoch = w_glob_keys
    times_in = []
    total_len = 0
    for ind, idx in enumerate(idxs_users):  # 对每一位用户
        start_in = time.time()
        if args.epochs == iter:
            local = DSCN_LocalUpdate(args=args, orlx_users=orlx_users[idx], orly_users=orly_users[idx])
            # idxs = dict_users_train[idx][:args.m_ft]#需要调整
        else:
            local = DSCN_LocalUpdate(args=args, orlx_users=orlx_users[idx], orly_users=orly_users[idx])
        net_local = copy.deepcopy(net_glob)
        w_local = net_local.state_dict()
        for k in w_locals[idx].keys():
            if k not in w_glob_keys:
                w_local[k] = w_locals[idx][k]
        net_local.load_state_dict(w_local)
        last = iter == args.epochs
        w_local, loss, acc1 = local.dscn_train(model=net_local.to(args.device), w_glob_keys=w_glob_keys, idx=idx)
        accc[idx].append(acc1)
        loss_locals.append(copy.deepcopy(loss))
        total_len += lens[idx]

        if len(w_glob) == 0:  # 每个客户端自己对应的global
            w_glob = copy.deepcopy(w_local)#copy dict
            for k, key in enumerate(net_glob.state_dict().keys()):
                w_glob[key] = w_glob[key] * lens[idx]
                w_locals[idx][key] = w_local[key]#update client model parameters
        else:
            for k, key in enumerate(net_glob.state_dict().keys()):
                w_glob[key] += w_local[key] * lens[idx]#global:加和
                w_locals[idx][key] = w_local[key]
        times_in.append(time.time() - start_in)

    loss_avg = sum(loss_locals) / len(loss_locals)
    loss_train.append(loss_avg)
    print("local loss:")
    print(loss_locals)

    # get weighted average for global weights  平均
    for k in net_glob.state_dict().keys():
        w_glob[k] = torch.div(w_glob[k], total_len)

    w_local = net_glob.state_dict()  # sever分发下去
    for k in w_glob.keys():
        w_local[k] = w_glob[k]

    if args.epochs != iter:
        net_glob.load_state_dict(w_glob)

    # if iter % args.test_freq == args.test_freq - 1 or iter >= args.epochs - 10:
    if times == []:
        times.append(max(times_in))
    else:
        times.append(times[-1] + max(times_in))

    # acc_test, loss_test, mni_test = accuracy.test_img_local_all(net_glob, args, dataset_test, dict_users_test,
    #                                              w_glob_keys=w_glob_keys, w_locals=w_locals, indd=indd,
    #                                              dataset_train=dataset_train, dict_users_train=dict_users_train,
    #                                              return_all=False)
    # accs.append(acc_test)
#
#     if iter != args.epochs:
#         print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.5f}, Test mni: {:.5f}'.format(
#             iter, loss_avg, loss_test, acc_test, mni_test))
#     else:
#         # in the final round, we sample all users, and for the algs which learn a single global model, we fine-tune the head for 10 local epochs for fair comparison with FedRep
#         print('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.5f}, Test loss: {:.5f}'.format(
#             loss_avg, loss_test, acc_test, mni_test))
#
#         if iter >= args.epochs - 10 and iter != args.epochs:  # ?  -10
#             accs10 += acc_test / 10
# print('Average accuracy final 10 rounds: {}'.format(accs10))
end = time.time()
print("all times:")
print(end-start)
print("times:")
print(times)
print("accc:")
print(accc)

