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
from torch.utils.data import DataLoader

import net
import configure
from update import LocalUpdate
import accuracy
import knn

import torch._utils
# try:
#     torch._utils._rebuild_tensor_v2
# except AttributeError:
#     def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
#         tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
#         tensor.requires_grad = requires_grad
#         tensor._backward_hooks = backward_hooks
#         return tensor
#     torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def mnist_iid(dataset, num_users, num_imgs):
    idxs_dict = {}
    count = 0
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)#字典，类别：[数据集属于该类的ID]
        count += 1#count=60000
    for i in range(10):
        print(len(idxs_dict[i]))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    # num_imgs = 200
    for i in range(num_users):
        for j in range(10):
            a = np.array(idxs_dict[j][i*num_imgs:(i+1)*num_imgs])
            # print(i,j,a.shape)
            # print(a)
            dict_users[i] = np.concatenate((dict_users[i], a), axis=0)

    # for i in range(100):
    #     print(dict_users[i].shape)
    return dict_users

def mnist_extr_noniid(train_dataset, test_dataset, num_users, n_class, num_samples, rate_unbalance):
    print("num_users:",num_users)
    print("n_class",n_class)
    print("num_samples", num_samples)
    num_shards_train, num_imgs_train = int(60000/num_samples), num_samples
    num_classes = 10
    num_imgs_perc_test, num_imgs_test_total = 1000, 10000
    assert(n_class * num_users <= num_shards_train)
    assert(n_class <= num_classes)
    idx_class = [i for i in range(num_classes)]
    idx_shard = [i for i in range(num_shards_train)]
    dict_users_train = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards_train*num_imgs_train)
    # labels = dataset.train_labels.numpy()
    labels = np.array(train_dataset.targets)
    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)
    #labels_test_raw = np.array(test_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    #print(idxs_labels_test[1, :])

    # divide and assign
    for i in range(num_users):
        user_labels = np.array([])
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
            else:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
            unbalance_flag = 1
        user_labels_set = set(user_labels)
        #print(user_labels_set)
        #print(user_labels)
        for label in user_labels_set:
            dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test[int(label)*num_imgs_perc_test:int(label+1)*num_imgs_perc_test]), axis=0)
        #print(set(labels_test_raw[dict_users_test[i].astype(int)]))
    return dict_users_train, dict_users_test

def mnist_noniid(dataset, num_users,shard_per_user):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # num_shards, num_imgs = 200, 300

    #20clients  10classes   eachclient:3000   3000/10 = 300   每类分为20组
    #每个client 每类给300个
    num_shards, num_imgs = 1000, 60
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, shard_per_user, replace=False))#每个client 抽20个
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def noniid(dataset, num_users, shard_per_user, num_classes, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    count = 0
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)#字典，类别：[数据集属于该类的ID]
            count += 1#count=60000


    shard_per_class = int(shard_per_user * num_users / num_classes)#  (2*50)/10
    samples_per_user = int( count/num_users )#60000/50
    # whether to sample more test samples per user
    if (samples_per_user < 100):#无需更多数据
        double = True
    else:
        double = False

    for label in idxs_dict.keys():#lable=1---10中某一类的字典
        x = idxs_dict[label]#list数据，lable类中的数据ID
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))#换成shard_per_class行
        x = list(x)#10行

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])#拼接
        idxs_dict[label] = x        #lable：[ 10个 array]

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class #10个0 -- 9的列表
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))  #50,2的矩阵，每个user对应的两类

    # divide and assign
    for i in range(num_users):
        if double:
            rand_set_label = list(rand_set_all[i]) * 50
        else:
            rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)#从len中,不重复采样
            if (samples_per_user < 100 ):#and testb
                rand_set.append(idxs_dict[label][idx])
            else:
                rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)#id:data

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])#去除数组中的重复数字，并进行排序之后输出。
        test.append(value)
    test = np.concatenate(test)

    return dict_users, rand_set_all

args = configure.args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
print(args)

# if not os.path.exists(args.save_dir):
#     os.makedirs(args.save_dir)


# # train test
if args.dataset == 'mnist':
    # 5:5421
    trans_mnist = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
    # trans_mnist = transforms.Compose([transforms.ToTensor()])
    dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)  ## 只使用训练数据集
    dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
    # dl = DataLoader(dataset_train)
    # X = dl.dataset.data  # (60000,28, 28)
    # y = dl.dataset.targets  # (60000)
    # # normalize to have 0 ~ 1 range in each pixel
    # X = X / 255.0
elif args.dataset == 'cifar10':
    trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
    trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)#trans_cifar10_notreans
    dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
# dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
num_client = int(args.local_bs / args.shard_per_user)
dict_users_train, dict_users_test = mnist_extr_noniid(dataset_train, dataset_test, args.num_users, args.shard_per_user, num_client,1.0)
# dict_users_train = mnist_iid(dataset_train,args.num_users,num_client)

knn.adjacent(dict_users_train, dataset_train, args)
# A = np.load('./adjacent/user0.npy')



lens = np.ones(args.num_users)
print(len(dataset_train))
print("samples of one user:")
print(len(dict_users_train[0]))


net_glob = net.DSCNet(num_sample=args.local_bs, channels=args.channels, kernels=args.kernels)
net_glob.train()
print(net)
# net_glob.to(args.device)
# print(net_glob.state_dict().keys())

# load the pretrained weights which are provided by the original author in
# https://github.com/panji1990/Deep-subspace-clustering-networks
# 目前删去预训练模型
# ae_state_dict = torch.load('pretrained_weights_original/%s.pkl' % db)
# net_glob.ae.load_state_dict(ae_state_dict)
# print("Pretrained ae weights are loaded successfully.")


total_num_layers = len(net_glob.state_dict().keys())#state_dict 是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系.(如model的每一层的weights及偏置等等)
print("total_num_layers:")
print(total_num_layers)
print("net_glob.state_dict().keys():")
print(net_glob.state_dict().keys())
net_keys = [*net_glob.state_dict().keys()]
print("net_keys:")
print(net_keys)
# 自表达不上传
# w_glob_keys = [net_keys[i] for i in [0,1,2,3]]
# w_local_keys = net_glob.weight_keys[4]
# mycode
w_glob_keys = [net_keys[i] for i in [0,1]]#,2,3
print("w_glob_keys:")
print(w_glob_keys)

# for n,p in net_glob.named_parameters():
#     print( n,":",p.size())

num_param_glob = 0
num_param_local = 0
for key in net_glob.state_dict().keys():
    print(key)
    num_param_local += net_glob.state_dict()[key].numel()
    print("num_param_local={}".format(num_param_local))
    if key in w_glob_keys:
        print(key)
        num_param_glob += net_glob.state_dict()[key].numel()
percentage_param = 100 * float(num_param_glob) / num_param_local
print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
        num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
print("batch size: {}".format(args.local_bs))

# generate list of local models for each user
net_local_list = []
w_locals = {}
for user in range(args.num_users):
    w_local_dict = {}
    for key in net_glob.state_dict().keys():
        w_local_dict[key] =net_glob.state_dict()[key]
            #print(w_local_dict[key].shape)
    w_locals[user] = w_local_dict   #id:{各层：参数}

# training
indd = None      # indices of embedding for sent140
loss_train = []
acc_train = []
nmi_train = []
ami_train = []
ari_train = []
accs = []
times = []
accs10 = 0
accs10_glob = 0
start = time.time()
for iter in range(args.epochs+1):#每一轮
    w_glob = {}
    loss_locals = []
    acc_locals = []
    nmi_locals = []
    ami_locals = []
    ari_locals = []
    m = max(int(args.frac * args.num_users), 1)
    if iter == args.epochs:
        m = args.num_users
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    # idxs_users[0] = 1
    if iter%3==0:
        idxs_users = [11, 14, 19, 15, 13]
    print("idxs_users:{}".format(idxs_users))
    w_keys_epoch = w_glob_keys
    times_in = []
    total_len = 0
    for ind, idx in enumerate(idxs_users):  # 对每一位用户
        start_in = time.time()
        if args.epochs == iter:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])#dict_users_train[idx][:args.local_bs]     m_ft
            # idxs = dict_users_train[idx][:args.m_ft]#需要调整
        else:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])#[:args.local_bs]
        net_local = copy.deepcopy(net_glob)
        # net_local.to(args.device)
        w_local = net_local.state_dict()
        for k in w_locals[idx].keys():
            if k not in w_glob_keys:
                w_local[k] = w_locals[idx][k]
        net_local.load_state_dict(w_local)
        last = iter == args.epochs
        w_local, loss, acc, nmi, ami, ari, indd = local.dscn_train(model=net_local.to(args.device), w_glob_keys=w_glob_keys, idx=idx, giter=iter)
        # loss_locals.append(copy.deepcopy(loss))
        loss_locals.append(copy.deepcopy(loss))
        acc_locals.append(copy.deepcopy(acc))
        nmi_locals.append(copy.deepcopy(nmi))
        ami_locals.append(copy.deepcopy(ami))
        ari_locals.append(copy.deepcopy(ari))
        total_len += lens[idx]

        if len(w_glob) == 0:  # 每个客户端自己对应的global
            w_glob = copy.deepcopy(w_local)#copy dict
            for k, key in enumerate(net_glob.state_dict().keys()):
                w_glob[key] = w_glob[key] * lens[idx]
                w_locals[idx][key] = w_local[key]#update client model parameters
        else:
            for k, key in enumerate(net_glob.state_dict().keys()):
                if key in w_glob_keys:
                    w_glob[key] += w_local[key] * lens[idx]
                else:
                    w_glob[key] += w_local[key] * lens[idx]
                #global:加和
                w_locals[idx][key] = w_local[key]
        times_in.append(time.time() - start_in)

    loss_avg = sum(loss_locals) / len(loss_locals)
    acc_avg = sum(acc_locals) / len(acc_locals)
    nmi_avg = sum(nmi_locals) / len(nmi_locals)
    ami_avg = sum(ami_locals) / len(ami_locals)
    ari_avg = sum(ari_locals) / len(ari_locals)
    loss_train.append(loss_avg)
    acc_train.append(acc_avg)
    nmi_train.append(nmi_avg)
    ami_train.append(ami_avg)
    ari_train.append(ari_avg)

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
    #
    # acc_test, loss_test, mni_test = accuracy.test_img_local_all(net_glob, args, dataset_test, dict_users_test,
    #                                              w_glob_keys=w_glob_keys, w_locals=w_locals, indd=indd,
    #                                              dataset_train=dataset_train, dict_users_train=dict_users_train,
    #                                              return_all=False)
    # accs.append(acc_test)

    if iter != args.epochs:
        print('Round {:3d}, Avg loss: {:.3f}, Avg accuracy: {:.5f}, Avg nmi: {:.5f}, Avg ami: {:.5f}, Avg ari: {:.5f}'.format(
            iter, loss_avg, acc_avg, nmi_avg, ami_avg, ari_avg))
    else:
        # in the final round, we sample all users, and for the algs which learn a single global model, we fine-tune the head for 10 local epochs for fair comparison with FedRep
        print('Final Round, Avg loss: {:.3f}, Avg accuracy: {:.5f}, Avg nmi: {:.5f}, Avg ami: {:.5f}, Avg ari: {:.5f}'.format(
             loss_avg, acc_avg, nmi_avg, ami_avg, ari_avg))
        if iter >= args.epochs - 10 and iter != args.epochs:  # ?  -10
            accs10 += acc_avg / 10
print("acc: {}".format(acc_train))
print("nmi: {}".format(nmi_train))
print("ari: {}".format(ari_train))
print("ami: {}".format(ami_train))
print('Average accuracy final 10 rounds: {}'.format(accs10))
end = time.time()
print("all times:")
print(end-start)
print("times:")
print(times)

