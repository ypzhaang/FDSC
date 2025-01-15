import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='DSCNet')
    parser.add_argument('--db', default='coil100',
                        choices=['coil20', 'coil100', 'orl', 'reuters10k', 'stl'])
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--show-freq', default=10, type=int)
    parser.add_argument('--ae-weights', default=None)
    parser.add_argument('--save-dir', default='results')
    parser.add_argument('--gpu', type=int, default=1, help="GPU ID, -1 for CPU")

    # DSC-Net
    parser.add_argument('--weight_coef', default=1.0)#100000     1000000    1.0
    parser.add_argument('--weight_selfExp', default=15)#75  #1     1     75
    parser.add_argument('--channels', default=[1, 15])#[1, 15]s
    parser.add_argument('--kernels', default=[3])#[3]
    parser.add_argument('--alpha', default=0.04)#0.04
    parser.add_argument('--dim_subspace', default=12)#12 dimension of each subspace
    parser.add_argument('--ro', default=8)


    # federated learning，都要手动变    尤其m_tr m_ft local_bs
    parser.add_argument('--num_users', default=20)#20
    parser.add_argument('--shard_per_user', default=5)



    parser.add_argument('--local_bs', default=500)#100  1000

    parser.add_argument('--num_classes', default=10)
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--local_ep', default=7)#5
    parser.add_argument('--frac', default=0.25)

    #不用
    # parser.add_argument('--m_tr', type=int, default=2000, help="maximum number of samples/user to use for training")#2500
    # parser.add_argument('--m_te', type=int, default=400, help="maximum number of samples/user to use for testing")

    args = parser.parse_args()
    return args
