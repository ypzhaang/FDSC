import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from post_clustering import spectral_clustering, acc, nmi
import scipy.io as sio
import math
import matplotlib.pyplot as plt
from torchvision import transforms
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class Conv2dSamePad(nn.Module):
    """
    Implement Tensorflow's 'SAME' padding mode in Conv2d.
    When an odd number, say `m`, of pixels are need to pad, Tensorflow will pad one more column at right or one more
    row at bottom. But Pytorch will pad `m+1` pixels, i.e., Pytorch always pads in both sides.
    So we can pad the tensor in the way of Tensorflow before call the Conv2d module.
    """

    def __init__(self, kernel_size, stride):
        super(Conv2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        out_height = math.ceil(float(in_height) / float(self.stride[0]))
        out_width = math.ceil(float(in_width) / float(self.stride[1]))
        pad_along_height = ((out_height - 1) * self.stride[0] + self.kernel_size[0] - in_height)
        pad_along_width = ((out_width - 1) * self.stride[1] + self.kernel_size[1] - in_width)
        pad_top = math.floor(pad_along_height / 2)
        pad_left = math.floor(pad_along_width / 2)
        pad_bottom = pad_along_height - pad_top
        pad_right = pad_along_width - pad_left
        return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], 'constant', 0)


class ConvTranspose2dSamePad(nn.Module):
    """
    This module implements the "SAME" padding mode for ConvTranspose2d as in Tensorflow.
    A tensor with width w_in, feed it to ConvTranspose2d(ci, co, kernel, stride), the width of output tensor T_nopad:
        w_nopad = (w_in - 1) * stride + kernel
    If we use padding, i.e., ConvTranspose2d(ci, co, kernel, stride, padding, output_padding), the width of T_pad:
        w_pad = (w_in - 1) * stride + kernel - (2*padding - output_padding) = w_nopad - (2*padding - output_padding)
    Yes, in ConvTranspose2d, more padding, the resulting tensor is smaller, i.e., the padding is actually deleting row/col.
    If `pad`=(2*padding - output_padding) is odd, Pytorch deletes more columns in the left, i.e., the first ceil(pad/2) and
    last `pad - ceil(pad/2)` columns of T_nopad are deleted to get T_pad.
    In contrast, Tensorflow deletes more columns in the right, i.e., the first floor(pad/2) and last `pad - floor(pad/2)`
    columns are deleted.
    For the height, Pytorch deletes more rows at top, while Tensorflow at bottom.
    In practice, we usually want `w_pad = w_in * stride`, i.e., the "SAME" padding mode in Tensorflow,
    so the number of columns to delete:
        pad = 2*padding - output_padding = kernel - stride
    We can solve the above equation and get:
        padding = ceil((kernel - stride)/2), and
        output_padding = 2*padding - (kernel - stride) which is either 1 or 0.
    But to get the same result with Tensorflow, we should delete values by ourselves instead of using padding and
    output_padding in ConvTranspose2d.
    To get there, we check the following conditions:
    If pad = kernel - stride is even, we can directly set padding=pad/2 and output_padding=0 in ConvTranspose2d.
    If pad = kernel - stride is odd, we can use ConvTranspose2d to get T_nopad, and then delete `pad` rows/columns by
    ourselves; or we can use ConvTranspose2d to delete `pad - 1` by setting `padding=(pad - 1) / 2` and `ouput_padding=0`
    and then delete the last row/column of the resulting tensor by ourselves.
    Here we implement the former case.
    This module should be called after the ConvTranspose2d module with shared kernel_size and stride values.
    And this module can only output a tensor with shape `stride * size_input`.
    A more flexible module can be found in `yaleb.py` which can output arbitrary size as specified.
    """

    def __init__(self, kernel_size, stride):
        super(ConvTranspose2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        pad_height = self.kernel_size[0] - self.stride[0]
        pad_width = self.kernel_size[1] - self.stride[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return x[:, :, pad_top:in_height - pad_bottom, pad_left: in_width - pad_right]


class ConvAE(nn.Module):
    def __init__(self, channels, kernels):
        """
        :param channels: a list containing all channels including the input image channel (1 for gray, 3 for RGB)
        :param kernels:  a list containing all kernel sizes, it should satisfy: len(kernels) = len(channels) - 1.
        """
        super(ConvAE, self).__init__()
        assert isinstance(channels, list) and isinstance(kernels, list)
        self.encoder = nn.Sequential()
        for i in range(1, len(channels)):
            #  Each layer will divide the size of feature map by 2
            # 目前删去原代码下面这句
            self.encoder.add_module('pad%d' % i, Conv2dSamePad(kernels[i - 1], 2))
            self.encoder.add_module('conv%d' % i,
                                    nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=2))#4, padding=(1,1)
            self.encoder.add_module('relu%d' % i, nn.ReLU(True))

        self.decoder = nn.Sequential()
        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        for i in range(len(channels) - 1):
            # Each layer will double the size of feature map
            self.decoder.add_module('deconv%d' % (i + 1),
                                    nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2))#kernels[i]
            # 目前删去原代码下面这句
            self.decoder.add_module('padd%d' % i, ConvTranspose2dSamePad(kernels[i], 2))
            self.decoder.add_module('relud%d' % i, nn.ReLU(True))

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y


class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient, x)
        return y


# DSCNet(
#   (ae): ConvAE(
#     (encoder): Sequential(
#       (pad1): Conv2dSamePad()
#       (conv1): Conv2d(1, 15, kernel_size=(3, 3), stride=(2, 2))
#       (relu1): ReLU(inplace=True)
#     )
#     (decoder): Sequential(
#       (deconv1): ConvTranspose2d(15, 1, kernel_size=(3, 3), stride=(2, 2))
#       (padd0): ConvTranspose2dSamePad()
#       (relud0): ReLU(inplace=True)
#     )
#   )
#   (self_expression): SelfExpression()
# )

class GCN(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class DSCNet(nn.Module):
    def __init__(self, channels, kernels, num_sample):
        super(DSCNet, self).__init__()
        self.n = num_sample
        self.ae = ConvAE(channels, kernels)
        self.self_expression = SelfExpression(self.n)

        ##三层图卷积
        # self.conv1 = GCNConv(2940, 128)#(2940, 128)#(2940, 16)
        # self.conv2 = GCNConv(128, 256)#(128, 2940)#(16, 48)
        # self.conv3 = GCNConv(256, 2940)

        ##两层图卷积
        # self.conv1 = GCNConv(2940, 128)  # (2940, 128)#(2940, 16)
        # self.conv2 = GCNConv(128, 2940)  # (128, 2940)#(16, 48)


        # self.linear = nn.Linear(48, 2940)
        self.weight_keys = [['ae.encoder.conv1.weight'], ['ae.encoder.conv1.bias'],
                            ['ae.decoder.deconv1.weight'], ['ae.decoder.deconv1.bias'],
                            ['self_expression.Coefficient']
                            ]

    def forward(self, x, edge):  # shape=[n, c, w, h]
        z = self.ae.encoder(x)

        # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
        shape = z.shape
        # print("encoderout z.shape={}".format(z.shape))
        z = z.view(self.n, -1)  # shape=[n, d]
        # print("encoderout z.shape={}".format(z.shape))#[1000, 2940]
        #
        # data = Data(x=z, edge_index=edge.t().contiguous())
        # datax, edge_index = data.x, data.edge_index
        # print("datax={}".format(datax.shape))  # [1000, 2940]
        # print("edge_index={}".format(edge_index.shape))  # [2, 101000]
        # datax = self.conv1(datax, edge_index)
        # datax = self.conv2(datax, edge_index)
        # datax = self.conv3(datax, edge_index)
        #datax=F.relu(datax)
        # print("datax={}".format(datax.shape))# [1000, 64]
        # datax = F.relu(datax)

        z_recon = self.self_expression(z)  # datax  shape=[n, d] [1000, 2940]
        # [1000, 2940]
        # print("datax={}".format(datax.shape))
        # print("z_recon={}".format(z_recon.shape))
        # z_recon = self.linear(z_recon)
        z_recon_reshape = z_recon.view(shape)# [1000, 15, 14, 14]
        # print("z_recon_reshape={}".format(z_recon_reshape.shape))

        x_recon = self.ae.decoder(z_recon_reshape)  # shape=[n, c, w, h]
        # print("x_recon={}".format(x_recon.shape))
        return x_recon, z, z_recon

    def loss_fn(self, idx, iter, x, x_recon, z, z_recon, weight_coef, weight_selfExp):
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        # loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        loss_coef = torch.norm(self.self_expression.Coefficient,2)
        # loss_coef = torch.norm(self.self_expression.Coefficient,p='nuc')
        loss_selfExp = F.mse_loss(z_recon, z, reduction='sum')
        # loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp


        # loss_ae = F.mse_loss(x_recon, x, reduction='mean')
        # loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        # # loss_coef = torch.norm(self.self_expression.Coefficient,p='nuc')#torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        # loss_selfExp = F.mse_loss(z_recon, z, reduction='mean')
        loss = 0.1 * loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp
        # if iter == 4:
        #     image = self.self_expression.Coefficient
        #     transform1 = transforms.Compose([
        #         transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        #     ]
        #     )
        #     image = image.detach().numpy()

            # image = transform1(image)
            # image = transforms.ToPILImage()(image)
            # image.save('./image/{}-{}.jpg'.format(idx,iter))

        # p=2:1,1000000,1     p='nuc':1,100000,1
        # print("loss1:{} loss2:{} loss3:{}".format(0.1 * loss_ae,weight_coef * loss_coef,weight_selfExp * loss_selfExp))
        #
        # print("loss{}".format(loss))

        return loss

class Autoencoder1(nn.Module):
    def __init__(self):
        super(Autoencoder1, self).__init__()
        # 我们用前面CNN中使用的特征提取部分当作这里的编码器encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            # nn.Conv2d(32, 32, 7)
        )

        # 对于解码器decoder，我们需要使用nn.ConvTranspose2d
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(32, 32, 7),
            # nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Autoencoder2(nn.Module):
    def __init__(self):
        super(Autoencoder2, self).__init__()
        self.encoder = nn.Sequential(  # 如果输入的是28 * 28 的图片
            nn.Conv2d(1, 16, 3, 3, 1),  # 10
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 5

            nn.Conv2d(16, 8, 3, 2, 1),  # 3
            nn.ReLU(True),
            nn.MaxPool2d(2, 1)  # 2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, 2),  # 5
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 8, 5, 3, 1),  # 15
            nn.ReLU(True),

            nn.ConvTranspose2d(8, 1, 2, 2, 1),  # 28

            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Autoencoder3(nn.Module):
    # def __init__(self):
    #     super(Autoencoder3, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
    #     self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
    #     self.conv2_drop = nn.Dropout2d()
    #     # self.fc1 = nn.Linear(1024, 128)
    #     # self.fc2 = nn.Linear(128, 64)
    #     # self.fc3 = nn.Linear(64, 10)
    #
    # def forward(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    #     # x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
    #     # x = F.relu(self.fc1(x))
    #     # x = F.relu(self.fc2(x))
    #     # x = self.fc3(x)
    #     return x#F.log_softmax(x, dim=1)
    def __init__(self):
        super(Autoencoder3, self).__init__()
        # 我们用前面CNN中使用的特征提取部分当作这里的编码器encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            # nn.Conv2d(32, 32, 7)
        )

        # 对于解码器decoder，我们需要使用nn.ConvTranspose2d
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(32, 32, 7),
            # nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Autoencoder4(nn.Module):
    # def __init__(self):
    def __init__(self):
        super(Autoencoder4, self).__init__()
        # 我们用前面CNN中使用的特征提取部分当作这里的编码器encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
            # nn.ReLU(),
            # nn.Linear(256, 64)
            # nn.Conv2d(32, 32, 7)
        )

        # 对于解码器decoder，我们需要使用nn.ConvTranspose2d
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(32, 32, 7),
            # nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DSCNet1(nn.Module):
    def __init__(self, channels, kernels, num_sample):
        super(DSCNet, self).__init__()
        self.n = num_sample
        self.ae = Autoencoder4()
        self.self_expression = SelfExpression(self.n)

        # self.softmax = nn.Softmax(dim=1)
        self.fc1 = nn.Linear(256,64)#(1568, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        # self.conv1 = GCNConv(1568, 1568)
        self.weight_keys = [['ae.encoder.0.weight'], ['ae.encoder.0.bias'], ['ae.encoder.2.weight'], ['ae.encoder.2.bias'],
                            # ['ae.fc1.weight'], ['ae.fc1.bias'], ['ae.fc2.weight'], ['ae.fc2.bias'], ['ae.fc3.weight'], ['ae.fc3.bias'],
                            'self_expression.Coefficient'
                            ]

    def forward(self, x, edge):  # shape=[n, c, w, h]
        shape = x.shape
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.ae.encoder(x)
        z = x#x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        # x = F.relu(self.fc1(z))
        # x = self.fc3(x)
        # z_out = F.log_softmax(x, dim=1)

        # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
        # shape = z.shape
        # print("encoderout z.shape={}".format(z.shape))
        # z = z.view(self.n, -1)  # shape=[n, d] 1000*32*7*7

        # z_out = z
        # z_out = F.relu(self.fc1(z_out))
        # z_out = F.relu(self.fc2(z_out))
        # z_out = self.fc3(z_out)
        # print("encoderout z.shape={}".format(z.shape))#[1000, 2940]
        #
        # data = Data(x=z, edge_index=edge.t().contiguous())
        # datax, edge_index = data.x, data.edge_index
        #
        # print("datax={}".format(datax.shape))  # [1000, 2940]
        # print("edge_index={}".format(edge_index.shape))  # [2, 101000]
        # datax = self.conv1(datax, edge_index)
        # datax = self.conv2(datax, edge_index)
        # datax = self.conv3(datax, edge_index)
        #datax=F.relu(datax)
        # print("datax={}".format(datax.shape))# [1000, 64]
        # datax = F.relu(datax)
        #z:1000*1568

        z_recon = self.self_expression(z)  # datax  shape=[n, d] [1000, 2940]

        # [1000, 2940]
        # print("datax={}".format(datax.shape))
        # print("z_recon={}".format(z_recon.shape))
        # z_recon = self.linear(z_recon)


        # z_recon_reshape = z_recon.view(shape)# [1000, 15, 14, 14]


        # print("z_recon_reshape={}".format(z_recon_reshape.shape))

        x_recon = self.ae.decoder(z_recon)  # shape=[n, c, w, h]
        x_recon = x_recon.view(shape)

        # print("x_recon={}".format(x_recon.shape))
        return x_recon, z, z_recon#, z_out #

    def loss_fn(self, idx, iter, x, x_recon, z, z_recon, weight_coef, weight_selfExp):
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        # loss_coef = torch.norm(self.self_expression.Coefficient,p='nuc')#torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        loss_selfExp = F.mse_loss(z_recon, z, reduction='sum')
        # loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp


        # loss_ae = F.mse_loss(x_recon, x, reduction='mean')
        # loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        # # loss_coef = torch.norm(self.self_expression.Coefficient,p='nuc')#torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        # loss_selfExp = F.mse_loss(z_recon, z, reduction='mean')
        loss = 0.1 * loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp

        # print("loss1:{} loss2:{} loss3:{}".format(0.1* loss_ae,weight_coef * loss_coef,weight_selfExp * loss_selfExp))
        # print("loss{}".format(loss))
        return loss


# class DSCNet(nn.Module):
#     def __init__(self, channels, kernels, num_sample):
#         super(DSCNet, self).__init__()
#         self.n = num_sample
#         self.ae = ConvAE(channels, kernels)
#         self.self_expression = SelfExpression(self.n)
#         self.weight_keys = [['ae.encoder.conv1.weight'], ['ae.encoder.conv1.bias'],
#                                     ['ae.decoder.deconv1.weight'], ['ae.decoder.deconv1.bias'],
#                                     ['self_expression.Coefficient']
#                                     ]
#
#     def forward(self, x):  # shape=[n, c, w, h]
#         z = self.ae.encoder(x)
#
#         # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
#         shape = z.shape
#         z = z.view(self.n, -1)  # shape=[n, d]
#         z_recon = self.self_expression(z)  # shape=[n, d]
#         # print("z_rezon/shape")
#         # print(z_recon.shape)
#         z_recon_reshape = z_recon.view(shape)
#         # print("z_recon_re/shape")
#         # print(z_recon_reshape.shape)
#
#         x_recon = self.ae.decoder(z_recon_reshape)  # shape=[n, c, w, h]
#         # print("x_rezon/shape")
#         # print(x_recon.shape)
#         return x_recon, z, z_recon
#
#     def loss_fn(self, x, x_recon, z, z_recon, weight_coef, weight_selfExp ):
#         loss_ae = F.mse_loss(x_recon, x, reduction='sum')
#         loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
#         loss_selfExp = F.mse_loss(z_recon, z, reduction='sum')
#
#         # image = self.self_expression.Coefficient
#         # transform1 = transforms.Compose([
#         #     transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
#         # ]
#         # )
#         # image = image.detach().numpy()
#         # image = transform1(image)
#         # image = transforms.ToPILImage()(image)
#         # image.save('./image/{}.jpg'.format(epoch))
#
#         loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp
#         # print("loss1:{} loss2:{} loss3:{}".format(loss_ae, weight_coef * loss_coef, weight_selfExp * loss_selfExp))
#
#         return loss



# mycode
# class DSCNet(nn.Module):
#     def __init__(self, channels, kernels, num_sample):
#         super(DSCNet, self).__init__()
#         self.n = num_sample
#         # self.ae = ConvAE(channels, kernels)
#         self.encoder1 = nn.Sequential(
#             nn.Conv2d(1, 15, kernel_size=3, stride=2, padding=1, bias=True),  # 灰度图像，通道为1
#             nn.ReLU(),
#             # nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1, bias=True),
#             # nn.ReLU(),
#             # nn.Conv2d(20, 30, kernel_size=4, stride=2, padding=1, bias=True),
#             # nn.ReLU(),
#         )
#         self.decoder1 = nn.Sequential(
#             # nn.ConvTranspose2d(30, 20, kernel_size=4, stride=2, padding=1, bias=True),
#             # nn.ReLU(),
#             # nn.ConvTranspose2d(20, 10, kernel_size=3, stride=1, padding=1, bias=True),
#             # nn.ReLU(),
#             nn.ConvTranspose2d(15, 1, kernel_size=3, stride=2, padding=0, bias=True),
#             nn.ReLU(),
#         )
#         self.self_expression = SelfExpression(self.n)
#         self.weight_keys = ['encoder1.0.weight', 'encoder1.0.bias',
#                             'encoder1.2.weight', 'encoder1.2.bias',
#                             'encoder1.4.weight', 'encoder1.4.bias',
#                             'decoder1.0.weight', 'decoder1.0.bias',
#                             'decoder1.2.weight', 'decoder1.2.bias',
#                             'decoder1.4.weight', 'decoder1.4.bias']
#
#     def forward(self, x):  # 自表达层，得到聚类
#         z = self.encoder1(x)
#         # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
#         shape = z.shape
#         # print("z.shape={}".format(z.shape))
#         z = z.view(self.n, -1)  # shape=[n, d]
#         z_recon = self.self_expression(z)  # shape=[n, d]
#         z_recon_reshape = z_recon.view(shape)
#         x_recon = self.decoder1(z_recon_reshape)  # shape=[n, c, w, h]
#         x_recon = x_recon[:, :, 0:28, 0:28]
#         return x_recon, z, z_recon
#
#     def loss_fn(self, x, x_recon, z, z_recon, weight_coef, weight_selfExp):
#         # print("x.shape={}".format(x.shape))
#         # print("x_recon.shape={}".format(x_recon.shape))
#         loss_ae = F.mse_loss(x_recon, x, reduction='sum')
#
#         loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
#         loss_selfExp = F.mse_loss(z_recon, z, reduction='sum')
#         loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp
#
#         return loss
#
