import skimage
from skimage import io
# import numpy as np
import torchvision.transforms as transforms

import numpy as np
from skimage import segmentation
# import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self, inp_dim, mod_dim1, mod_dim2):
        super(MyNet, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),   ###3   64
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0), ###64   32
            nn.BatchNorm2d(mod_dim2),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),####32  64
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),####  64    32
            nn.BatchNorm2d(mod_dim2),
        )

    def forward(self, x):
        return self.seq(x)


def image_transforms():
    return transforms.Compose([  #torchvision.transforms是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起
            transforms.ToTensor()#归一化到(0,1)，简单直接除以255
        ])


def read_rgb1(path):  ###,choose=TRUE
    """Read rgb image as np array

    Returns:
    img: numpy array with shape (h, w, c) = (375 x 1242 x 3)
    """
    # img = io.imread(path)
    image = io.imread(path)
    # print("11111111111")
    # print(img.shape())

    # '''segmentation ML'''
    # image = img
    seg_map = segmentation.felzenszwalb(image, scale=32, sigma=0.5, min_size=64)
    # seg_map = segmentation.slic(image, n_segments=10000, compactness=100)
    seg_map = seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0]
               for u_label in np.unique(seg_map)]

    '''train init'''
    # device = torch.device('cpu')

    tensor = image.transpose((2, 0, 1))
    tensor = tensor.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, :, :, :]
    tensor = torch.from_numpy(tensor).to('cpu')

    model = MyNet(inp_dim=3, mod_dim1=64, mod_dim2=3).to('cpu')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-1, momentum=0.0)

    image_flatten = image.reshape((-1, 3))
    color_avg = np.random.randint(255, size=(256, 3))
    '''train loop'''
    model.train()
    for batch_idx in range(32):###############1
        '''forward'''
        optimizer.zero_grad()
        output = model(tensor)[0]
        output = output.permute(1, 2, 0).view(-1, 3)
        target = torch.argmax(output, 1)
        im_target = target.data.cpu().numpy()
        '''refine'''
        for inds in seg_lab:
            u_labels, hist = np.unique(im_target[inds], return_counts=True)
            im_target[inds] = u_labels[np.argmax(hist)]
        '''backward'''
        target = torch.from_numpy(im_target)
        target = target.to('cpu')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        '''show image'''
        un_label, lab_inverse = np.unique(im_target, return_inverse=True)
        if un_label.shape[0] < 256:  # update show
            img_flatten = image_flatten.copy()
            if len(color_avg) != un_label.shape[0]:
                color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=np.int) for label in un_label]
            for lab_id, color in enumerate(color_avg):
                img_flatten[lab_inverse == lab_id] = color
            show = img_flatten.reshape(image.shape)

        if len(un_label) < 8: ###6
            break

    img = show
    # # print(img.shape())
    return img.astype('float32')  ##采用小的存储方式

def read_rgb2(path):  ###,choose=TRUE
    """Read rgb image as np array

    Returns:
    img: numpy array with shape (h, w, c) = (375 x 1242 x 3)
    """
    # img = io.imread(path)
    image = io.imread(path)
    seg_map = segmentation.felzenszwalb(image, scale=32, sigma=0.5, min_size=64)
    # seg_map = segmentation.slic(image, n_segments=10000, compactness=100)
    seg_map = seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0]
               for u_label in np.unique(seg_map)]

    '''train init'''
    # device = torch.device('cpu')

    tensor = image.transpose((2, 0, 1))
    tensor = tensor.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, :, :, :]
    tensor = torch.from_numpy(tensor).to('cpu')

    model = MyNet(inp_dim=3, mod_dim1=64, mod_dim2=3).to('cpu')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-1, momentum=0.0)

    image_flatten = image.reshape((-1, 3))
    color_avg = np.random.randint(255, size=(256, 3))
    '''train loop'''
    model.train()
    for batch_idx in range(32):###############1
        '''forward'''
        optimizer.zero_grad()
        output = model(tensor)[0]
        output = output.permute(1, 2, 0).view(-1, 3)
        target = torch.argmax(output, 1)
        im_target = target.data.cpu().numpy()
        '''refine'''
        for inds in seg_lab:
            u_labels, hist = np.unique(im_target[inds], return_counts=True)
            im_target[inds] = u_labels[np.argmax(hist)]
        '''backward'''
        target = torch.from_numpy(im_target)
        target = target.to('cpu')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        '''show image'''
        un_label, lab_inverse = np.unique(im_target, return_inverse=True)
        if un_label.shape[0] < 256:  # update show
            img_flatten = image_flatten.copy()
            if len(color_avg) != un_label.shape[0]:
                color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=np.int) for label in un_label]
            for lab_id, color in enumerate(color_avg):
                img_flatten[lab_inverse == lab_id] = color
            show = img_flatten.reshape(image.shape)

        # print('Loss:', batch_idx, loss.item())
        if len(un_label) < 8: ###6
            break

    img = show
    # # print(img.shape())
    return img.astype('float32')  ##采用小的存储方式



def read_rgb(path):###,choose=TRUE
    """Read rgb image as np array

    Returns:
    img: numpy array with shape (h, w, c) = (375 x 1242 x 3)
    """
    img = io.imread(path)

    return img.astype('float32')  ##采用小的存储方式
    
def read_lidar(path):##读入深度数据 并且通过深度数据制作为  掩码标签真值
    """Read lidar image and generate mask#

    Returns:
    lidar: np array with shape (h, w, 1) = (375 x 1242 x 1)
    mask: np array with shape (h, w, 1)
    """
    lidar = io.imread(path) # with shape (h, w)
    lidar = lidar * 1.0 / 256.0######################33不是很理解为什么要除上256
    mask = np.where(lidar > 0.0, 1.0, 0.0) # with shape (h, w)  #有深度信息的地方标记为1 没有的地方不进行标记

    lidar = lidar[:, :, np.newaxis].astype('float32')  #更改格式
    mask = mask[:, :, np.newaxis].astype('float32')
    return lidar, mask    

def read_gt(path):
    """Read gt image.

    Returns:
    dense: np array with shape (h, w, 1) = (375 x 1242 x 1)
    """
    dense = io.imread(path)
    dense = dense * 1.0 / 256.0 ####同样为什么要除上一个256 不是很理解
    dense = dense[:, :, np.newaxis].astype('float32')

    return dense


def read_normal(path):
    """Read surface normal image:

    Returns:
    """
    
    normal = io.imread(path)
    normal_gray = skimage.color.rgb2gray(normal) #Gray = R*0.299 + G*0.587 + B*0.114  RGB图像转换为灰度图像
    normal = normal.astype('float32')
    normal = normal * 1 / 127.5 - np.ones_like(normal) * 1.0  ##减去一个全1 的数组      ###为什么除上127.5    ??????????

    mask = np.zeros_like(normal).astype('float32')

    mask[:, :, 0] = np.where(normal_gray > 0, 1.0, 0.0)
    mask[:, :, 1] = np.where(normal_gray > 0, 1.0, 0.0)
    mask[:, :, 2] = np.where(normal_gray > 0, 1.0, 0.0)

    return normal, mask