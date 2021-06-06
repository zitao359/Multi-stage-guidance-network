import os
from torch.utils.data import Dataset, DataLoader
from skimage import io
import numpy as np
import random
from dataloader.reader import *
from peizhi import KITTI_DATASET_PATH

INTRINSICS = {
    "2011_09_26": (721.5377, 609.5593, 172.8540),
    "2011_09_28": (707.0493, 604.0814, 180.5066),
    "2011_09_29": (718.3351, 600.3891, 181.5122),
    "2011_09_30": (707.0912, 601.8873, 183.1104),
    "2011_10_03": (718.8560, 607.1928, 185.2157),
}





def get_loader(split='train', batch_size=6, shuffle=True, num_workers=0, num_data=None, crop=True):  ##.1

    rgb_image_paths, lidar_image_paths, gt_image_paths, normal_image_paths = get_paths(split)



    dataset = depth_dataset(rgb_image_paths, lidar_image_paths, gt_image_paths, normal_image_paths, num_data, crop=crop)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader

class depth_dataset(Dataset):


    def __init__(self, rgb_image_paths, lidar_image_paths, gt_image_paths, normal_image_paths=None, num_data=None, h=256, w=512, crop=True):

        self.rgb_image_paths = np.array(rgb_image_paths)
        self.lidar_image_paths = np.array(lidar_image_paths)
        self.gt_image_paths = np.array(gt_image_paths)
        
        if normal_image_paths:
            self.normal_image_paths = np.array(normal_image_paths)

        if num_data:
            np.random.seed(0)
            indices = np.random.choice(len(lidar_image_paths), size=num_data, )
            self.rgb_image_paths = self.rgb_image_paths[indices]
            self.lidar_image_paths = self.lidar_image_paths[indices]
            self.gt_image_paths = self.gt_image_paths[indices]
            if normal_image_paths:
                self.normal_image_paths = self.normal_image_paths[indices]


        self.crop_or_not = crop
        self.h = h
        self.w = w
    
        self.transforms = image_transforms()

    def __len__(self):
        return len(self.rgb_image_paths)

    def __getitem__(self, idx):

        date = self.rgb_image_paths[idx].split('/')[6][:10] #

        intrinsics = INTRINSICS[date]
        params = np.ones((self.h, self.w, 3)).astype('float32')


        params[:, :, 0] = params[:, :, 0] * intrinsics[0]
        params[:, :, 1] = params[:, :, 1] * intrinsics[1]
        params[:, :, 2] = params[:, :, 2] * intrinsics[2]


        rgb = read_rgb(self.rgb_image_paths[idx])#####



        lidar, mask = read_lidar(self.lidar_image_paths[idx])
        gt = read_gt(self.gt_image_paths[idx])

        surface_normal, mask_normal = read_normal(self.normal_image_paths[idx])


        height, width, channel = rgb.shape#######

        x_lefttop = random.randint(0, height - self.h)
        y_lefttop = random.randint(0, width - self.w)



        if self.crop_or_not:
            rgb = self._crop(rgb, x_lefttop, y_lefttop, self.h, self.w)#######################################3333333333333333333333333

            lidar = self._crop(lidar, x_lefttop, y_lefttop, self.h, self.w)
            mask = self._crop(mask, x_lefttop, y_lefttop, self.h, self.w)
            gt = self._crop(gt, x_lefttop, y_lefttop, self.h, self.w)
            surface_normal = self._crop(surface_normal, x_lefttop, y_lefttop, self.h, self.w)
            mask_normal = self._crop(mask_normal, x_lefttop, y_lefttop, self.h, self.w)




        return self.transforms(rgb), self.transforms(lidar), self.transforms(mask), self.transforms(gt), self.transforms(params),\
            self.transforms(surface_normal), self.transforms(mask_normal)#, self.transforms(yuyi) ############################################################3333333333333333
        


    def _crop(self, img, x, y, h, w):

        return img[x:x+h, y:y+w, :]


rgb_folder = os.path.join(KITTI_DATASET_PATH, 'data_rgb')


lidar_folder = os.path.join(KITTI_DATASET_PATH, 'data_depth_velodyne')

gt_folder = os.path.join(KITTI_DATASET_PATH, 'data_depth_annotated')

normal_folder = os.path.join(KITTI_DATASET_PATH, 'data_depth_normals')

rgb2_subfolder = os.path.join('image_02', 'data')
rgb3_subfolder = os.path.join('image_03', 'data')

lidar2_subfolder = os.path.join('proj_depth', 'velodyne_raw', 'image_02')
lidar3_subfolder = os.path.join('proj_depth', 'velodyne_raw', 'image_03')

gt2_subfolder = os.path.join('proj_depth', 'groundtruth', 'image_02') # the same as normal
gt3_subfolder = os.path.join('proj_depth', 'groundtruth', 'image_03')

def get_paths(split='train'):

    assert split in {'train', 'val'}


    date_folder_list = os.listdir(os.path.join(rgb_folder, split))
    date_folder_list.sort()

    rgb_image_paths = []
    lidar_image_paths = []
    gt_image_paths = []
    normal_image_paths = []

    for date_folder in date_folder_list:

        rgb2_base = os.path.join(rgb_folder, split, date_folder, rgb2_subfolder)
        rgb3_base = os.path.join(rgb_folder, split, date_folder, rgb3_subfolder)

        lidar2_base = os.path.join(lidar_folder, split, date_folder, lidar2_subfolder)
        lidar3_base = os.path.join(lidar_folder, split, date_folder, lidar3_subfolder)   
        
        gt2_base = os.path.join(gt_folder, split, date_folder, gt2_subfolder)
        gt3_base = os.path.join(gt_folder, split, date_folder, gt3_subfolder)         
        
        normal2_base = os.path.join(normal_folder, split, date_folder, gt2_subfolder)
        normal3_base = os.path.join(normal_folder, split, date_folder, gt3_subfolder) 


        img_filenames = os.listdir(os.path.join(lidar_folder, split, date_folder, lidar2_subfolder))
        img_filenames.sort()


        rgb_image_paths.extend([os.path.join(rgb2_base, fn) for fn in img_filenames])
        rgb_image_paths.extend([os.path.join(rgb3_base, fn) for fn in img_filenames])########################把image2 image3 都加入进去了

        lidar_image_paths.extend([os.path.join(lidar2_base, fn) for fn in img_filenames])
        lidar_image_paths.extend([os.path.join(lidar3_base, fn) for fn in img_filenames])

        gt_image_paths.extend([os.path.join(gt2_base, fn) for fn in img_filenames])
        gt_image_paths.extend([os.path.join(gt3_base, fn) for fn in img_filenames])

        normal_image_paths.extend([os.path.join(normal2_base, fn) for fn in img_filenames])
        normal_image_paths.extend([os.path.join(normal3_base, fn) for fn in img_filenames])

    assert len(rgb_image_paths) == len(lidar_image_paths) == len(gt_image_paths) == len(normal_image_paths)

    print('The number of {} data: {}'.format(split, len(rgb_image_paths)))

    return rgb_image_paths, lidar_image_paths, gt_image_paths, normal_image_paths


if __name__ == '__main__':
    get_paths()
    loader = get_loader('train')
    for rgb, lidar, mask, gt, params in loader:
        pass