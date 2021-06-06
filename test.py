import cv2
from tqdm import tqdm
import argparse
import os
import torch
import torch.nn as nn
# import cv2
from dataloader.reader import *
from model.model import deepMsG
# from model.pre import deepLidar
import torch.nn.functional as F
from PIL import Image
from training.utils import *
from peizhi import PREDICTED_RESULT_DIR, KITTI_DATASET_PATH
import time
from PIL import Image

parser = argparse.ArgumentParser(description='MsG Depth Completion')
parser.add_argument('-m', '--model_path', default='/home/aszitao/pre7/model__A_e10.tar',help='loaded model path')##A2  /home/aszitao/test2/model__A_e19.tar  /home/aszitao/test(good)/model__A_e2.tar

parser.add_argument('-n', '--num_testing_image', type=int, default=1000,
                    help='The number of testing image to be runned')
parser.add_argument('-cpu', '--using_cpu', action='store_true', help='use cpu')
parser.add_argument('-s', '--save_fig', action='store_true',default='/home/aszitao/test/', help='save predicted result or not')  #0.2557

args = parser.parse_args()


DEVICE = 'cuda' if torch.cuda.is_available() and not args.using_cpu else 'cpu'



def rmse(pred, gt):
    dif = gt[np.where(gt>0)] - pred[np.where(gt>0)]
    error = np.sqrt(np.mean(dif**2))
    return error

def mae(img,gt):
    dif = gt[np.where(gt>0)] - img[np.where(gt>0)]
    error = np.mean(np.fabs(dif))
    return error
def irmse(img,gt):
    dif = 1.0/gt[np.where(gt>0)] - 1.0/img[np.where(gt>0)]
    error = np.sqrt(np.mean(dif**2))
    return error
def imae(img,gt):
    dif = 1.0/gt[np.where(gt>0)] - 1.0/img[np.where(gt>0)]
    error = np.mean(np.fabs(dif))
    return error

total_time = []
def test(model, rgb, lidar, mask):


    model.eval()
    model = model.to(DEVICE)
    rgb = rgb.to(DEVICE)
    lidar = lidar.to(DEVICE)
    mask = mask.to(DEVICE)

    with torch.no_grad():
        torch.cuda.synchronize()
        a = time.perf_counter()


        color_path_dense, normal_path_dense, color_attn, normal_attn, surface_normal = model(rgb=rgb, lidar=lidar, mask=mask, stage='A')
        torch.cuda.synchronize()
        b = time.perf_counter()
        total_time.append(b - a)
        predicted_dense, pred_color_path_dense, pred_normal_path_dense = \
                            get_predicted_depth(color_path_dense, normal_path_dense, color_attn, normal_attn)

        return torch.squeeze(pred_color_path_dense).cpu()


def get_testing_img_paths():
    gt_folder = os.path.join(KITTI_DATASET_PATH, 'depth_selection', 'val_selection_cropped', 'groundtruth_depth')
    rgb_folder = os.path.join(KITTI_DATASET_PATH, 'depth_selection', 'val_selection_cropped', 'image')
    lidar_folder = os.path.join(KITTI_DATASET_PATH, 'depth_selection', 'val_selection_cropped', 'velodyne_raw')

    gt_filenames = sorted([img for img in os.listdir(gt_folder)])
    rgb_filenames = sorted([img for img in os.listdir(rgb_folder)])
    lidar_filenames = sorted([img for img in os.listdir(lidar_folder)])

    gt_paths = [os.path.join(gt_folder, fn) for fn in gt_filenames]
    rgb_paths = [os.path.join(rgb_folder, fn) for fn in rgb_filenames]
    lidar_paths = [os.path.join(lidar_folder, fn) for fn in lidar_filenames]

    return rgb_paths, lidar_paths, gt_paths

def main():

    rgb_paths, lidar_paths, gt_paths = get_testing_img_paths()

    num_testing_image = len(rgb_paths) if args.num_testing_image == -1 else args.num_testing_image


    model = deepMSG()
    dic = torch.load(args.model_path, map_location=DEVICE)
    state_dict = dic["state_dict"]
    model.load_state_dict(state_dict)
    print('Loss of loaded model: {:.4f}'.format(dic['val_loss']))
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


    transformer = image_transforms()
    pbar = tqdm(range(num_testing_image))
    running_error = 0
    running_error1 = 0
    running_error2= 0
    running_error3 = 0

    for idx in pbar:

        rgb = read_rgb(rgb_paths[idx]) # h x w x 3  #

        lidar, mask = read_lidar(lidar_paths[idx]) #
        gt = read_gt(gt_paths[idx]) # h x w x 1


        rgb = transformer(rgb).unsqueeze(0)  #

        lidar, mask = transformer(lidar).unsqueeze(0), transformer(mask).unsqueeze(0)

        fn = os.path.basename(rgb_paths[idx])
        saved_path = os.path.join(PREDICTED_RESULT_DIR, fn)


        pred = test(model, rgb, lidar, mask).numpy()

        pred = np.where(pred <= 0.0, 0.9, pred)


        gt = gt.reshape(gt.shape[0], gt.shape[1])
        mae_loss = mae(pred, gt)*1000

        irmse_loss = irmse(pred, gt) * 1000
        imae_loss = imae(pred, gt) * 1000
        rmse_loss = rmse(pred, gt) * 1000




        running_error += rmse_loss
        mean_error = running_error / (idx + 1)

        running_error1 += mae_loss
        mae_error = running_error1 / (idx + 1)


        running_error2 += irmse_loss
        irmse_error = running_error2 / (idx + 1)

        running_error3 += imae_loss
        imae_error = running_error3 / (idx + 1)

        pbar.set_description('rmse error: {:.4f},Mae error: {:.4f},irmse error: {:.4f},imae error: {:.4f}'.format(mean_error,mae_error,irmse_error,imae_error))

        if args.save_fig:


            pred_show = pred* 256.0
            pred_show = pred_show.astype('uint16')
            res_buffer = pred_show.tobytes()
            img = Image.new("I", pred_show.T.shape)
            img.frombytes(res_buffer, 'raw', "I;16")
            img.save(saved_path)

    print('average_time: ', sum(total_time[100:]) / (len(total_time[100:])))
if __name__ == '__main__':
    main()