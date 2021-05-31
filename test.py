import cv2
from tqdm import tqdm
import argparse
import os
import torch
import torch.nn as nn
# import cv2
from dataloader.image_reader import *
# from model.DeepLidar import deepLidar
from model.copy4 import deepLidar
# from model.pre import deepLidar
import torch.nn.functional as F
from PIL import Image
from training.utils import *
from env import PREDICTED_RESULT_DIR, KITTI_DATASET_PATH
import time
from PIL import Image
from dataloader.testerf import test1
parser = argparse.ArgumentParser(description='Depth Completion')
parser.add_argument('-m', '--model_path', default='/home/aszitao/pre7/model__A_e10.tar',help='loaded model path')##A2  /home/aszitao/test2/model__A_e19.tar  /home/aszitao/test(good)/model__A_e2.tar
# /home/aszitao/test(todo)/model__A_e15.tar   /home/aszitao/pre7/model__A_e15.tar  /home/aszitao/pre8/model__A_e12
parser.add_argument('-n', '--num_testing_image', type=int, default=1000,
                    help='The number of testing image to be runned')
parser.add_argument('-cpu', '--using_cpu', action='store_true', help='use cpu')
parser.add_argument('-s', '--save_fig', action='store_true',default='/home/aszitao/test/', help='save predicted result or not')  #0.2557
# Number of model parameters: 53475870   1018  0.2557
#
# 1071     31610703   0.2688

# 0。2448    982.8408:  32104792

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

    # print(rgb_yuyi.shape)
    # print(rgb.shape)

    # yuyi=test1(rgb_yuyi)
    model.eval()

    model = model.to(DEVICE)
    rgb = rgb.to(DEVICE)
    lidar = lidar.to(DEVICE)
    mask = mask.to(DEVICE)
    # yuyi=test(rgb)
    # yuyi =yuyi.to(DEVICE)

    # yuyi= yuyi.to(DEVICE)

    with torch.no_grad():
        torch.cuda.synchronize()
        a = time.perf_counter()


        color_path_dense, normal_path_dense, color_attn, normal_attn, surface_normal = model(rgb=rgb, lidar=lidar, mask=mask, stage='A')
        torch.cuda.synchronize()
        b = time.perf_counter()
        total_time.append(b - a)
        predicted_dense, pred_color_path_dense, pred_normal_path_dense = \
                            get_predicted_depth(color_path_dense, normal_path_dense, color_attn, normal_attn)

        # surface_normal, initguidance
        # print(predicted_dense.size())yidon
        # print(normal_path_dense.size())
        return torch.squeeze(pred_color_path_dense).cpu()

# /home/aszitao/data/depth_selection/test_depth_completion_anonymous
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
    # get image paths
    rgb_paths, lidar_paths, gt_paths = get_testing_img_paths()

    # set the number of testing images
    num_testing_image = len(rgb_paths) if args.num_testing_image == -1 else args.num_testing_image

    # load model
    model = deepLidar()
    dic = torch.load(args.model_path, map_location=DEVICE)
    state_dict = dic["state_dict"]
    model.load_state_dict(state_dict)
    print('Loss of loaded model: {:.4f}'.format(dic['val_loss']))
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


    transformer = image_transforms()   #对数据进行归一化
    pbar = tqdm(range(num_testing_image))
    running_error = 0
    running_error1 = 0
    running_error2= 0
    running_error3 = 0

    for idx in pbar:
        # read image
        # print(rgb_paths[idx])
        # print(lidar_paths[idx])
        #exit()
        rgb = read_rgb(rgb_paths[idx]) # h x w x 3  #读取
        # rgb_yuyi=rgb
        # rgb_yuyi = rgb_yuyi.transpose(1, 2, 0)
        # print(rgb_yuyi.shape,"54634532")
        # rgb1 = read_rgb1(rgb_paths[idx])
        # print(rgb.shape)
        lidar, mask = read_lidar(lidar_paths[idx]) # h x w x 1 #读入激光雷达深度数据
        gt = read_gt(gt_paths[idx]) # h x w x 1

        # 将numpy转换为张量并添加批处理维数
        rgb = transformer(rgb).unsqueeze(0)  ###归一化   并且在第0维度上增加一维度
        # rgb_yuyi=transformer(rgb_yuyi)
        # print(rgb_yuyi.shape, "77777777777777777777777777")
        lidar, mask = transformer(lidar).unsqueeze(0), transformer(mask).unsqueeze(0)
        # yuyi = transformer(yuyi).unsqueeze(0)

        # saved file path
        fn = os.path.basename(rgb_paths[idx])
        saved_path = os.path.join(PREDICTED_RESULT_DIR, fn)

        # run model
        pred = test(model, rgb, lidar, mask).numpy()

        # plt.imshow(normal[1,:,:])
        # plt.imshow(init[0,:,:])
        # plt.imshow(mid[1,:,:])
        # plt.imshow(last,camp='gray')
        # cv2.imshow("img",last)
        # cv2.waitKey(0)
        # plt.imshow(result[1,:,:])

        # plt.show()
        # print(pred.size())
        # pred = result
        # pred=last
        # maxi=last.max()
        # print(maxi)
        # last=last*-255/maxi
        # last=last.transpose(1,2,0)#.astype(np.unit8)
        # last=Image.fromarray(last)
        # plt.imshow(last)
        # plt.show()
        # pred=init

        pred = np.where(pred <= 0.0, 0.9, pred)
        # print(pred.shape,"22")

        gt = gt.reshape(gt.shape[0], gt.shape[1])
        mae_loss = mae(pred, gt)*1000

        irmse_loss = irmse(pred, gt) * 1000
        imae_loss = imae(pred, gt) * 1000
        rmse_loss = rmse(pred, gt) * 1000

        # print("测试误差显示''''''''''''''''''''''''''''''''''''''''")
        # print('rmse')
        # print(rmse(pred, gt)* 1000)
        # print('mae')
        # print(mae(pred, gt)* 1000)
        # print('irmse')
        # print(irmse(pred, gt)* 1000)
        # print('imae')
        # print(imae(pred, gt)* 1000)




        running_error += rmse_loss
        mean_error = running_error / (idx + 1)

        running_error1 += mae_loss
        mae_error = running_error1 / (idx + 1)


        running_error2 += irmse_loss
        irmse_error = running_error2 / (idx + 1)

        running_error3 += imae_loss
        imae_error = running_error3 / (idx + 1)

        pbar.set_description('rmse error: {:.4f},Mae error: {:.4f},irmse error: {:.4f},imae error: {:.4f}'.format(mean_error,mae_error,irmse_error,imae_error))
        # pbar.set_description('Mae error: {:.4f}'.format(mae_error))
        # pbar.set_description('irmse error: {:.4f}'.format(irmse_error))
        # pbar.set_description('imae error: {:.4f}'.format(imae_error))

        if args.save_fig:
            # save image

            pred_show = pred* 256.0
            pred_show = pred_show.astype('uint16')
            res_buffer = pred_show.tobytes()
            img = Image.new("I", pred_show.T.shape)
            img.frombytes(res_buffer, 'raw', "I;16")
            img.save(saved_path)
            # print(rmse())
    print('average_time: ', sum(total_time[100:]) / (len(total_time[100:])))
if __name__ == '__main__':
    main()