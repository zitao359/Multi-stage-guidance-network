import argparse
import os

import torch
import torch.optim as optim
from tqdm import tqdm
from dataloader.dataloader import get_loader
from model.model import deepMsG
from training.train import EarlyStop, train_val
from training.utils import get_depth_and_normal
from peizhi import SAVED_MODEL_PATH


parser = argparse.ArgumentParser(description='MsG Depth Completion')
parser.add_argument('-b', '--batch_size', type=int, default=6, help='batch size')
parser.add_argument('-e', '--epoch', type=int, default=10, help='number of epochs')
parser.add_argument('-m', '--saved_model_name', type=str, default='model_', help='saved_model_name')
parser.add_argument('-cpu', '--using_cpu', action='store_true', help='use cpu')#/home/aszitao/test(todo)/model__N_e18.tar  /home/aszitao/test(todo)/model__N_e18.tar
parser.add_argument('-l', '--load_model',  default='/home/aszitao/pre7/model__N_e10.tar',help='load model') #/home/aszitao/test/model__A_e2.tar/home/aszitao/all_not/model__D_e6   8   /home/aszitao/all_not/model__N_e15.tar /home/aszitao/test2/model__N_e20.tar
parser.add_argument('-n', '--num_data', type=int, default=85896, help='the number of data used to train') #85898   /home/aszitao/test(good)/model__N_e17.tar  3000
args = parser.parse_args()#19692  /home/aszitao/pre7/model__N_e21.tar


DEVICE = 'cuda' if torch.cuda.is_available() and not args.using_cpu else 'cpu'


def main_train(model, stage):
    #
    tensorboard_path = 'runs/{}_{}'.format(stage, args.saved_model_name)
    tb_writer = TensorboardWriter(tensorboard_path)

    testing_rgb, testing_lidar, testing_mask = tb_writer.get_testing_img()########################
    testing_rgb, testing_lidar, testing_mask = testing_rgb.to(DEVICE), testing_lidar.to(DEVICE), testing_mask.to(DEVICE)

    early_stop = EarlyStop(patience=10, mode='min')

    loader = {'train': get_loader('train', num_data=args.num_data), \
              'val': get_loader('val', shuffle=False, num_data=6852)}#

    for epoch in range(args.epoch):
        saved_model_path = os.path.join(SAVED_MODEL_PATH, "{}_{}_e{}".format(args.saved_model_name, stage, epoch+1))
        train_losses, val_losses = train_val(model, loader, epoch, DEVICE, stage)

        predicted_dense, pred_surface_normal = get_depth_and_normal(model, testing_rgb, testing_lidar, testing_mask)
        tb_writer.tensorboard_write(epoch, train_losses, val_losses, predicted_dense, pred_surface_normal)

        if early_stop.stop(val_losses[0], model, epoch+1, saved_model_path):
            break

    tb_writer.close()



def main():
    model = deepMsG().to(DEVICE)
    if args.load_model:
        dic = torch.load(args.load_model)
        state_dict = dic["state_dict"]
        model.load_state_dict(state_dict)
        print('Loss of loaded model: {:.4f}'.format(dic['val_loss']))

    main_train(model, 'A')
#

if __name__ == '__main__':
    main()
