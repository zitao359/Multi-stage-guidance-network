from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from training.utils import get_loss



def get_optimizer(model, stage):

    assert stage in {'D', 'N', 'A'}

    if stage == 'N':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.normal.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(model.normal.parameters(), lr=0.001, betas=(0.9, 0.999))#0.000125
        loss_weights = [0, 0, 0, 1]

    elif stage == 'D':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.color_path.parameters():
            param.requires_grad = True
        for param in model.normal_path.parameters():
            param.requires_grad = True

        optimizer = optim.Adam([{'params':model.color_path.parameters()},
                                {'params':model.normal_path.parameters()}], lr=0.000125, betas=(0.9, 0.999))
        loss_weights = [0.3, 0.3, 0.0, 0.1]

    else:
        for param in model.parameters():
            param.requires_grad = True
        for param in model.normal.parameters():
            param.requires_grad = False



        optimizer = optim.Adam([{'params':model.color_path.parameters()},
                                {'params':model.normal_path.parameters()},
                                {'params':model.mask_block_C.parameters()},
                                {'params':model.mask_block_N.parameters()}], lr=0.001, betas=(0.9, 0.999))

        loss_weights = [0.3, 0.3, 0.5, 0.1]

    return model, optimizer, loss_weights



def train_val(model, loader, epoch, device, stage):
    """Train and validate the model

    Returns: training and validation loss
    """

    model, optimizer, loss_weights = get_optimizer(model, stage)
    train_loss, val_loss = [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]####################################train00000000000000000000000000

    for phase in ['train', 'val']:
        total_loss, total_loss_d, total_loss_c, total_loss_n, total_loss_normal,smoothloss_total,loss5_total = 0, 0, 0, 0, 0,0,0
        total_pic = 0 # used to calculate average loss
        data_loader = loader[phase]
        pbar = tqdm(iter(data_loader))

        if phase == 'train':
            model.train()
        else:
            model.eval()

        for num_batch, (rgb, lidar, mask, gt_depth, params, gt_surface_normal, gt_normal_mask) in enumerate(pbar):  #####rgb1
            """
 
            """
            rgb, lidar, mask = rgb.to(device), lidar.to(device), mask.to(device)
            gt_depth, params = gt_depth.to(device), params.to(device)
            gt_surface_normal, gt_normal_mask = gt_surface_normal.to(device), gt_normal_mask.to(device)
            # yuyi = yuyi.to(device)#####################################

            if phase == 'train':
                color_path_dense, normal_path_dense, color_attn, normal_attn, pred_surface_normal = model(rgb, lidar, mask, stage) ###rgb1
            else:
                with torch.no_grad():
                    color_path_dense, normal_path_dense, color_attn, normal_attn, pred_surface_normal = model(rgb, lidar, mask, stage)####rgb1

            loss_c, loss_n, loss_d, loss_normal,loss5= get_loss(color_path_dense, normal_path_dense, color_attn,\
                                                            normal_attn, pred_surface_normal, stage,\
                                                            gt_depth, params, gt_surface_normal, gt_normal_mask)



            # print(loss_weights[3],loss5) #loss_weights[3] * 0.0433 * 0.5   0.0462*0.5  0.047*0.5   0.0422*0.5  0.5*0.0409

            loss = loss_weights[0] * loss_c + loss_weights[1] * loss_n + loss_weights[2] * loss_d +loss_weights[3] * 0.0434* 0.5 +loss_weights[3] * loss5##loss_weights[4]*smoothloss######################################################
##loss_weights[3] * loss_normal loss_weights[3] *0.0448*0.8+loss_weights[3] *loss5+loss_weights[3] *0.0448*0.8  loss_weights[3] * 0.0576*0.5  loss_weights[3] * 0.0555*0.5

            # loss_weights[3] * 0.0433 * 0.5   loss_weights[3] * 0.0423*0.5
            total_loss += loss.item()
            total_loss_d += loss_d.item()
            total_loss_c += loss_c.item()
            total_loss_n += loss_n.item()
            total_loss_normal += loss_normal.item()
            loss5_total += loss5.item()
            # smoothloss_total+=smoothloss.item()##################################################

            total_pic += rgb.size(0)

            if phase == 'train':
                train_loss[0] = total_loss/total_pic
                train_loss[1] = total_loss_d/total_pic
                train_loss[2] = total_loss_c/total_pic
                train_loss[3] = total_loss_n/total_pic
                train_loss[4] = total_loss_normal/total_pic
                train_loss[5] = loss5_total / total_pic


                # train_loss[5] = smoothloss_total / total_pic

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            else:
                val_loss[0] = total_loss/total_pic
                val_loss[1] = total_loss_d/total_pic
                val_loss[2] = total_loss_c/total_pic
                val_loss[3] = total_loss_n/total_pic
                val_loss[4] = total_loss_normal/total_pic
                val_loss[5] = loss5_total / total_pic

            pbar.set_description(
                '[{}] Epoch: {}; loss: {:.4f}; loss_d: {:.4f}, loss_c: {:.4f}, loss_n: {:.4f}, loss_normal: {:.4f}, loss_5l: {:.4f}'. \
                format(phase.upper(), epoch + 1, total_loss / total_pic, total_loss_d / total_pic,
                       total_loss_c / total_pic, total_loss_n / total_pic, total_loss_normal / total_pic, loss5_total / total_pic
                      ))

    return train_loss, val_loss

class EarlyStop():
    """Early stop training if validation loss didn't improve for a long time"""
    def __init__(self, patience, mode = 'min'):
        self.patience = patience
        self.mode = mode

        self.best = float('inf') if mode == 'min' else 0
        self.cur_patience = 0

    def stop(self, loss, model, epoch, saved_model_path):
        update_best = loss < self.best if self.mode == 'min' else loss > self.best

        if update_best:
            self.best = loss
            self.cur_patience = 0

            torch.save({'val_loss': loss, \
                        'state_dict': model.state_dict(), \
                        'epoch': epoch}, saved_model_path+'.tar')
            print('SAVE MODEL to {}'.format(saved_model_path))
        else:
            self.cur_patience += 1
            if self.patience == self.cur_patience:
                return True
        
        return False

