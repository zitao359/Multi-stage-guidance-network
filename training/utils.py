import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


def get_predicted_depth(color_path_dense, normal_path_dense, color_attn, normal_attn):

    pred_color_path_dense = color_path_dense[:, 0, :, :] # b x 128 x 256
    pred_normal_path_dense = normal_path_dense[:, 0, :, :]



    color_attn = torch.squeeze(color_attn) # b x 128 x 256
    normal_attn = torch.squeeze(normal_attn) # b x 128 x 256


    pred_attn = torch.zeros_like(color_path_dense) # b x 2 x 128 x 256
    pred_attn[:, 0, :, :] = color_attn
    pred_attn[:, 1, :, :] = normal_attn
    pred_attn = F.softmax(pred_attn, dim=1) # b x 2 x 128 x 256

    color_attn, normal_attn = pred_attn[:, 0, :, :], pred_attn[:, 1, :, :]


    predicted_dense = pred_color_path_dense * color_attn + pred_normal_path_dense * normal_attn # b x 128 x 256

    predicted_dense = predicted_dense.unsqueeze(1)
    pred_color_path_dense = pred_color_path_dense.unsqueeze(1) 
    pred_normal_path_dense = pred_normal_path_dense.unsqueeze(1)

    return predicted_dense, pred_color_path_dense, pred_normal_path_dense

def get_depth_and_normal(model, rgb, lidar, mask):
    """Given model and input of model, get dense depth and surface normal

    Returns:
    predicted_dense: b x c x h x w
    pred_surface_normal: b x c x h x w
    """
    model.eval()
    with torch.no_grad():
        color_path_dense, normal_path_dense, color_attn, normal_attn, pred_surface_normal = model(rgb=rgb,lidar= lidar, mask=mask,stage='A')
        predicted_dense, _, _ = get_predicted_depth(color_path_dense, normal_path_dense, color_attn, normal_attn)
    return predicted_dense, pred_surface_normal




def normal_to_0_1(img):
    """Normalize image to [0, 1], used for tensorboard visualization."""
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img))


def normal_loss(pred_normal, gt_normal, gt_normal_mask):
    """Calculate loss of surface normal (in the stage N)

    Params:
    pred: b x 3 x 128 x 256
    normal_gt: b x 3 x 128 x 256
    normal_mask: b x 3 x 128 x 256
    """

    valid_mask = (gt_normal_mask > 0.0).detach()

    #pred_n = pred.permute(0,2,3,1)
    pred_normal = pred_normal[valid_mask]
    gt_normal = gt_normal[valid_mask]

    pred_normal = pred_normal.contiguous().view(-1,3)
    pred_normal = F.normalize(pred_normal)
    gt_normal = gt_normal.contiguous().view(-1, 3)

    loss_function = nn.CosineEmbeddingLoss()
    loss = loss_function(pred_normal, gt_normal, torch.Tensor(pred_normal.size(0)).to(pred_normal.device).fill_(1.0))
    return loss


def mse_loss(input,target):
    return torch.sum((input - target)**2) / input.data.nelement()


def get_depth_loss(dense, c_dense, n_dense, gt):
    """
    dense: b x 1 x 128 x 256
    c_dense: b x 1 x 128 x 256
    n_dense: b x 1 x 128 x 256
    gt: b x 1 x 128 x 256
    params: b x 3 x 128 x 256
    normals: b x 128 x 256 x 3
    """




    valid_mask = (gt > 0.0).detach() # b x 1 x 128 x 256

    gt = gt[valid_mask]
    dense, c_dense, n_dense = dense[valid_mask], c_dense[valid_mask], n_dense[valid_mask]


    criterion = nn.MSELoss()

    loss_d = torch.sqrt(criterion(dense, gt))#####################################
    loss_c = torch.sqrt(criterion(c_dense, gt))
    loss_n = torch.sqrt(criterion(n_dense, gt))

    # loss_d = mse_loss(dense, gt)
    # loss_c = mse_loss(c_dense, gt)
    # loss_n = criterion(n_dense, gt)






    
    return loss_d, loss_c, loss_n


k1 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
k2 = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
x=256
y=512
p=6
MatI=np.zeros((p,x,y), dtype=np.float32)  ###args.batch_size1
for i in range(MatI.shape[1]):
    MatI[:,i,:]= i
MatJ = np.zeros((p,x,y), dtype=np.float32)
for j in range(MatJ.shape[2]):
    MatJ[:,:,j] = j

MatI = np.reshape(MatI, [p,x,y, 1]).astype(np.float32)
MatJ = np.reshape(MatJ, [p,x,y, 1]).astype(np.float32)
MatI = torch.FloatTensor(MatI).cuda()##################################
MatJ = torch.FloatTensor(MatJ).cuda()###################3
MatI = torch.squeeze(MatI)
MatJ = torch.squeeze(MatJ)
def dense_normal_loss(pred, targetN,params,depthI,depthJ):
    depthI = depthI.permute(0, 2, 3, 1)
    depthJ = depthJ.permute(0, 2, 3, 1)

    predN_1 = torch.zeros_like(targetN)
    predN_2 = torch.zeros_like(targetN)
    params=params.permute(0, 2, 3, 1)

    f = params[:, :, :, 0]
    cx = params[:, :, :, 1]
    cy = params[:, :, :, 2]

    z1 = depthJ - pred
    z1 = torch.squeeze(z1)
    depthJ = torch.squeeze(depthJ)
    predN_1[:, :, :, 0] = ((MatJ - cx) * z1 + depthJ) * 1.0 / f
    predN_1[:, :, :, 1] = (MatI - cy) * z1 * 1.0 / f
    predN_1[:, :, :, 2] = z1

    z2 = depthI - pred
    z2 = torch.squeeze(z2)
    depthI = torch.squeeze(depthI)
    predN_2[:, :, :, 0] = (MatJ - cx) * z2  * 1.0 / f
    predN_2[:, :, :, 1] = ((MatI - cy) * z2 + depthI) * 1.0 / f
    predN_2[:, :, :, 2] = z2

    predN = torch.cross(predN_1, predN_2)

    pred_n = F.normalize(predN)
    pred_n = pred_n.contiguous().view(-1, 3)
    target_n = targetN.contiguous().view(-1, 3)

    loss_function = nn.CosineEmbeddingLoss()
    loss = loss_function(pred_n, target_n, Variable(torch.Tensor(pred_n.size(0)).cuda().fill_(1.0)))
    return loss



def get_loss(color_path_dense, normal_path_dense, color_attn, normal_attn, pred_surface_normal, stage, gt_depth, params, gt_surface_normal, gt_normal_mask):
    assert stage in {'D', 'N', 'A'}

    zero_loss = nn.MSELoss()(torch.ones(1, 1).to(gt_depth.device), torch.ones(1, 1).to(gt_depth.device))
    loss_d, loss_c, loss_n, loss_normal ,loss5= zero_loss, zero_loss, zero_loss, zero_loss, zero_loss

    if stage == 'N':
        loss_normal = normal_loss(pred_surface_normal, gt_surface_normal, gt_normal_mask)
    else:
 
        predicted_dense, pred_color_path_dense, pred_normal_path_dense = \
                            get_predicted_depth(color_path_dense, normal_path_dense, color_attn, normal_attn)


        loss_d, loss_c, loss_n = get_depth_loss(predicted_dense, pred_color_path_dense, pred_normal_path_dense, gt_depth)
        loss_normal = normal_loss(pred_surface_normal, gt_surface_normal, gt_normal_mask)

        conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        w = torch.from_numpy(k1).float().unsqueeze(0).unsqueeze(0).cuda()#################33
        conv1.weight = nn.Parameter(w,requires_grad=True)
        depthJ1 = conv1(predicted_dense)
        conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        w2 = torch.from_numpy(k2).float().unsqueeze(0).unsqueeze(0).cuda()####################
        conv2.weight = nn.Parameter(w2,requires_grad=True)
        depthI1 = conv2(predicted_dense)


        pred = predicted_dense.permute(0, 2, 3, 1)
        pred_surface_normal=pred_surface_normal.permute(0, 2, 3, 1)


        loss5 = dense_normal_loss(pred, pred_surface_normal, params, depthI1, depthJ1)
        # print(loss5)@！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！


        ###########jiashang!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # smo=SmoothnessLoss()
        # print(predicted_dense.size())


        # smooth_loss = 0#smo(predicted_dense[:,0,:,:])###########################################################################

    return loss_c, loss_n, loss_d, loss_normal,loss5###############################################################3

#