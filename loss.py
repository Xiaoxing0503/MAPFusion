import torch
import torch.nn as nn
import torch.nn.functional as F


def Sobelxy(x):
    kernelx = [[-1, 0, 1],
              [-2,0 , 2],
              [-1, 0, 1]]
    kernely = [[1, 2, 1],
              [0,0 , 0],
              [-1, -2, -1]]
    kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
    kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
    weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
    weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    sobelx=F.conv2d(x, weightx, padding=1)
    sobely=F.conv2d(x, weighty, padding=1)
    return torch.abs(sobelx)+torch.abs(sobely)

def L_Grad(image_A, image_B, image_fused):
    image_A_Y = image_A[:, :1, :, :]
    image_B_Y = image_B[:, :1, :, :]
    image_fused_Y = image_fused[:, :1, :, :]
    gradient_A = Sobelxy(image_A_Y)
    gradient_B = Sobelxy(image_B_Y)
    gradient_fused = Sobelxy(image_fused_Y)
    gradient_joint = torch.max(gradient_A, gradient_B)
    Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
    return Loss_gradient

def L_Int(image_A, image_B, image_fused):
    image_A_Y = image_A[:, :1, :, :]
    image_B_Y = image_B[:, :1, :, :]
    image_fused_Y = image_fused[:, :1, :, :]
    # x_in_max=torch.add(image_A_Y,image_B_Y)/2
    x_in_max = torch.max(image_A_Y, image_B_Y)
    loss_in = F.l1_loss(x_in_max, image_fused_Y)
    return loss_in

# class Fusionloss(nn.Module):
#     def __init__(self):
#         super(Fusionloss, self).__init__()
#         self.sobelconv=Sobelxy()
#
#     def forward(self,image_vis,image_ir,generate_img):
#         image_y=image_vis[:,:1,:,:]
#         x_in_max=torch.max(image_y,image_ir)
#         loss_in=F.l1_loss(x_in_max,generate_img)
#         y_grad=self.sobelconv(image_y)
#         ir_grad=self.sobelconv(image_ir)
#         generate_img_grad=self.sobelconv(generate_img)
#         x_grad_joint=torch.max(y_grad,ir_grad)
#         loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
#         loss_total=loss_in+10*loss_grad
#         return loss_total,loss_in,loss_grad
#
# class Sobelxy(nn.Module):
#     def __init__(self):
#         super(Sobelxy, self).__init__()
#         kernelx = [[-1, 0, 1],
#                   [-2,0 , 2],
#                   [-1, 0, 1]]
#         kernely = [[1, 2, 1],
#                   [0,0 , 0],
#                   [-1, -2, -1]]
#         kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
#         kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
#         self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
#         self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
#     def forward(self,x):
#         sobelx=F.conv2d(x, self.weightx, padding=1)
#         sobely=F.conv2d(x, self.weighty, padding=1)
#         return torch.abs(sobelx)+torch.abs(sobely)