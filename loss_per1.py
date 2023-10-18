import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg16
import warnings
import os


warnings.filterwarnings('ignore')

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

def vgg16_loss_w(feature_module,loss_func,y,y_):
    out=feature_module(y)
    out_=feature_module(y_)
    s = 0
    for i in range(out.size(1)):
        grad_out = Sobelxy(out[:,i:i+1,:,:])
        s += torch.norm(grad_out,p='fro')
    s = s/out.size(1)
    loss=loss_func(out,out_)
    return loss,s


# 计算特征提取模块的感知损失
def vgg16_loss(feature_module,loss_func,y,y_):
    out=feature_module(y)
    out_=feature_module(y_)
    loss=loss_func(out,out_)
    return loss

# 获取指定的特征提取模块
# def get_feature_module(layer_index,device=None):





# 计算指定的组合模块的感知损失
class PerceptualLoss(nn.Module):
    def __init__(self,loss_func,layer_indexs=None,device=None):
        super(PerceptualLoss, self).__init__()
        self.creation=loss_func
        self.layer_indexs=layer_indexs
        self.device=device

    def forward(self,y,y_):
        loss=0
        s = 0
        vgg = vgg16(pretrained=True, progress=True).features.cuda()
        vgg[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        vgg.eval()
        # 冻结参数
        for parm in vgg.parameters():
            parm.requires_grad = False

        for index in self.layer_indexs:
            feature_module = vgg[0:index + 1]
            feature_module.cuda()
            loss_1,s_1 = vgg16_loss_w(feature_module,self.creation,y,y_)
            loss+=loss_1
            s+=s_1
        loss=loss/len(self.layer_indexs)
        s = s/len(self.layer_indexs)
        return loss,s

def per_loss(x,y):
    device = torch.device("cuda")
    x, y = x.cuda(), y.cuda()

    # layer_indexs = [29]
    layer_indexs = [3, 8, 15, 22, 29]
    # 基础损失函数：确定使用那种方式构成感知损失，比如MSE、MAE
    loss_func = nn.MSELoss().cuda()
    # 感知损失
    creation = PerceptualLoss(loss_func, layer_indexs, device)
    perceptual_loss,s = creation(x, y)
    return perceptual_loss,s


if __name__ == "__main__":
    x = torch.ones((1, 1, 256, 256))
    y = torch.zeros((1, 1, 256, 256))
    perceptual_loss,s=per_loss(x,y)
    print(perceptual_loss,s)