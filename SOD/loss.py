import torch.nn as nn
import torch.nn.functional as F
import pytorch_ssim
import pytorch_iou

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, input, target):
        target = target.float()
        loss = hybrid_loss(input[:, :, :, :], target.unsqueeze(1))

        return loss

bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)
    
def hybrid_loss(pred,target):
    bce_out = bce_loss(pred,target)
    iou_out = iou_loss(pred,target)
    ssim_out = 1 - ssim_loss(pred,target)

    loss = bce_out + iou_out + ssim_out

    return loss