import torch
import torch.nn as nn
from torch.nn import functional as F
import copy

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, device, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.device = device
        self.centers = None

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        #x = F.normalize(x, p=2, dim=1)
        self.centers.data = F.normalize(self.centers.data, p=2, dim=1)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        #distmat = F.linear(x, self.centers)
        
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
    
    def _add_classes(self, n_classes):
        self.num_classes = n_classes

        new_centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

        if self.centers is not None:
            new_centers.data[:self.num_classes] = copy.deepcopy(self.centers.data)
            
        del self.centers
        self.centers = new_centers