import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable
from scipy.sparse import block_diag
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class ClusterLoss(nn.Module):
    def __init__(self):
        super(ClusterLoss, self).__init__()


    def forward(self, x_clusters, cluster_num_list):
        
        x_clusters_list = [] 
        x_clusters_mean_list = []
        x_clusters_mean_expand_list = []
        ptr = 0

        for i in range(len(cluster_num_list)):
            clusters_num = cluster_num_list[i]
            if clusters_num == 1: 
                x_clusters_mean_tmp = x_clusters[ptr]
                x_clusters_mean_list.append(x_clusters_mean_tmp)
            else:
                x_clusters_mean_tmp = torch.mean(x_clusters[ptr:ptr+clusters_num], dim=0)
                x_clusters_mean_list.append(x_clusters_mean_tmp)
                x_clusters_mean_expand_list.append(repeat(x_clusters_mean_tmp.unsqueeze(0), '1 n->b n', b=clusters_num))
                x_clusters_list.append(x_clusters[ptr:ptr+clusters_num])  
            ptr = ptr + clusters_num

        x_clusters_mean = torch.stack(x_clusters_mean_list)
        if len(x_clusters_list) != 0: 
            x_clusters_cal = torch.cat(x_clusters_list, dim=0)
            x_clusters_mean_expand = torch.cat(x_clusters_mean_expand_list, dim=0)
        
            dist = torch.sqrt(torch.pow(x_clusters_cal-x_clusters_mean_expand, 2).sum(dim=1))
            loss_1 = torch.mean(dist, dim=0)

            # Compute pairwise distance
            dist_2 = torch.pow(x_clusters_mean, 2).sum(dim=1, keepdim=True).expand(8, 8)
            dist_2 = dist_2 + dist_2.t()
            dist_2.addmm_(1, -2, x_clusters_mean, x_clusters_mean.t())
            dist_2 = dist_2.clamp(min=1e-12).sqrt()  # for numerical stability

            dist_2 = dist_2.clamp(max=1.0)
            ref_matrix = (torch.ones((8,8)) - torch.eye(8)).cuda()
            loss_2 = torch.sqrt(torch.pow(dist_2-ref_matrix, 2).sum(dim=1))
            loss_2 = torch.mean(loss_2, dim=0)

            loss = loss_1 + loss_2

            return loss
        else:

            # Compute pairwise distance
            dist_2 = torch.pow(x_clusters_mean, 2).sum(dim=1, keepdim=True).expand(8, 8)
            dist_2 = dist_2 + dist_2.t()
            dist_2.addmm_(1, -2, x_clusters_mean, x_clusters_mean.t())
            dist_2 = dist_2.clamp(min=1e-12).sqrt()  # for numerical stability

            dist_2 = dist_2.clamp(max=1.0)
            ref_matrix = (torch.ones((8,8)) - torch.eye(8)).cuda()
            loss_2 = torch.sqrt(torch.pow(dist_2-ref_matrix, 2).sum(dim=1))
            loss_2 = torch.mean(loss_2, dim=0)

            loss = loss_2

            return loss

class CenterClusterLoss(nn.Module):
    def __init__(self):
        super(CenterClusterLoss, self).__init__()


    def forward(self, x_clusters, x1_rep, x2_rep):
        
        x_clusters_12 = rearrange(repeat(x_clusters.unsqueeze(dim=1), 'b 1 n->b t n', t=12), 'b t n->(b t) n')
        x1_rep = rearrange(x1_rep, 'b t n->(b t) n')
        x2_rep = rearrange(x2_rep, 'b t n->(b t) n')
        dist_x1 = torch.sqrt(torch.pow(x1_rep-x_clusters_12, 2).sum(dim=1))
        dist_x2 = torch.sqrt(torch.pow(x2_rep-x_clusters_12, 2).sum(dim=1))
        dist = torch.cat((dist_x1, dist_x2), dim=0)
        loss_1 = torch.mean(dist, dim=0)

        # Compute pairwise distance
        dist_2 = torch.pow(x_clusters, 2).sum(dim=1, keepdim=True).expand(8, 8)
        dist_2 = dist_2 + dist_2.t()
        dist_2.addmm_(1, -2, x_clusters, x_clusters.t())
        dist_2 = dist_2.clamp(min=1e-12).sqrt()  # for numerical stability
        
        dist_2 = dist_2.clamp(max=1.0)
        ref_matrix = (torch.ones((8,8)) - torch.eye(8)).cuda()
        loss_2 = torch.sqrt(torch.pow(dist_2-ref_matrix, 2).sum(dim=1))
        loss_2 = torch.mean(loss_2, dim=0)

        loss = loss_1 + loss_2
        return loss

class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    def __init__(self, batch_size, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.batch_size = batch_size
        self.mask = torch.eye(batch_size)
    def forward(self, input, target):
        """
        Args:
        - input: feature matrix with shape (batch_size, feat_dim)
        - target: ground truth labels with shape (num_classes)
        """
        n = self.batch_size
        input1 = input.narrow(0,0,n)
        input2 = input.narrow(0,n,n)
        
        # Compute pairwise distance, replace by the official when merged
        dist = pdist_torch(input1, input2)
        
        # For each anchor, find the hardest positive and negative
        # mask = target1.expand(n, n).eq(target1.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i,i].unsqueeze(0))
            dist_an.append(dist[i][self.mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct*2
        
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx