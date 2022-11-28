import copy
from multiprocessing import pool
from tkinter import N
from turtle import forward
from unicodedata import bidirectional
from sklearn.metrics import pair_confusion_matrix
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from resnet import resnet50, resnet18
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from sklearn.cluster import DBSCAN

import numpy as np

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if m.bias:
            init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_kaiming_2(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base
        self.layer4 = copy.deepcopy(self.base.layer4)

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        t_x = self.layer4(x)
        x = self.base.layer4(x)
        return x,t_x

class SpatialRefinementModule(nn.Module):
    """
        Spatial Refinement Module
    """
    def __init__(self):
        super(SpatialRefinementModule, self).__init__()    
        self.cosine_similarity = nn.CosineSimilarity(dim=2, eps=1e-8)
        self.softmax = nn.Softmax(dim=-1)
        self.tau = 0.4
    
    def forward(self, l_feat, g_feat):

        l_feat_tmp = l_feat.unsqueeze(dim=3) # [32, 9, 2048, 1]
        g_feat_tmp = g_feat.unsqueeze(dim=1) # [32, 1, 2048, 9]
        cos_sim = self.cosine_similarity(l_feat_tmp,g_feat_tmp)
        
        # similarity matrix
        sim_matrix = torch.exp(cos_sim/self.tau)

        # degree matrix
        degree_matrix = torch.diag_embed(torch.sum(sim_matrix,dim=2))
        degree_matrix_inv = torch.linalg.inv(degree_matrix)
        
        # normalize
        sim_matrix = torch.bmm(degree_matrix_inv, sim_matrix) 

        g_feat = g_feat.permute(0,2,1)
        g_enh = torch.bmm(sim_matrix, g_feat)
        feat_enh = l_feat + g_enh
        return feat_enh


class embed_net(nn.Module):
    def __init__(self, class_num, eps_increment=0.5, min_samples=3, drop=0.2, arch="resnet50", global_mutual_learning=False, local_mutual_learning=False):
        super(embed_net, self).__init__()

        # hyper parameters
        pool_dim = 2048
        self.dropout = drop
        self.eps_increment = eps_increment
        self.min_samples = min_samples

        # feature extract
        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
         
        # spatial refinement module
        self.SRU = SpatialRefinementModule()

        # temporal weighted
        self.lstm = nn.LSTM(2048, 2048, 2)
        self.spatial_proj = nn.Linear(pool_dim, 512, bias=False)
        self.temporal_proj = nn.Linear(pool_dim, 512, bias=False)

        self.linear_proj = nn.Sequential(
            nn.Linear(512*512, 1024, bias=False),
            nn.BatchNorm1d(1024),   
            nn.ReLU(),
            nn.Linear(1024, 2048, bias=False),
            nn.BatchNorm1d(2048),   
            nn.ReLU()
        )  

        # classification layers
        self.l2norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.local_bottleneck = nn.BatchNorm1d(pool_dim)
        self.local_bottleneck.bias.requires_grad_(False)  # no shift

        # initialize
        self.bottleneck.apply(weights_init_kaiming)
        self.local_bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        self.global_mutual_learning = global_mutual_learning
        self.local_mutual_learning = local_mutual_learning


        if self.global_mutual_learning:
            self.rgb_global_classifier = nn.Linear(pool_dim, class_num, bias=False)
            self.ir_global_classifier = nn.Linear(pool_dim, class_num, bias=False)

            # mean classifier
            self.rgb_global_classifier_  = nn.Linear(pool_dim, class_num, bias=False)
            self.rgb_global_classifier_.weight.requires_grad_(False)  
            self.rgb_global_classifier_.weight.data = self.rgb_global_classifier.weight.data  

            self.ir_global_classifier_  = nn.Linear(pool_dim, class_num, bias=False)
            self.ir_global_classifier_.weight.requires_grad_(False)  
            self.ir_global_classifier_.weight.data = self.ir_global_classifier.weight.data  
            self.global_update_rate = 0.2
            
            self.rgb_global_classifier.apply(weights_init_classifier)
            self.ir_global_classifier.apply(weights_init_classifier)

        if self.local_mutual_learning:
            self.rgb_local_classifier = nn.Linear(pool_dim, class_num, bias=False)
            self.ir_local_classifier = nn.Linear(pool_dim, class_num, bias=False)

            # mean classifier
            self.rgb_local_classifier_  = nn.Linear(pool_dim, class_num, bias=False)
            self.rgb_local_classifier_.weight.requires_grad_(False)  
            self.rgb_local_classifier_.weight.data = self.rgb_local_classifier.weight.data  

            self.ir_local_classifier_  = nn.Linear(pool_dim, class_num, bias=False)
            self.ir_local_classifier_.weight.requires_grad_(False) 
            self.ir_local_classifier_.weight.data = self.ir_local_classifier.weight.data  
            self.local_update_rate = 0.2
            
            self.rgb_local_classifier.apply(weights_init_classifier)
            self.ir_local_classifier.apply(weights_init_classifier)


    def forward(self, x1, x2, modal=0, seq_len = 6):
        b, c, h, w = x1.size()
        t = seq_len
        x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
        x2 = x2.view(int(b * seq_len), int(c / seq_len), h, w)
        
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        x, x_t = self.base_resnet(x)

        # baseline
        # x_l = self.avgpool(x_t).squeeze()
        # x_l_enh = x_l.view(x_l.size(0)//t, t, -1)
        # x_g_enh = torch.mean(x_l_enh, dim=1)
        # feat = self.bottleneck(x_g_enh)
        
        # spatial refinement module
        x_gap = rearrange(x_t,'(b t) n h w->b t n h w', b=x_t.shape[0]//t, t=t, n=x_t.shape[1], h=x_t.shape[2], w=x_t.shape[3])
        x_gap = torch.mean(x_gap, dim=1).squeeze()
        x_gap = x_gap.reshape(x_gap.shape[0],x_gap.shape[1],9,x_gap.shape[2]//3,x_gap.shape[3]//3)
        x_gap = self.avgpool(x_gap).squeeze()

        x_l_part = x_t.reshape(x_t.shape[0],x_t.shape[1],9,x_t.shape[2]//3,x_t.shape[3]//3)
        x_l_part = self.avgpool(x_l_part).squeeze()
        x_l = rearrange(x_l_part, '(b t) n p-> t b p n', b=x_t.shape[0]//t, t=t, p=9, n=x_t.shape[1])

        x_l_enh_list = []
        for i in range(t):
            x_l_enh_tmp = self.SRU(x_l[i], x_gap)
            x_l_enh_list.append(x_l_enh_tmp)
        
        x_l_enh = torch.stack(x_l_enh_list)
        x_l_enh = rearrange(x_l_enh, 't b p n-> b t p n', t=t, b=x_t.shape[0]//t, p=9, n=x_t.shape[1])
        
        x_l_enh = torch.mean(x_l_enh, dim=2)

        # spatial-temporal interaction module
        x_lstm = x_l_enh.permute(1, 0, 2)
        h0 = torch.zeros(2, x_lstm.shape[1], x_lstm.shape[2]).cuda()
        c0 = torch.zeros(2, x_lstm.shape[1], x_lstm.shape[2]).cuda()
        if self.training: self.lstm.flatten_parameters()
        
        output, (hn, cn) = self.lstm(x_lstm, (h0, c0))  # [6, 32, 4096]

        temporal_relation_list = []
        for i in range(output.shape[0]):

            temporal_feat = output[i]
            frame_feat = x_l_enh.permute(1,0,2)[i]

            temporal_proj = self.temporal_proj(temporal_feat).unsqueeze(2)
            spatial_proj = self.spatial_proj(frame_feat).unsqueeze(1)

            inter_map = torch.bmm(temporal_proj, spatial_proj)  # [32, 256, 256]
            inter_map = rearrange(inter_map, 'b m n-> b (m n)')

            relation_out = self.linear_proj(inter_map)
            temporal_relation_list.append(relation_out)

        temporal_relation = torch.stack(temporal_relation_list)
        x_global_weighted = 0.2*temporal_relation.permute(1,0,2) + 0.8*x_l_enh
        x_g_enh = torch.mean(x_global_weighted, dim=1).squeeze()
        
        feat = self.bottleneck(x_g_enh)
    
        if self.training:
            logits = self.classifier(feat)

            cluster_center_list = []
            clusters_num_list = []
            outliers_num_list = []

            for i in range(0, int(x_l_enh.shape[0]/2), 2):
                tmp_list = []
                tmp_list.append(x_l_enh[i])
                tmp_list.append(x_l_enh[i+1])
                tmp_list.append(x_l_enh[i+16])
                tmp_list.append(x_l_enh[i+17])
                
                local_feat = torch.stack(tmp_list)
                local_feat = rearrange(local_feat, 'b t n->(b t) n')
                local_feat_detach = local_feat.detach().cpu().numpy()

                # DBSCAN 
                eps = 0.1
                cluster_method = DBSCAN(eps=eps, min_samples=self.min_samples)
                cluster_res = cluster_method.fit(local_feat_detach)
                labels = cluster_res.labels_
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                if -1 in labels:  
                    outliers = np.where(labels == -1)[0]
                    outliers_num = len(outliers)
                else:
                    outliers_num = 0

                while(n_clusters_ == 0 or outliers_num > 0): 
                    eps = eps + self.eps_increment
                    cluster_method = DBSCAN(eps=eps, min_samples=3)
                    cluster_res = cluster_method.fit(local_feat_detach)
                    labels = cluster_res.labels_
                    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)     
                    if -1 in labels:  
                        outliers = np.where(labels == -1)[0]
                        outliers_num = len(outliers)
                    else:
                        outliers_num = 0
                
                labels = torch.tensor(labels).cuda()
                labels_uni = torch.unique(labels)
 
                all_num = len(labels_uni) 
                if -1 in labels_uni:
                    clusters_num = all_num - 1
                else:
                    clusters_num = all_num

                clusters_num_list.append(clusters_num)
                outliers_num_list.append(outliers_num)
                local_centers = []

                for j in range(all_num):
                    
                    if labels_uni[j] == -1:
                        continue

                    index_tmp = torch.where(labels==labels_uni[j])[0]
                    local_center_tmp = torch.zeros(2048).cuda()
                    
                    for index in index_tmp:
                        local_center_tmp += local_feat[index]
                    local_centers.append(local_center_tmp/len(index_tmp)) 
                cluster_center_list.append(torch.stack(local_centers))

            cluster_center = torch.cat(cluster_center_list, dim=0)

            # Global-Local Mutual Learning Module
            x1_global_feat = feat[0:b,:]
            x2_global_feat = feat[b:,:]

            x1_global_logits = self.rgb_global_classifier(x1_global_feat)  
            x2_global_logits = self.ir_global_classifier(x2_global_feat)  
            x1_global_logits_ = self.ir_global_classifier_(x1_global_feat)
            x2_global_logits_ = self.rgb_global_classifier_(x2_global_feat)

            global_logits = torch.cat((x1_global_logits, x2_global_logits), dim=0)
            global_logits_ = torch.cat((x1_global_logits_, x2_global_logits_), dim=0)
            global_logits = F.softmax(global_logits, dim=1)
            global_logits_ = F.log_softmax(global_logits_, dim=1)

            x_local_feat = self.local_bottleneck(rearrange(x_l_enh, 'b c n->(b c) n'))
            x1_local_feat = x_local_feat[0:b*t,...]
            x2_local_feat = x_local_feat[b*t:,...]

            x1_local_logits = repeat(self.rgb_local_classifier(x1_global_feat).unsqueeze(dim=1), 'b 1 d -> b n d', n=seq_len)
            x2_local_logits = repeat(self.ir_local_classifier(x2_global_feat).unsqueeze(dim=1), 'b 1 d -> b n d', n=seq_len)
            
            x1_local_logits = rearrange(x1_local_logits, 'b n d -> (b n) d')
            x2_local_logits = rearrange(x2_local_logits, 'b n d -> (b n) d')

            x1_local_logits_ = self.ir_local_classifier_(x1_local_feat)
            x2_local_logits_ = self.rgb_local_classifier_(x2_local_feat)

            local_logits = torch.cat((x1_local_logits, x2_local_logits), dim=0)
            local_logits_ = torch.cat((x1_local_logits_, x2_local_logits_), dim=0)

            local_logits = F.softmax(local_logits, dim=1)
            local_logits_ =  F.log_softmax(local_logits_, dim=1)

            return x_g_enh, logits, cluster_center, clusters_num_list, global_logits, global_logits_, x1_global_logits, x2_global_logits, local_logits, local_logits_, x1_local_logits, x2_local_logits
        else:
            return self.l2norm(feat)
    
    def update_mean_classifier(self, epoch, warm_up_epochs, mode=0):
        if epoch < warm_up_epochs:
            pass
        elif epoch == warm_up_epochs-1:
            if mode == 0:
                with torch.no_grad():
                    self.rgb_global_classifier_.weight.data = self.rgb_global_classifier.weight.data
                    self.ir_global_classifier_.weight.data = self.ir_global_classifier.weight.data
            elif mode == 1:
                with torch.no_grad():
                    self.rgb_local_classifier_.weight.data = self.rgb_local_classifier.weight.data
                    self.ir_local_classifier_.weight.data = self.ir_local_classifier.weight.data
            elif mode == 2:
                with torch.no_grad():
                    self.rgb_global_classifier_.weight.data = self.rgb_global_classifier.weight.data
                    self.ir_global_classifier_.weight.data = self.ir_global_classifier.weight.data   
                    self.rgb_local_classifier_.weight.data = self.rgb_local_classifier.weight.data
                    self.ir_local_classifier_.weight.data = self.ir_local_classifier.weight.data
        else:
            if mode == 0:
                with torch.no_grad():
                    self.rgb_global_classifier_.weight.data = self.rgb_global_classifier_.weight.data * (1 - self.global_update_rate) \
                                                        + self.rgb_global_classifier.weight.data * self.global_update_rate
                    self.ir_global_classifier_.weight.data = self.ir_global_classifier_.weight.data * (1 - self.global_update_rate) \
                                                        + self.ir_global_classifier.weight.data * self.global_update_rate
            elif mode == 1:
                with torch.no_grad():    
                    self.rgb_local_classifier_.weight.data = self.rgb_local_classifier_.weight.data * (1 - self.local_update_rate) \
                                                        + self.rgb_local_classifier.weight.data * self.local_update_rate
                    self.ir_local_classifier_.weight.data = self.ir_local_classifier_.weight.data * (1 - self.local_update_rate) \
                                                        + self.ir_local_classifier.weight.data * self.local_update_rate
            elif mode == 2:
                with torch.no_grad():
                    self.rgb_global_classifier_.weight.data = self.rgb_global_classifier_.weight.data * (1 - self.global_update_rate) \
                                                        + self.rgb_global_classifier.weight.data * self.global_update_rate
                    self.ir_global_classifier_.weight.data = self.ir_global_classifier_.weight.data * (1 - self.global_update_rate) \
                                                        + self.ir_global_classifier.weight.data * self.global_update_rate
                    self.rgb_local_classifier_.weight.data = self.rgb_local_classifier_.weight.data * (1 - self.local_update_rate) \
                                                        + self.rgb_local_classifier.weight.data * self.local_update_rate
                    self.ir_local_classifier_.weight.data = self.ir_local_classifier_.weight.data * (1 - self.local_update_rate) \
                                                        + self.ir_local_classifier.weight.data * self.local_update_rate

