# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:58:50 2024

@author: 赵之谦
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
class NTXentLoss(nn.Module):
    def __init__(self, args, device, memory=True, sup_weight=0, BA=None):
        """
        初始化类
        Arguments:
            args (dict): 包含各种参数的配置字典
            device (torch.device): 计算设备（如 'cpu' 或 'cuda'）
            memory (nn.Module, optional): 存储memory，用于增强样本。默认为 True
            sup_weight (int, optional): 支持样本权重。默认值为 0
            BA(optional): BA模块，默认值为 None
        """
        super(NTXentLoss, self).__init__()
        self.args = args
        self.device = device
        self.memory = memory
        self.sup_weight = sup_weight
        self.BA = BA
    def negmask(self,z,qeaue,index,qeindex):
        # 扩展维度以便于广播
        index_expanded = index.view(-1, 1)   # [12, 1]
        qeindex_expanded = qeindex.view(1, -1)  # [1, 11]
        
        # 生成基本掩码矩阵，表示不同索引的位置
        mask_matrix = index_expanded != qeindex_expanded
        
        # 特殊处理 OOD 样本对（索引为 -1 的位置）
        # 确定 OOD 样本的掩码位置
        ood_mask_index = (index == -1).view(-1, 1)
        ood_mask_qeindex = (qeindex == -1).view(1, -1)
        
        # 找出 `index` 和 `qeindex` 都为 -1 的 OOD 样本对
        ood_mask = ood_mask_index & ood_mask_qeindex
        
        # 在 OOD 样本对中进一步检查 `z` 和 `qeaue` 是否相等，若相等则将掩码设为 `False`
        # 否则保持 `True`
        ood_pairwise_equal = torch.cdist(z, qeaue) == 0
        mask_matrix[ood_mask] = ~ood_pairwise_equal[ood_mask]
        
        return mask_matrix
        # mask_matrix = torch.zeros((len(index), len(qeindex)), dtype=torch.bool)
        # for i, idx in enumerate(index):
        #     for j, qeidx in enumerate(qeindex):
        #         # 如果索引相同且不是-1，则设置为True
        #         if idx != qeidx:
        #             mask_matrix[i, j] = True
        #         else:
        #             if idx == qeidx==-1 and torch.equal(z[i],qeaue[j]) or idx == qeidx!=-1:
        #                 mask_matrix[i, j] = False
        #             else:
        #                 mask_matrix[i, j] = True
  
        # return mask_matrix


    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    def positive(self, p, z):
        z = z.detach()
        z = F.normalize(z, dim=1)
        p = F.normalize(p, dim=1)
        return p * z

    def neg1(self, z1, z2, quean,batch_size,index,qeindex,epoch=None, pos_threshold=0.8):
        z = torch.cat((z1, z2), 0)
        zx=z
        z = F.normalize(z, dim=-1)
       
        quean1=quean
        if self.memory is None or epoch < self.args.memory_start_epoch:
            a = torch.mm(z, z.t().contiguous())
            neg = torch.exp(a / self.args.temp)
            sup_neg = neg
            mask = self.get_negative_mask(batch_size).to(self.device)
            #neg = neg.masked_select(mask).view(2 * batch_size, - 1)
            neg = torch.where(mask,neg, torch.zeros_like(neg))
            Ng = neg.sum(dim=-1)
            #print('wwNg',Ng)
            return sup_neg, neg, Ng
        else:
            # a = torch.mm(z, quean.T)
            # #global_min = a.min().detach()
            # #global_max =a.max().detach()
            
            # #对整个张量进行归一化
            # #normalized_out = (a  - global_min) / (global_max - global_min)
            # mask=a<= pos_threshold
            # #n_neg = torch.sum(a[ mask])     
            # a[a > pos_threshold] = 0.0
            # neg = torch.exp(a / self.args.temp)
            # #print('neg',neg.shape)
            # #print('a',a.shape)
            # sup_neg = neg
            # neg= torch.where(mask,neg, torch.zeros_like(neg))
            # Ng = neg.sum(dim=-1)#.div(n_neg).sum()
            # #print('normalized_out[mask]',normalized_out[mask].shape)
            # #print('Ng',Ng)
        
            # return sup_neg, neg, Ng
            #print('BA模块启动')
            quean = F.normalize(quean, dim=-1)
            a = torch.mm(z, quean.T)
            #a[a > 0.8] = 0.0
            #a=torch.clamp(a, max=5)
            #has_nan_or_infa = torch.isnan(a).any() or torch.isinf(a).any()
            #print("a中NaN or Inf values:", has_nan_or_infa)
            #print(a)
            neg = torch.exp(a / (self.args.temp*10))
            sup_neg = neg
            #print(sup_neg)
            mask = self.negmask(zx,quean1,index,qeindex)
            #mask=a<= pos_threshold
            neg = torch.where(mask,neg, torch.zeros_like(neg))
            Ng = neg.sum(dim=-1)
            #has_nan_or_inf = torch.isnan(Ng).any() or torch.isinf(Ng).any()
            #print("Ng中NaN or Inf values:", has_nan_or_inf)
            
            
            #print(Ng)
          
        
            return sup_neg, neg, Ng
            

        

    def forward(self, z_teacher, p_student, z_student, z_index, p_index,s_index,quean,qeindex,epoch):
        bsz = self.args.batch_size
        if epoch >= self.args.memory_start_epoch and self.args.enhance_batch and not self.args.use_nnclr:
            bsz = self.args.batch_size * (1 + self.args.topk)
        z1, z2 = torch.split(z_teacher, [bsz, bsz], dim=0)
        z1_s, z2_s = torch.split(z_student, [bsz, bsz], dim=0)
        p1, p2 = torch.split(p_student, [bsz, bsz], dim=0)
        z_index1, z_index2 = torch.split(z_index, [bsz, bsz], dim=0)
        s_index1, s_index2 = torch.split(s_index, [bsz, bsz], dim=0)
        p_index1, p_index2 = torch.split(p_index, [bsz, bsz], dim=0)
        if epoch >= self.args.memory_start_epoch and self.args.enhance_batch and not self.args.use_nnclr:
            pos1 = torch.exp(torch.sum(self.positive(p1, z2), dim=-1) / (self.args.temp*10))
            pos2 = torch.exp(torch.sum(self.positive(p2, z1), dim=-1) / (self.args.temp*10))
        else:
            pos1 = torch.exp(torch.sum(self.positive(p1, z2), dim=-1) / self.args.temp)
            pos2 = torch.exp(torch.sum(self.positive(p2, z1), dim=-1) / self.args.temp)  
        pos = torch.cat([pos1, pos2], dim=0)
        #print(pos)
        #index = index[:bsz * 2]

        if self.args.uniformity_config != "TT":
            z1 = z1_s
            index1=s_index1
            index2=z_index1
            if self.args.uniformity_config == "SS":
                z2 = z2_s
                index2=s_index2
            index=torch.cat((index1,index2), 0)
        else:
            index=z_index
 
        if self.args.use_memory_in_loss:
            sup_neg, neg, Ng = self.neg1(z1, z2, quean,bsz,index,qeindex,epoch)
        else:
            sup_neg, neg, Ng = self.neg1(z1, z2,quean,index,qeindex,bsz)

        if (index == -1).sum() != 0 and self.COLT:
            if epoch >= self.args.memory_start_epoch and self.args.enhance_batch and not self.args.use_nnclr:
                index_positive = index > 0
                quean_positive = qeindex > 0
                
                # 利用广播机制，直接比较正负相等的位置
                mask_matrix = (index_positive[:, None] == quean_positive[None, :])
                
                sup_pos_mask = mask_matrix.float()
                mask_pos_view = torch.zeros_like(sup_pos_mask)
                mask_pos_view[sup_pos_mask.bool()] = 1
            else:
                index=index[:self.args.batch_size]
                id_mask = (index != -1)
                ood_mask = (index == -1)
        
                sup_pos_mask = (((ood_mask.view(-1, 1) & ood_mask.view(1, -1)) | (
                            id_mask.view(-1, 1) & id_mask.view(1, -1))).repeat(2, 2) & (
                                    ~(torch.eye(index.shape[0] * 2).bool())))
        
                sup_pos_mask = sup_pos_mask.float()
        
                mask_pos_view = torch.zeros_like(sup_pos_mask)
                mask_pos_view[sup_pos_mask.bool()] = 1
        norm = pos + Ng

        neg_logits = torch.div(neg, norm.view(-1, 1))
        #print('neg',neg.shape)
        #print('loss_neg_logits',neg_logits.shape)
        if epoch >= self.args.memory_start_epoch and self.args.enhance_batch and not self.args.use_nnclr:
            neg_logits =neg_logits
        else:
            neg_logits = torch.cat([neg_logits[:bsz].unsqueeze(0), neg_logits[bsz:].unsqueeze(0)], dim=0)

        loss = (-torch.log(pos / (pos + Ng)))
        loss_reshape = loss.clone().detach().view(2, bsz).mean(0)
        loss = loss.mean()
        has_nan_or_infa = torch.isnan(loss).any() or torch.isinf(loss).any()
                #print("LOSS中NaN or Inf values:", has_nan_or_infa)
        if has_nan_or_infa:
            print('duibiloss中出现nan或者inf')
            sys.exit()        
        #print('loss1',loss)

        if (index == -1).sum() != 0 and self.COLT:
            if epoch >= self.args.memory_start_epoch and self.args.enhance_batch and not self.args.use_nnclr:   
                AA=sup_neg.T / (pos + Ng)
                #print('AA',AA)
                
                #has_nan_or_infa = torch.isnan(AA).any() or torch.isinf(AA).any()
                #print("AA中NaN or Inf values:", has_nan_or_infa)
                
                sup_loss = (-torch.log(AA.T))
                #print('LOSS',sup_loss)
                has_nan_or_infa = torch.isnan(sup_loss).any() or torch.isinf(sup_loss).any()
                #print("LOSS中NaN or Inf values:", has_nan_or_infa)
                if has_nan_or_infa:
                    print('sup_loss含有nan或inf')
                    
                    sys.exit()
            else:
                AA=sup_neg / (pos + Ng)
                sup_loss = (-torch.log(AA))
            #print('sup_loss_shape',sup_loss.shape)
            #print(mask_pos_view.shape)
            #print('1 / mask_pos_view.sum(1)',1 / mask_pos_view.sum(1))
            #has_nan_or_infa = torch.isnan(1 / mask_pos_view.sum(1)).any() or torch.isinf(1 / mask_pos_view.sum(1)).any()
            #print("1 / mask_pos_view.sum(1)中NaN or Inf values:", has_nan_or_infa)
            #print('mask_pos_view * sup_loss',mask_pos_view * sup_loss)
            #has_nan_or_infa = torch.isnan(mask_pos_view * sup_loss).any() or torch.isinf(mask_pos_view * sup_loss).any()
            #print("mask_pos_view * sup_loss中NaN or Inf values:", has_nan_or_infa)
            
            sup_loss = self.sup_weight * (1 / mask_pos_view.sum(1)) * (mask_pos_view * sup_loss).sum(1)
            if  torch.isnan(sup_loss).any():
                print('总loss出现nan或inf')
                sys.exit()
            loss += sup_loss.mean()
            #print('loss2',loss)

        return neg_logits, loss_reshape, loss
    
