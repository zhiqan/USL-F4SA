# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 06:27:28 2024

@author: Owner
"""

from __future__ import print_function

import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
#import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate_fewshot
from BA2 import BA

from transform.build_transform import DataAugmentationBECLR
from utils.utils1 import (LARS, AverageMeter, bool_flag,
                         build_fewshot_loader,build_fewshot_loader_HIT,
                         build_train_loader, cancel_gradients_last_layer,
                         cosine_scheduler, get_params_groups,
                         get_world_size, grad_logger,
                         load_student_teacher, save_student_teacher)


from utils_2 import build_student_teacher,sample_batch#,nt_xent,BECLRLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dataset.data_CWRU import CWRUDataset1,Random_PUDataset

from loss3 import NTXentLoss
##
##tem=0.5 or 1
#lr=0.1
budget = 4000
def args_parser():
    parser = argparse.ArgumentParser(
        'BECLR training arguments', add_help=False)

    parser.add_argument('--cnfg_path', type=str,
                        default=None, help='path to train configuration file')
    parser.add_argument('--save_path', type=str,
                        default='D:\\无监督元学习_数据裁剪_域外数据\\BECLR-main', help='path for saving checkpoints')
    parser.add_argument('--log_path', type=str,
                        default=None, help='path for tensorboard logger')
    parser.add_argument('--data_path', type=str,
                        default=None, help='path to dataset root')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        choices=['tieredImageNet', 'miniImageNet',
                                 'CIFAR-FS', 'FC100'],
                        help='choice of dataset for pre-training')
    parser.add_argument('--print_freq', type=int,
                        default=10, help='print frequency')
    parser.add_argument('--num_workers', type=int,
                        default=0, help='num of workers to use')
    parser.add_argument('--ckpt_freq', type=int,
                        default=15, help='checkpoint save frequency')
    parser.add_argument('--ckpt_path', type=str,
                        default='D:\\自监督元学习故障诊断+域外特征+数据剪裁\\xunlian', help='path to model checkpoint')
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    # model settings
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet10', 'resnet18',
                                 'resnet34', 'resnet50'],
                        help='Choice of backbone network for the encoder')
    parser.add_argument('--size', type=int, default=224,
                        help='input image size')
    parser.add_argument('--enhance_batch', default=True, type=bool_flag,
                        help='Whether to artificially enhance the batch size')
    parser.add_argument('--topk', default=10, type=int,
                        help='Number of topk NN to extract, when enhancing the \
                        batch size.')
    parser.add_argument('--out_dim', default=512, type=int,
                        help='Dimensionality of output.')
    parser.add_argument('--momentum_teacher', default=0.9, type=float,
                        help='Base EMA parameter for teacher update. The value \
                        is increased to 1 during training with cosine schedule.')
    parser.add_argument('--freeze_last_layer', default=1, type=int,
                        help='Number of epochs during which we keep the output\
                        layer fixed. Typically doing so during the first epoch \
                        helps training. ')

    # contrastive loss settings
    parser.add_argument('--uniformity_config', type=str, default='SS',
                        choices=['ST', 'SS', 'TT'],
                        help='Choice of unifmormity configurations for view 1\
                        and view 2(SS: both views from student, ST: one view\
                        from student & the other from teacher, TT: both views\
                        from teacher)')
                            
     ############+
#####                       
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')
    
    parser.add_argument('--adj_tau', type=str, default='step',
                        help='cos or step')    
    parser.add_argument('--lamb_neg', type=float, default=0.2,
                        help='lambda for uniformity loss')
    parser.add_argument('--use_memory_in_loss', default=True, type=bool_flag,
                        help='Whether to use memory in uniformity loss')
    parser.add_argument('--pos_threshold', default=0.8, type=float,
                        help='When the cosine similarity of two embeddings is \
                        above this threshold, they are treated as positives, \
                        and masked out from the uniformity loss')

    # memory settings
    parser.add_argument("--memory_scale", default=20, type=int,
                        help="memory size compared to number of clusters, i.e.:\
                        memory_size = memory_scale * num_clusters")
    parser.add_argument('--num_clusters', type=int,
                        default=300, help='number of memory clusters')
    parser.add_argument('--cluster_algo', type=str, default='kmeans',
                        choices=['kmeans'], help='Choice of clustering algorithm\
                        for initializing the memory clusters')
    parser.add_argument('--recluster', default=True, type=bool_flag,
                        help='Wether to occasionally recluster the memory \
                        embeddings all together')
    parser.add_argument('--cluster_freq', type=int,
                        default=60, help='memory reclustering frequency')
    parser.add_argument('--memory_start_epoch', default=32, type=int,
                        help=' Epoch after which enhance_batch is \
                        activated.')
    parser.add_argument('--memory_momentum', default=0.0, type=float,
                        help='the momentum value for updating the cluster \
                        means in the memory')
    parser.add_argument('--memory_dist_metric', type=str, default='euclidean',
                        choices=['cosine', 'euclidean'], help='Choice of \
                        distance metric for the OT cost matrix in the memory')
    parser.add_argument("--sinkhorn_iterations", default=10, type=int,
                        help='number of iterations in Sinkhorn-Knopp algorithm')
    parser.add_argument("--epsilon", default=0.05, type=float,
                        help="regularization parameter for Sinkhorn-Knopp \
                        algorithm")
    parser.add_argument("--visual_freq", default=15, type=int,
                        help='memory embeddings visualization frequency')

    # masking settings
    parser.add_argument('--patch_size', type=int, default=16,
                        help='size of input square patches for masking in \
                        pixels, default 16 (for 16x16 patches)')
    parser.add_argument('--mask_ratio', default=0.0, type=float, nargs='+',
                        help='Ratio of masked-out patches. If a list of ratio\
                        is specified, one of them will be randomly choosed for\
                        each image.')
    parser.add_argument('--mask_ratio_var', default=0, type=float, nargs='+',
                        help='Variance of partial masking ratio. Length \
                        should be indentical to the length of mask_ratio. \
                        0 for disabling. ')
    parser.add_argument('--mask_shape', default='block',
                        type=str, help='Shape of partial prediction.')
    parser.add_argument('--mask_start_epoch', default=0, type=int,
                        help='Start epoch to perform masked image prediction.')

    # optimization settings
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer', choices=['adamw', 'lars', 'sgd'])
    parser.add_argument('--lr', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=1.0e-04, help='weight decay')
    parser.add_argument('--min_lr', type=float,
                        default=1.0e-6, help='final learning rate')
    parser.add_argument('--weight_decay_end', type=float,
                        default=0.0001, help='final weight decay')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='batch_size')
    parser.add_argument('--epochs', type=int, default=182,
                        help='number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='number of warmup epochs')

    # few-shot evaluation settings
    parser.add_argument('--n_way', type=int, default=5,
                        help='number of classes per episode')
    parser.add_argument('--n_query', type=int, default=15,
                        help='number of queries per episode')
    parser.add_argument('--n_test_task', type=int,
                        default=600, help='total test few-shot episodes')
    parser.add_argument('--test_batch_size', type=int,
                        default=5, help='episode_batch_size')
    parser.add_argument('--eval_freq', type=int,
                        default=15, help='evaluation frequency')

    # parallelization settings
    parser.add_argument("--dist_url", default="env://", type=str,
                        help="""url used to set up distributed training; see \
                        https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--world_size", default=-1, type=int,
                        help='number of processes: it is set automatically and \
                        should not be passed as argument')
    parser.add_argument('--rank', default=0, type=int,
                        help='rank of this process: it is set automatically \
                        and should not be passed as argument')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='this argument is not used and should be ignored')

    parser.add_argument('--eucl_norm', default=True, type=bool_flag,
                        help='Whether normalize before applying eucl distance')

    parser.add_argument('--use_nnclr', default=False, type=bool_flag,
                        help='Whether to use the memory of nnclr')

    return parser


parser = argparse.ArgumentParser(
    'BECLR training arguments', parents=[args_parser()])

args = parser.parse_args()


def train_one_epoch(train_loader: torch.utils.data.DataLoader,
                    student: nn.Module,
                    teacher: nn.Module,
                    optimizer: nn.Module,
                    epoch: int,
                    momentum_tail_score:torch.Tensor,
                    shadow:torch.Tensor,
                    lr_schedule: np.array,
                    wd_schedule: np.array,
                    momentum_schedule: np.array,
                    writer: SummaryWriter,
                    beclr_loss: nn.Module,
                    args: dict,
                    teacher_nn_replacer: BA,
                    student_nn_replacer: BA,
                    student_f_nn_replacer: BA = None):
    """
    Performs one epoch of the self-supervised pre-training stage of the network.
    
    Arguments:
        - train_loader (torch.utils.data.DataLoader): train dataloader
        - student (nn.Module): student network
        - teacher (nn.Module): teacher network
        - optimizer (nn.Module): optimizer module
        
        - epoch (int): current training epoch
        - lr_schedule (np.array): learning rate cosine schedule
        - wd_schedule (np.array): weight decay cosine schedule
        - momentum_schedule (np.array): teacher momentum cosine schedule
        - writer (SummaryWriter): TensorBoard SummaryWritter
        - beclr_loss (nn.Module): contrastive loss module
        - args (dict): parsed keyword training arguments
        - teacher_nn_replacer: teacher memory queue 
        - student_nn_replacer: student memory queue 
        - student_f_nn_replacer: student projections memory queue (optional)
    
    Returns:
        - The average loss value for the current epoch
    """
    student.train()
    
    # initialize logging metrics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_hist = AverageMeter()
    loss_pos_hist = AverageMeter()
    loss_neg_hist = AverageMeter()
    std_hist = AverageMeter()
    
    end = time.time()
#    train_loader=data_loader
    
    for it, data in enumerate(tqdm(train_loader)):
        # if it==0:
        #     break
        images = data[0]
        index= data[1]
        
        
        data_time.update(time.time() - end)
        bsz = images.shape[0]
        
        if bsz != args.batch_size:
            continue
        
        # # common params
        names_q, params_q, names_k, params_k = [], [], [], []
        for name_q, param_q in student.named_parameters():
            names_q.append(name_q)
            params_q.append(param_q)
        for name_k, param_k in teacher.named_parameters():
            names_k.append(name_k)
            params_k.append(param_k)
        names_common = list(set(names_q) & set(names_k))
        # get student & teacher parameters
        params_q = [param_q for name_q, param_q in zip(
            names_q, params_q) if name_q in names_common]
        params_k = [param_k for name_k, param_k in zip(
            names_k, params_k) if name_k in names_common]
        
        # update weight decay and learning rate according to their schedule
        global_it = len(train_loader) * epoch + it  # global training iteration

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[global_it]
            if "resnet" in args.backbone:
                param_group["weight_decay"] = wd_schedule[global_it]
            else:
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[global_it]
        
        # move images to gpu
        images1 = torch.cat([images[:,:,:1024], images[:,:,1024:1024*2]],
                           dim=0)
        # Add zero masking on the teacher branch
        # if args.mask_ratio[0] > 0.0 and args.dataset not in ["FC100", "CIFAR-FS"]:
        #     masks = data[-1]
        #     masks = torch.cat([masks[0], masks[1]],
        #                       dim=0)
        #     masked_images = apply_mask_resnet(
        #         images, masks, args.patch_size)
        # else:
        masked_images = torch.cat([images[:,:,1024:1024*2],images[:,:,1024*2:]],
                           dim=0)
        
        index1= index
        index = torch.cat((index, index), 0)
        
        
        
        # pass images from student/teacher encoders
        p, z_student = student(masked_images)
        z_teacher = teacher(images1)
     
        # concat the features of top-k neighbors for both student &
        #teacher if batch size increase is activated
        if args.enhance_batch:
            z_teacher,zt_index = teacher_nn_replacer.get_top_kNN(
                z_teacher.detach(), index,epoch, args, k=args.topk)
            p,p_index= student_nn_replacer.get_top_kNN(
                p, index,epoch, args, k=args.topk)
            z_student,zs_index = student_f_nn_replacer.get_top_kNN(
                z_student,index, epoch, args, k=args.topk)
            # print(student_nn_replacer.bank)
            # print(p1)
            # print(student_nn_replacer.index_bank)
        # if args.enhance_batch:
        #     z_teacher1,zt_index1 = teacher_nn_replacer.get_top_kNN(
        #         z_teacher.detach(), index,epoch, args, k=args.topk)
        #     p1,p_index1= student_nn_replacer.get_top_kNN(
        #         p, index,epoch, args, k=args.topk)
        #     z_student1,zs_index1 = student_f_nn_replacer.get_top_kNN(
        #         z_student,index, epoch, args, k=args.topk)        
        # elif args.use_nnclr:
        #     z_teacher = teacher_nn_replacer.get_NN(
        #         z_teacher.detach(), epoch, args)
        
        # calculate contrastive loss
        # bsz1=bsz
        # if z_teacher.shape[0]!=bsz:
        #     bsz=int(z_teacher.shape[0]/2)
        # z1, z2 = torch.split(z_teacher, [bsz, bsz], dim=0)
        # z1_s, z2_s = torch.split(z_student, [bsz, bsz], dim=0)
        # p1, p2 = torch.split(p, [bsz, bsz], dim=0)
        
        # ztindex1, ztindex2 = torch.split(zt_index, [bsz, bsz], dim=0)
        # zsindex1, zsindex2 = torch.split(zs_index, [bsz, bsz], dim=0)
        # pindex1, pindex2 = torch.split(p_index, [bsz, bsz], dim=0)
        
        # out = torch.cat([z1_s, z2_s,p1], dim=0)
        # index = torch.cat([zsindex1, zsindex2,pindex1], dim=0)
        
        # if z_student.shape[0]>512:
        #     result,ii=contains_with_order(student_nn_replacer.index_bank, index1)
        #     print("当有时indexs是否包含index1且顺序一致:", result)
        #     print("位置为",ii)
        #     result,ii=contains_with_order(student_nn_replacer.bank, p1)
        #     print("当有时indexs是否包含index1且顺序一致:", result)
        #     print("位置为",ii)            
        #     #print('两个模型的索引是否相同',torch.equal(zsindex1,zsindex2))
            
        # else:
        #     result,ii = contains_with_order(zt_index, index1)
        #     print("当无batch增强时indexs是否包含index1且顺序一致:", result)
        #     print("位置为",ii)
        #     # mask1 = torch.isin(index1, zt_index)
            
            # print('当维度相同时索引是否相同',mask1.all().item())
        if epoch < args.memory_start_epoch:
            if (i + 1) == len(train_loader):
                res = len(train_loader.dataset) % len(train_loader)
                #last_batch_each_gpu = math.ceil(len(index) / len(trloader))
                mask = torch.zeros_like(index1, dtype=torch.bool)
    
                for j in range(len(index1), res, -1):
                    mask[ j - 1] = True
    
                index = index[(~mask)]
                #print(index)
                z_teacher = z_teacher[(~mask).repeat(2)]
                z_student =  z_student[(~mask).repeat(2)]
                p = p[(~mask).repeat(2)]
        
       
    
        # neg_logits, loss_sample_wise, loss_ood= nt_xent(out, t=0.05,
        #                                                           index=index,
        #                                                           sup_weight=0.2,
        #                                                           COLT=True)

        #     z_teacher, p, z_student, args, epoch=epoch,
        #     memory=student_nn_replacer.bank.to(device))
        neg_logits, loss_reshape, loss = loss_fn(z_teacher, p, z_student, zt_index,p_index, 
                                                 zs_index,student_nn_replacer.bank.to(device),
                                                 student_nn_replacer.index_bank.to(device),epoch)
        
        
        #print('neg_logits',neg_logits.shape)
        beta=0.97
        k_largest_logits=10
        #print(z_teacher.shape[0])
        #print(bsz)
        if epoch >= args.memory_start_epoch and args.enhance_batch and not args.use_nnclr:
            neg_logits = neg_logits.detach()
            #print(neg_logits.shape)
            index=zs_index
            
            for count in range(index.shape[0]):
                if not int(index[count]) == -1:
                    new_average = neg_logits[count].sort(descending=True)[0][
                                      :k_largest_logits].sum().clone().detach()
                    shadow[int(index[count])] = new_average  
                    momentum_tail_score[epoch-1, int(index[count])] = new_average
            
            
        else:
            #print(index)
            #print(index.shape)
            neg_logits = neg_logits.mean(dim=0).detach()
            #print( neg_logits.shape)
            for count in range(index.shape[0] //2):
                if not index[count] == -1:
                    if epoch > 1:
                        new_average = (1.0 - beta) * neg_logits[count].sort(descending=True)[0][
                                                                        :k_largest_logits].sum().clone().detach() \
                                      + beta * shadow[index[count]]
                    else:
                        new_average = neg_logits[count].sort(descending=True)[0][
                                      :k_largest_logits].sum().clone().detach()
                    shadow[index[count]] = new_average
                    momentum_tail_score[epoch-1, index[count]] = new_average

        # loss_state = beclr_loss(        
        # loss_dy = loss_state['loss']
        # loss=loss_dy+0.2*loss_ood
        #loss=loss_dy
        
        #print('loss_ood',loss_ood.item())
        #print('loss_all',loss.item())
        # update student weights through backpropagation
        optimizer.zero_grad()
        loss.backward()
        cancel_gradients_last_layer(epoch, student,
                                    args.freeze_last_layer)
        optimizer.step()
        
        
        # update teacher weights through EMA
        with torch.no_grad():
            m = momentum_schedule[global_it]  # momentum parameter
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        
        # logging
        loss_hist.update(loss.item(), bsz)
        #loss_pos_hist.update(loss_state["loss_pos"].item(), bsz)
        #loss_neg_hist.update(loss_state["loss_neg"].item(), bsz)
        #std_hist.update(loss_state["std"].item(), bsz)
        batch_time.update(time.time() - end)
        end = time.time()
        
        if (it + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                      epoch, global_it + 1 - epoch * len(train_loader),
                      len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=loss_hist))
            sys.stdout.flush()
            #break
    
    
    # log weight gradients
    grad_stats = grad_logger(student.named_parameters())
    
    _new_lr = lr_schedule[global_it]
    _new_wd = wd_schedule[global_it]
    
    writer.add_scalar("Loss", loss_hist.avg, epoch)
    writer.add_scalar("Alignment Loss", loss_pos_hist.avg, epoch)
    writer.add_scalar("Uniformity Loss", loss_neg_hist.avg, epoch)
    writer.add_scalar("Standard Deviation", std_hist.avg, epoch)
    writer.add_scalar("Batch Time", batch_time.avg, epoch)
    writer.add_scalar("Data Time", data_time.avg, epoch)
    writer.add_scalar("Learning Rate", _new_lr, epoch)
    writer.add_scalar("Weight Decay", _new_wd, epoch)
    writer.add_scalar("Weight Gradient Average", grad_stats.avg, epoch)
    writer.flush()
    
    
    
    return loss_hist.avg,momentum_tail_score,shadow


def contains_with_order(tensor_a, tensor_b):
    # 如果 tensor_b 的长度比 tensor_a 长度还大，直接返回 False
    if tensor_b.numel() > tensor_a.numel():
        return False
    
    # 滑动窗口查找
    for i in range(tensor_a.numel() - tensor_b.numel() + 1):
        # 在 tensor_a 中提取和 tensor_b 相同长度的子张量
        sub_tensor = tensor_a[i:i + tensor_b.numel()]
        
        # 检查子张量是否和 tensor_b 相等
        if torch.equal(sub_tensor, tensor_b):
            return True, i
    
    return False



root= 'E:\\研究数据\\西储大学\\xichu76lei_380'
#root='D:\\自监督元学习故障诊断+域外特征+数据剪裁\\DATA'


dataset_train = CWRUDataset1(root, mode='train',imb_type='exp', imb_factor=0.1,rand_number=0,num_expert=3)   
    
shadow = torch.zeros(len(dataset_train))    

momentum_tail_score = torch.zeros(args.epochs, len(dataset_train)).to(device)
    
    # build train data loader
data_loader = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True
    )


train_loader_test_trans= torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
print(f"Data loaded: there are {len(dataset_train)} samples.")

# ============ building model ... ============
student, teacher = build_student_teacher(num_classes=109,device='cpu')

# ============ preparing loss ... ============
# beclr_loss = BECLRLoss(args, lamb_neg=args.lamb_neg,
#                        temp=args.temp).to(device)
loss_fn = NTXentLoss(args=args, device=device, memory=True, sup_weight=0.1, COLT=True).to(device)
# ============ preparing memory queue ... ============
memory_size = (args.memory_scale * args.num_clusters //
               (args.batch_size * 2) + 1) * args.batch_size * 2 + 1
print("Memory Size: {} \n".format(memory_size))

teacher_nn_replacer = BA(size=memory_size, origin="teacher")
student_nn_replacer = BA(size=memory_size, origin="student")
student_f_nn_replacer = BA(size=memory_size, origin="student_f")
local_runs = Path(args.save_path) / Path("logs")
print("Log Path: {}".format(local_runs))
print("Checkpoint Save Path: {} \n".format(args.save_path))

# initialize tensorboard logger
writer = SummaryWriter(log_dir=local_runs)

# ============ preparing optimizer ... ============
params_groups = get_params_groups(student)

if args.optimizer == "adamw":
    optimizer = torch.optim.AdamW(params_groups)
elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(
        params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
elif args.optimizer == "lars":
    optimizer = LARS(params_groups)

# for mixed precision training

# ============ init schedulers ... ============
lr_schedule = cosine_scheduler(
    args.lr * (args.batch_size * get_world_size()
               ) / 256.,  # linear scaling rule
    args.min_lr,
    args.epochs, len(data_loader)+int(budget/256)+1,
    warmup_epochs=args.warmup_epochs,
)
wd_schedule = cosine_scheduler(
    args.weight_decay,
    args.weight_decay_end,
    args.epochs, len(data_loader)+int(budget /256)+1,
)
# momentum parameter is increased to 1 with a cosine schedule
momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                     args.epochs, len(data_loader)+int(budget/256)+1)


cls_num_list =dataset_train.get_cls_num_list()
print(cls_num_list)
start_epoch = 1
batch_size = args.batch_size
start_time = time.time()
#%%
#Train_Data = np.load('D:\\自监督长尾故障诊断\\PUshuju\\train_1024.npy')
#Test_Data = np.load('D:\\自监督长尾故障诊断\\PUshuju\\test_1024.npy')
Train_Data = np.load('D:\\自监督长尾故障诊断\\PUSHUJU\\train_1024_250.npy')
Test_Data = np.load('D:\\自监督长尾故障诊断\\PUSHUJU\\test_1024_250.npy')
combined_array = np.concatenate((Train_Data, Test_Data), axis=0)

data_array=combined_array.reshape(combined_array.shape[0]*combined_array.shape[1],combined_array.shape[3])

num_samples = data_array.shape[0]
num_classes = 128
samples_per_class = combined_array.shape[1]

# 计算总标签数量
total_labels = num_classes * samples_per_class

# 生成标签数组
labels_array = np.arange(num_classes)  # 生成 [0, 1, 2, ..., num_classes-1]
labels_array = np.repeat(labels_array, samples_per_class)  # 每个标签重复 samples_per_class 次

# 确保标签数量与 num_samples 匹配
labels_array = np.tile(labels_array, num_samples // total_labels) 
ood_train = Random_PUDataset(data_array, labels_array, cls_num=num_classes, 
                          test_size=0.4, rand_number=0, ood=False)   #INDEX
ood_train_tran = Random_PUDataset(data_array, labels_array, cls_num=num_classes, 
                          test_size=0.4, rand_number=0, ood=True)  #INDEX=-1

ood_loader = torch.utils.data.DataLoader(
                ood_train, 
                num_workers=0,
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=True)

#%%

combined_array= np.load('D:\\无监督元学习_数据裁剪_域外数据\\HIT航空发动机\\HIT_hangkong.npy')
#combined_array = np.concatenate((Train_Data, Test_Data), axis=0)

data_array=combined_array.reshape(combined_array.shape[0]*combined_array.shape[1],combined_array.shape[3])

num_samples = data_array.shape[0]
num_classes = 109
samples_per_class = combined_array.shape[1]

# 计算总标签数量
total_labels = num_classes * samples_per_class

# 生成标签数组
labels_array = np.arange(num_classes)  # 生成 [0, 1, 2, ..., num_classes-1]
labels_array = np.repeat(labels_array, samples_per_class)  # 每个标签重复 samples_per_class 次

# 确保标签数量与 num_samples 匹配
labels_array = np.tile(labels_array, num_samples // total_labels) 
ood_train = Random_PUDataset(data_array, labels_array, cls_num=num_classes, 
                          test_size=0.4, rand_number=0, ood=False)   #INDEX

ood_train_tran = Random_PUDataset(data_array, labels_array, cls_num=num_classes, 
                          test_size=0.4, rand_number=0, ood=True)  #INDEX=-1

ood_loader = torch.utils.data.DataLoader(
                ood_train, 
                num_workers=0,
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=True)

#%%

#start_epoch = 20
# args.epochs=17
# args.memory_start_epoch=5
# args.eval_freq=5
aa=15
test_loader = build_fewshot_loader(args, 'val')
#test_loader =build_fewshot_loader_HIT(args, 'val')
for epoch in tqdm(range(start_epoch, args.epochs)):
    time1 = time.time()
    # data_loader.sampler.set_epoch(epoch)
    # if args.dataset not in ["FC100", "CIFAR-FS"]:
    #     data_loader.dataset.set_epoch(epoch)

    # ============ training one epoch of BECLR ... ============
    if epoch-1>14 and (epoch-1)%aa==0:
            

        #a=validate(teloader, encoder1, flag='val')
        print('开始构建域外池')
        sample_idx,clulabel = sample_batch(train_loader_test_trans, student,ood_loader,budget,momentum_weight, args=None)
        ood_sample_subset = torch.utils.data.Subset(ood_train_tran, sample_idx.tolist())
        
        #ood_sample_subset = dataset_test
        del data_loader
        
        new_train_datasets = torch.utils.data.ConcatDataset([dataset_train, ood_sample_subset])
        data_loader = torch.utils.data.DataLoader(
                    new_train_datasets,
                    num_workers=0,
                    batch_size=args.batch_size,
                    shuffle=True,
                    pin_memory=True)

    
    loss,momentum_tail_score,shadow = train_one_epoch(data_loader, student, teacher, optimizer,
                           epoch, momentum_tail_score,shadow,lr_schedule, wd_schedule,
                           momentum_schedule, writer, loss_fn, args,
                           teacher_nn_replacer, student_nn_replacer,
                           student_f_nn_replacer)
        
    time2 = time.time()
    momentum_weight = momentum_tail_score[epoch-1]

    print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
   

    # ============ Save checkpoint & Memory State ... ============
    if args.save_path is not None and epoch % args.ckpt_freq == 0:

        save_file = os.path.join(
            args.save_path, 'epoch_{}.pth'.format(epoch))
        save_student_teacher(args, student, teacher, epoch, loss,
                             optimizer, batch_size, save_file,
                             teacher_nn_replacer, student_nn_replacer,
                             student_proj_memory=student_f_nn_replacer)

    # evaluate test performance every args.eval_freq epochs during training
    if epoch % args.eval_freq == 0 and epoch > 0:
        student.encoder.masked_im_modeling = False
        results = evaluate_fewshot(args, student.encoder,
                                   test_loader, n_way=args.n_way,
                                   n_shots=[1, 5], n_query=args.n_query,
                                   classifier='LR')
        student.encoder.masked_im_modeling = True
        # log accuracy and confidence intervals
        writer.add_scalar("1-Shot Accuracy", results[0][0], epoch)
        writer.add_scalar("5-Shot Accuracy", results[1][0], epoch)
        writer.add_scalar("1-Shot C95", results[0][1], epoch)
        writer.add_scalar("5-Shot C95", results[1][1], epoch)
    writer.flush()














