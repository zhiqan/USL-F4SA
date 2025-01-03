# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 05:13:04 2024

@author: Owner
"""
import os
import numpy as np
import pandas as pd
import csv
import torch
from torch.utils.data import Dataset
import random
from sklearn.model_selection import train_test_split
import torch.distributions as distributions

class Random_PUDataset(Dataset):
    def __init__(self, data_array, labels_array, cls_num=109, test_size=0.5, rand_number=0,ood=False):
    
        self.cls_num = cls_num
        self.data = data_array
        self.targets = labels_array.squeeze().astype(np.int64)  # 将标签数组扁平化为一维
        self.new_labels = []
        self.ood=ood

        # 划分训练集和测试集
        indices = np.arange(len(self.data))
        train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=rand_number, stratify=self.targets)
        self.data = self.data[train_indices]
        self.targets = self.targets[train_indices]
        print(len(self.targets))

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def add_laplace_noise(self, x, u=0, b=0.2):
        laplace_noise = np.random.laplace(u, b, x.shape).reshape(x.shape)
        return laplace_noise + x

    def add_wgn(self, x, snr=15):
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(x ** 2) / len(x)
        npower = xpower / snr
        return x + (np.random.randn(*x.shape) * np.sqrt(npower))

    def Amplitude_scale(self, x, snr=0.05):
        return x * (1 - snr)

    def Translation(self, x, p=0.5):
        a = len(x)
        return np.concatenate((x[int(a*p):], x[:int(a*p)]), axis=0)
    def mask_noise(self,x, p= 0.5, u= 0.1, temperature = 1.):
    # 生成是否应用增强的样本概率
        p_sample = p
    
        # 生成掩码矩阵
        u = torch.full((1024,), u)  # 创建一个长度为1024，值全为u的张量
        
        u_dist = distributions.RelaxedBernoulli(temperature, u)
        u_sample = u_dist.rsample()
        Mask = u_sample.le(0.5).float()
        Mask = Mask - u_sample.detach() + u_sample
    
        
        aug_x = Mask * x
        x = p_sample * aug_x + (1 - p_sample) * x
        
        return x

    def __getitem__(self, index):
        label_ = self.targets[index]
        pic = self.data[index]
        if self.ood:
            ccc = ['self.Amplitude_scale(pic)','self.Amplitude_scale(pic)', 'self.Translation(pic)', 'self.add_wgn(pic)', 'self.add_laplace_noise(pic)']
            n1 = np.random.choice(ccc, 3, replace=False)
            aa = pic.T
            bb = eval(n1[1]).T
            cc=eval(n1[2]).T
            aa2 = np.concatenate((aa, bb, cc), axis=-1)
            aa2=aa2[np.newaxis, :] 

            return torch.tensor(aa2, dtype=torch.float).detach(), -1
        else:
            ccc = ['self.mask_noise(pic)', 'self.Translation(pic)', 'self.add_wgn(pic)', 'self.add_laplace_noise(pic)']
            n1 = np.random.choice(ccc, 3, replace=False)
            aa = pic.T
            bb = eval(n1[1]).T
            cc=eval(n1[2]).T
            aa2 = np.concatenate((aa, bb, cc), axis=-1)
            aa2=aa2[np.newaxis, :] 

            return torch.tensor(aa2, dtype=torch.float).detach(),  index
    def __len__(self):
        return len(self.targets)
    
    




class CWRUDataset(Dataset):
    #cls_num =109 # 根据实际情况设置类别数
    

    def __init__(self, root, mode='train', simclr=False):
        self.simclr = simclr
        self.path = os.path.join(root, 'xichu_细粒度分类')  # 数据文件夹路径
        self.mode = mode
        csvdata = self.loadCSV(os.path.join(root, 'ALL.csv'))  # 加载CSV文件
        self.data = []
        self.img2label = {}
        self.targets = []
        self.new_labels = []
        self.cls_num=cls_num

        all_data = []
        all_labels = []

        for i, (k, v) in enumerate(csvdata.items()):
            all_data.extend([(item, i) for item in v])  # [(filename1, label1), (filename2, label1), ...]
        
        train_data, test_data = train_test_split(all_data, test_size=test_size, random_state=rand_number, stratify=[label for _, label in all_data])
        print(len(train_data))
        
        if mode == 'train':
            selected_data = train_data
        else:
            selected_data = test_data

        for filename, label in selected_data:
            self.data.append(filename)
            self.targets.append(label)
            if label not in self.img2label:
                self.img2label[label] = len(self.img2label)

        # if mode == 'train':
        #     np.random.seed(rand_number)
        #     img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        #     self.gen_imbalanced_data(img_num_list, num_expert)

        # self.transform = transform
        # self.target_transform = target_transform

    # def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
    #     img_max = len(self.targets) / cls_num
    #     img_num_per_cls = []
    #     if imb_type == 'exp':
    #         for cls_idx in range(cls_num):
    #             num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
    #             img_num_per_cls.append(int(num))
    #     elif imb_type == 'step':
    #         for cls_idx in range(cls_num // 2):
    #             img_num_per_cls.append(int(img_max))
    #         for cls_idx in range(int(cls_num -cls_num // 2)):
    #             img_num_per_cls.append(int(img_max * imb_factor))
    #     else:
    #         img_num_per_cls.extend([int(img_max)] * cls_num)
    #     return img_num_per_cls

    # def gen_imbalanced_data(self, img_num_per_cls, num_expert):
    #     new_data = []
    #     new_targets = []
    #     targets_np = np.array(self.targets, dtype=np.int64)
    #     classes = np.unique(targets_np)
    #     self.num_per_cls_dict = dict()
    #     for the_class, the_img_num in zip(classes, img_num_per_cls):
    #         self.num_per_cls_dict[the_class] = the_img_num
    #         idx = np.where(targets_np == the_class)[0]
    #         np.random.shuffle(idx)
    #         selec_idx = idx[:the_img_num]
    #         new_data.extend([self.data[i] for i in selec_idx])
    #         new_targets.extend([the_class] * the_img_num)
    #     self.data = new_data
    #     self.targets = new_targets

    # def get_cls_num_list(self):
    #     cls_num_list = []
    #     for i in range(self.cls_num):
    #         cls_num_list.append(self.num_per_cls_dict[i])
    #     return cls_num_list

    def RCNN(self, X_n):
        N, C, W = X_n.size()
        p = np.random.rand()
        K = [1, 3, 5, 7, 11, 15, 17]
        if p > 0.5:
            k = K[np.random.randint(0, len(K))]
            Conv = torch.nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=k//2, bias=False)
            torch.nn.init.xavier_normal_(Conv.weight)
            X_n = Conv(X_n.reshape(-1, C, W)).reshape(N, C, W)
        return X_n.reshape(C, W).detach()

    def add_laplace_noise(self, x, u=0, b=0.2):
        laplace_noise = np.random.laplace(u, b, len(x)).reshape(1024, 1)  # 为原始数据添加μ为0，b为0.1的噪声
        return laplace_noise + x

    def add_wgn(self, x, snr=15):
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(x ** 2) / len(x)
        npower = xpower / snr
        return x + (np.random.randn(len(x)) * np.sqrt(npower)).reshape(1024, 1)

    def Amplitude_scale(self, x, snr=0.05):
        return x * (1 - snr)

    def Translation(self, x, p=0.5):
        a = len(x)
        return np.concatenate((x[int(a*p):], x[0:int(a*p)]), axis=0)

    def loadCSV(self, csvf):
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def __getitem__(self, index):
        if self.mode == 'train':
            label_ = self.targets[index]
            pic = pd.read_csv(os.path.join(self.path, self.data[index]), header=None)
            pic = pic.values

            if self.simclr:
                ccc = ['self.Amplitude_scale(pic)', 'self.Translation(pic)', 'self.add_wgn(pic)', 'self.add_laplace_noise(pic)']
                n1 = np.random.choice(ccc, 3, replace=False)
                aa = pic.T
                bb = eval(n1[1]).T
                cc=eval(n1[2]).T
                aa2=np.concatenate((aa,bb,cc), axis=-1)

                return torch.tensor(aa2, dtype=torch.float).detach(), label_
            else:
                pic3 = torch.tensor(pic.T, dtype=torch.float)
                return pic3, label_
        else:
            label_ = self.targets[index]
            pic = pd.read_csv(os.path.join(self.path, self.data[index]), header=None)
            pic = pic.values
            ccc = ['self.Amplitude_scale(pic)', 'self.Translation(pic)', 'self.add_wgn(pic)', 'self.add_laplace_noise(pic)']
            n1 = np.random.choice(ccc, 3, replace=False)
            aa = pic.T
            bb = eval(n1[1]).T
            cc=eval(n1[2]).T
            aa2=np.concatenate((aa,bb,cc), axis=-1)
            #pic3 = torch.tensor(pic.T, dtype=torch.float)
            return torch.tensor(aa2, dtype=torch.float).detach(), label_

    def __len__(self):
        return len(self.targets)

class CWRUDataset1(Dataset):
    cls_num =90
    def __init__(self, root, mode='train',imb_type='exp', imb_factor=0.5,rand_number=0,num_expert=3):
        #self.simclr = simclr
        self.path = os.path.join(root, 'xichu_细粒度分类')  # 数据文件夹路径
        self.mode = mode
        csvdata = self.loadCSV(os.path.join(root, mode+'.csv'))  # 加载CSV文件
        self.data = []
        self.img2label = {}
        self.targets = []
        self.new_labels = []

        all_data = []
        all_labels = []

        for i, (k, v) in enumerate(csvdata.items()):
            all_data.extend([(item, i) for item in v])  # [(filename1, label1), (filename2, label1), ...]
        
        
        selected_data = all_data


        for filename, label in selected_data:
            self.data.append(filename)
            self.targets.append(label)
            if label not in self.img2label:
                self.img2label[label] = len(self.img2label)

        if self.mode == 'train':
            
            np.random.seed(rand_number)
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list, num_expert)

        # self.transform = transform
        # self.target_transform = target_transform

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.targets) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(int(cls_num -cls_num // 2)):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls, num_expert):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.extend([self.data[i] for i in selec_idx])
            new_targets.extend([the_class] * the_img_num)
        self.data = new_data
        self.targets = new_targets


    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def RCNN(self, X_n):
        N, C, W = X_n.size()
        p = np.random.rand()
        K = [1, 3, 5, 7, 11, 15, 17]
        if p > 0.5:
            k = K[np.random.randint(0, len(K))]
            Conv = torch.nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=k//2, bias=False)
            torch.nn.init.xavier_normal_(Conv.weight)
            X_n = Conv(X_n.reshape(-1, C, W)).reshape(N, C, W)
        return X_n.reshape(C, W).detach()

    def add_laplace_noise(self, x, u=0, b=0.2):
        laplace_noise = np.random.laplace(u, b, len(x)).reshape(1024, 1)  # 为原始数据添加μ为0，b为0.1的噪声
        return laplace_noise + x

    def add_wgn(self, x, snr=15):
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(x ** 2) / len(x)
        npower = xpower / snr
        return x + (np.random.randn(len(x)) * np.sqrt(npower)).reshape(1024, 1)

    def Amplitude_scale(self, x, snr=0.05):
        return x * (1 - snr)

    def Translation(self, x, p=0.5):
        a = len(x)
        return np.concatenate((x[int(a*p):], x[0:int(a*p)]), axis=0)
    def mask_noise(self,x, p= 0.5, u= 0.1, temperature = 1.):
    # 生成是否应用增强的样本概率
        p_sample = p
        mask_shape = x.shape  # 获取x的形状
        mask = np.random.binomial(1, u, size=mask_shape).astype(np.float32)
        
        # 应用温度参数，将二项分布转换为近似伯努利分布
        mask = np.where(mask < (1 - u) / temperature, 1, 0)
        
        # 应用掩码
        aug_x = mask * x
        
        # 根据p_sample决定是否应用增强
        x = p_sample * aug_x + (1 - p_sample) * x
        
        return x
    
        # u = torch.full((1024, 1), u)  # 创建一个形状为(1024, 1)，值全为u的张量
    
        # u_dist = distributions.RelaxedBernoulli(temperature, u)
        # u_sample = u_dist.rsample()
        # Mask = u_sample.le(0.5).float()
        
        # # 由于Mask和u_sample的维度已经是(1024, 1)，不需要额外的操作
        # # Mask = Mask - u_sample.detach() + u_sample  # 这一行是不必要的，可以删除
        
        # aug_x = Mask * x
        # x = p_sample * aug_x + (1 - p_sample) * x
        # return x
    
    # def mask_noise(self,x,probability=0.2):
    #     x=torch.from_numpy(x)
    #     masked_data = x.clone()
        
    #     # 生成一个同样大小的随机布尔张量，True表示将对应的数据点设置为0.5
    #     random_numbers = torch.rand_like(masked_data)
    
    # # 设定阈值，小于阈值的数据点将被设置为0.5
    #     threshold = probability
    
    # # 应用阈值，将随机选择的数据点设置为0.5
    #     masked_data[random_numbers < threshold] = 0.5
    
    #     return masked_data

    def loadCSV(self, csvf):
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def __getitem__(self, index):
        self.label_ = self.targets[index]
        pic = pd.read_csv(os.path.join(self.path, self.data[index]), header=None)
        pic = pic.values
        if self.mode=='train':
            ccc = ['self.mask_noise(pic)', 'self.Translation(pic)', 'self.add_wgn(pic)', 'self.add_laplace_noise(pic)']
            n1 = np.random.choice(ccc, 3, replace=False)
            aa = pic.T
            #print(aa.shape)
            bb = self.mask_noise(pic).T
            #print(bb.shape)
            cc=eval(n1[2]).T
            aa2=np.concatenate((aa,bb,cc), axis=-1)

            return torch.tensor(aa2, dtype=torch.float).detach(), self.label_
        else:
            pic3 = torch.tensor(pic.T, dtype=torch.float)
            return pic3, self.label_

    def __len__(self):
        return len(self.targets)