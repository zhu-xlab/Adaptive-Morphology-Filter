import random
from struct import calcsize
from sys import path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as tvf
import numpy as np
import os
from scipy.io import loadmat
import cv2
from sklearn.decomposition import PCA, FastICA

class HyperSpecData(Dataset):
    def __init__(self, data_path,  label_path, data_key='indian_pines_corrected',
                 label_key='indian_pines_gt', train_ratio=0.1, pca=3, patch_size=31, is_train=False):
        self.data_key = data_key
        self.is_train = is_train
        self.train_ratio = train_ratio
        self.patch_size = patch_size
        self.mat = np.float32(loadmat(data_path)[data_key])
        self.shape = self.mat.shape
        # matlab format, label start with 1, 0 represents background
        self.label = np.int32(loadmat(label_path)[label_key])
        self.h, self.w, self.c = np.shape(self.mat)
        self.catalog = self.label.max()
        print("dataset:{}, label_num: {}, shape:{}, dtype:{}".format(
            data_key.split('_')[0], self.label.max(), self.shape, self.mat.dtype))

        self.mat = np.reshape(self.mat, [-1, self.c])
        self.mat_pca = PCA(n_components=pca,whiten=True,svd_solver='arpack').fit_transform(self.mat)
        self.mat_pca = np.reshape(self.mat_pca, [self.h, self.w, pca])

        # foreground mask
        np.random.seed(1234)
        self.label_loc = self.get_location(self.label) 
        self.idxs = np.arange(self.label_loc.shape[0])
        np.random.shuffle(self.idxs)

        self.pca_no_pad = self.mat_pca 
        self.label_no_pad = self.label 

        # self.train_idx = self.idxs[:int(self.label_loc.shape[0]*self.train_ratio)]
        # self.test_idx = self.idxs[int(self.label_loc.shape[0]*self.train_ratio):]
        # print(self.test_idx.shape[0])
        self.train_idx, self.test_idx = self.get_train_idx_by_class()

        if self.is_train:
            self.data_idx = self.train_idx
        else:
            self.data_idx = self.test_idx

        self.train_mask,self.test_mask = self.get_mask()

        self.train_mask = self.pad(self.train_mask, self.patch_size//2) 
        # self.test_mask = self.pad(self.test_mask, self.patch_size//2)
        self.label = self.pad(self.label, self.patch_size//2)
        self.mat_pca = self.pad(self.mat_pca, self.patch_size//2)

        self.train_mask = torch.from_numpy(self.train_mask).long()
        # self.test_mask = torch.from_numpy(self.test_mask).long()
        self.label = torch.from_numpy(self.label).long()
        self.mat_pca = torch.from_numpy(self.mat_pca.copy()).permute(2, 0, 1).float()

    def get_train_idx_by_class(self):
        idxs = self.idxs.copy()
        train_idx = []
        test_idx = []
        if 'indian' in self.data_key:
            class_count = [5,142,83,24,48,73,5,46,5,94,245,59,21,127,39,9]
        if 'salinas' in self.data_key:
            class_count = [200,372,198,140,268,395,358,1127,620,328,107,193,92,107,727,180]
        if 'pavia' in self.data_key:
            class_count = [663,1865,210,306,133,503,133,368,94]
        for c in range(self.catalog):
            count = 0
            for i in range(idxs.shape[0]):
                if count < class_count[c]:
                    if self.label_no_pad[self.label_loc[idxs[i],0],self.label_loc[idxs[i],1]] == c+1:
                        count = count + 1
                        train_idx.append(idxs[i])
                else:
                    break

        for i in range(idxs.shape[0]):
            if idxs[i] not in train_idx:
                test_idx.append(idxs[i])
        return np.array(train_idx), np.array(test_idx)

    def get_mask(self):
        train_mask = np.zeros([self.shape[0], self.shape[1]])
        test_mask = np.zeros([self.shape[0], self.shape[1]])
        for i in range(self.train_idx.shape[0]):
            h,w = self.label_loc[self.train_idx[i]][0], self.label_loc[self.train_idx[i]][1]
            train_mask[h,w] = 1
        for i in range(self.test_idx.shape[0]):
            h,w = self.label_loc[self.test_idx[i]][0], self.label_loc[self.test_idx[i]][1]
            test_mask[h,w] = 1
        return train_mask, test_mask

    def get_location(self, label):
        h_idx,w_idx = np.nonzero(label)
        idx = np.concatenate([h_idx[:,None], w_idx[:,None]], 1)
        return idx

    @staticmethod
    def pad(data, pad):
        if len(data.shape) == 3:
            data = np.pad(data, ((pad, pad), (pad, pad), (0, 0)))
        if len(data.shape) == 2:
            data = np.pad(data, ((pad, pad), (pad, pad)))
        return data

    def get_eval(self):
        data = torch.from_numpy(self.pca_no_pad.copy()).unsqueeze(0).permute(0, 3, 1, 2).float()
        label = torch.from_numpy(self.label_no_pad.copy()).unsqueeze(0).long()-1
        mask = torch.from_numpy(self.test_mask.copy()).unsqueeze(0).long()
        return data, label, mask

    def pair_aug(self, i, j):
        h = self.patch_size
        w = self.patch_size
        # i = i+self.patch_size//2
        # j = j+self.patch_size//2
        #top left height width
        cropped_mask = tvf.crop(self.train_mask.unsqueeze(0), i, j, h, w)
        cropped_label = tvf.crop(self.label.unsqueeze(0), i, j, h, w)
        cropped_pca = tvf.crop(self.mat_pca, i, j, h, w)

        if self.is_train:
            #rotation(0, 90, 180,270 degrees)
            for rotation_num in range(4):
                cropped_pca = tvf.rotate(cropped_pca, angle=90*rotation_num)
                cropped_label = tvf.rotate(cropped_label, angle=90*rotation_num)
                cropped_mask = tvf.rotate(cropped_mask, angle=90*rotation_num)

            # Either horizontal inversion or vertical inversion
            # Invert(horizontal direction)
            for h_flip_num in range(2):
                cropped_mask = transforms.RandomHorizontalFlip(
                    p=h_flip_num)(cropped_mask)
                cropped_pca = transforms.RandomHorizontalFlip(
                    p=h_flip_num)(cropped_pca)
                cropped_label = transforms.RandomHorizontalFlip(
                    p=h_flip_num)(cropped_label)

            for v_flip_num in range(2):
                cropped_mask = transforms.RandomVerticalFlip(
                    p=v_flip_num)(cropped_mask)
                cropped_pca = transforms.RandomVerticalFlip(
                    p=v_flip_num)(cropped_pca)
                cropped_label = transforms.RandomVerticalFlip(
                    p=v_flip_num)(cropped_label)
        
        return cropped_pca, cropped_label.squeeze(0)-1,  cropped_mask.squeeze(0)

    def __getitem__(self, index):
        i = np.random.randint(0, self.label.shape[0])
        j = np.random.randint(0, self.label.shape[1])
        data, label, data_pca = self.pair_aug(i,j)
        return data, label, data_pca

    def __len__(self):
        return 64*5000


if __name__ == '__main__':
    data_path = './Indian_pines_corrected.mat'
    label_path = './Indian_pines_gt.mat'
    data_key = 'indian_pines_corrected'
    label_key = 'indian_pines_gt'

    train_data = HyperSpecData(data_path, label_path, data_key, label_key, pca=3, is_train=True)
    dataloader = DataLoader(train_data, batch_size=1, shuffle=False)
    for i, (data, label, pca) in enumerate(dataloader):
        print(i, data.shape, label.shape, pca.shape)


