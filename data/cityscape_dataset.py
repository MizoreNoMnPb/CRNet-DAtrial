import numpy as np
import os
import cv2
import torch
import random
from tqdm import tqdm
from os.path import join as opj
from multiprocessing.dummy import Pool
from data.base_dataset import BaseDataset

# Cityscapes dataset
class CityspaceDataset(BaseDataset):
    def __init__(self, opt, split='train', dataset_name='Cityscapes'):
        super(CityspaceDataset, self).__init__(opt, split, dataset_name)
        
        self.batch_size = opt.batch_size
        self.patch_size = opt.patch_size
        self.beta = opt.beta
    
        if split == 'train':
            self._getitem = self._getitem_train
            self.names, self.meta_dirs, self.raw_dirs, self.gt_dirs = self._get_image_dir(self.root, split, name='Train')
            self.len_data = 5000 * self.batch_size

        elif split == 'test':
            self._getitem = self._getitem_test
            self.names, self.meta_dirs, self.raw_dirs, self.gt_dirs = self._get_image_dir(self.root, split, name='Test')
            self.len_data = len(self.names)

        else:
            raise ValueError('split must be train or test')
        
        self.meta_data = [0] * len(self.names)
        self.raw_images = [0] * len(self.names)
        self.gt_images = [0] * len(self.names)
        read_images(self)
        
    def __getitem__(self, index):s
        return self._getitem(index)

    def __len__(self):
        return self.len_data
    
    def _get_image_names(self, dataroot, split=None, name=None):
        image_names = []
        gt_names = []
        
        return image_names, gt_names
    
    def _getitem_train(self, index):
        index = index % len(self.names)
        
        raws = torch.from_numpy(np.float32(np.array(self.raw_images[index]))) / (2**10 - 1)
        gt = torch.from_numpy(np.float32(self.gt_images[index]))
        
        raws, gt = self._crop_patch(raws, gt, self.patch_size)
        
        return {'gt': gt, # [4, H, W]
                'raws': raws, # [T=5, 4, H, W]
                'fname': self.names[index]} 
        
    def _getitem_test(self, index):
        index = index % len(self.image_names)
    
    def _crop_patch(self, raws, gt, p):
        pass

    def _process_metadata(self, metadata):
        metadata_item = metadata.item()
        meta = {}
        for key in metadata_item:
            meta[key] = torch.from_numpy(metadata_item[key])
        return meta


def iter_obj(num, objs):
    for i in range(num):
        yield (i, objs)
        

def imreader(arg):
    i, obj = arg
    for _ in range(3):
        try:
            imgs = []
            for m in range(obj.beta):
                imgs.append(np.load(obj.raw_dirs[i][m]), allow_pickle=True).transpose(2, 0, 1)
            obj.raw_image[i] = imgs
            if obj.split == 'train':
                obj.gt_images[i] = np.load(obj.gt_dirs[i], allow_pickle=True).transpose(2, 0, 1)
            obj.meta_data[i] = np.load(obj.meta_dirs[i], allow_pickle=True)
            failed = False
        except:
            failed = True
    if failed: print('%s fails!' % obj.names[i])
    

def read_images(obj):
    print('Loading images via multiple image readers')
    pool = Pool()
    for _ in tqdm(pool.imap(imreader, iter_obj(len(obj.names), obj)), total=len(obj.names)):
        pass
    pool.close()
    pool.join()


if __name__ == "__main__":
    pass