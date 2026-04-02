import numpy as np
import os
import cv2
import torch
import random
from PIL import Image
from tqdm import tqdm
from os.path import join as opj
from multiprocessing.dummy import Pool
from data.base_dataset import BaseDataset

# Cityscapes dataset
class CityscapeDataset(BaseDataset):
    def __init__(self, opt, split='train', dataset_name='Cityscapes'):
        super(CityscapeDataset, self).__init__(opt, split, dataset_name)
        
        self.batch_size = opt.batch_size
        self.patch_size = opt.patch_size
        self.beta = opt.beta
    
        if split == 'train':
            self._getitem = self._getitem_train
            self.names, self.train_dirs, self.gt_dirs = self._get_image_dir(self.root, split, name='JPEGImages')
            self.len_data = len(self.names) * self.batch_size

        elif split == 'test':
            self._getitem = self._getitem_test
            self.names, self.train_dirs, self.gt_dirs = self._get_image_dir(self.root, split, name='JPEGImages')
            self.len_data = len(self.names)

        else:
            raise ValueError('split must be train or test')
        
        self.meta_data = [0] * len(self.names)
        self.train_images = [0] * len(self.names)
        self.gt_images = [0] * len(self.names)
        
    def __getitem__(self, index):
        return self._getitem(index)

    def __len__(self):
        return self.len_data
    
    def _get_image_dir(self, dataroot, split=None, name=None):
        image_names = []
        train_dirs = []
        gt_dirs = []
        
        all_files = [f for f in os.listdir(opj(dataroot, name)) if f.endswith('.png')]
        all_files.sort()
        train_map = {}
        
        for f in all_files:
            if '_foggy_' not in f:
                scene = f.split('_leftImg8bit')[0]
                image_names.append(scene)
                train_map[scene] = []
                if split == 'train':
                    gt_dirs.append(opj(dataroot, name, f))
            else:
                train_map[scene].append(opj(dataroot, name, f))
        
        for scene in image_names:
            train_dirs.append(train_map[scene])
                
        return image_names, train_dirs, gt_dirs
    
    def _getitem_train(self, index):
        index = index % len(self.names)
        
        gt_images = load_image(self.gt_dirs[index])
        train_images = []
        for m in range(self.beta):
            train_images.append(load_image(self.train_dirs[index][m]))
            
        train_images = torch.from_numpy(np.float32(np.array(train_images))) / 255.0
        gt_images = torch.from_numpy(np.float32(gt_images))

        # train_images, gt_images = self._crop_patch(train_images, gt_images, self.patch_size)
        # the nessary of cropping is not clear for the Cityscapes dataset, so we do not crop the images for now.
        
        return {'gt_images': gt_images,
                'train_images': train_images,
                'file_name': self.names[index]} 
        
    def _getitem_test(self, index):
        """ This task is a Source-Free task, so the dataset only has one directory for image, so the way to test is not given now."""
        pass   
    
    def _crop_patch(self, train_imgs, gt, p):
        ih, iw = train_imgs.shape[-2:]
        ph = random.randrange(10, ih - p + 1 - 10)
        pw = random.randrange(10, iw - p + 1 - 10)
        return train_imgs[..., ph:ph + p, pw:pw + p], \
            gt[..., ph:ph + p, pw:pw + p]


def load_image(path):
    try: 
        with Image.open(path) as image:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_np = np.array(image).astype(np.float32).transpose(2, 0, 1)
        return image_np
    except Exception as e:
        print(f'Loading image {path}: {e}')
        return None
   

# def iter_obj(num, objs):
#     for i in range(num):
#         yield (i, objs)
        

# def imreader(arg):
#     i, obj = arg
#     for _ in range(3):
#         try:
#             images = []
#             for m in range(obj.beta):
#                 with Image.open(obj.train_dirs[i][m]) as image:
#                     if image.mode != 'RGB':
#                         image = image.convert('RGB')
#                     image_np = np.array(image).astype(np.float32).transpose(2, 0, 1) / 255.0
#                 images.append(image_np)
#             obj.train_images[i] = images
#             if obj.split == 'train':
#                 with Image.open(obj.gt_dirs[i]) as image:
#                     if image.mode != 'RGB':
#                         image = image.convert('RGB')
#                     gt_np = np.array(image).astype(np.float32).transpose(2, 0, 1)
#             obj.gt_images[i] = gt_np
#             break
#         except Exception as e:
#             print(f'Loading image {obj.train_dirs[i][m]}: {e}')
    

# def read_images(obj):
#     print('Loading images via multiple image readers')
#     pool = Pool()
#     for _ in tqdm(pool.imap(imreader, iter_obj(len(obj.names), obj)), total=len(obj.names)):
#         pass
#     pool.close()
#     pool.join()


if __name__ == "__main__":
    pass