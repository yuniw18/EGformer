import os
from torch.utils import data
from torchvision import transforms
import math
from PIL import ImageEnhance
import random
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.utils.data import RandomSampler
import torchvision.transforms.functional as F
from imageio import imread
import numpy as np
from skimage import io
import math
import os.path as osp
import torch.utils.data

class S3D_loader(data.Dataset):
    def __init__(self,root,transform = None,transform_t = None):
            "makes directory list which lies in the root directory"
            if True:
                dir_path = list(map(lambda x:os.path.join(root,x),os.listdir(root)))
                dir_path_deep=[]
                left_path=[]
                right_path=[]
                self.image_paths = []
                self.depth_paths = []
                self.semantic_paths = []
                dir_sub_dir=[]

                for dir_sub in dir_path:
                    
                    sub_path = os.path.join(dir_sub,'2D_rendering')
                    sub_path_list = os.listdir(sub_path)
                    

                    for path in sub_path_list:
                        dir_sub_dir.append(os.path.join(sub_path,path,'panorama/full'))
                

                for final_path in dir_sub_dir:
                    self.image_paths.append(os.path.join(final_path,'rgb_rawlight.png'))                    
                    self.depth_paths.append(os.path.join(final_path,'depth.png'))                    
                    # self.semantic_paths.append(os.path.join(final_path,'semantic.png'))                    
                    
                self.transform = transform
                self.transform_t = transform_t


    def __getitem__(self,index):
           
        if True:
            
            image_path = self.image_paths[index]
            depth_path = self.depth_paths[index]
                
            image = Image.open(image_path).convert('RGB')
            depth = io.imread(depth_path,as_gray=True).astype(np.float)
            mask = (depth>0.)

            data=[]

        if self.transform is not None:
            data = {'color':self.transform(image),'depth':self.transform_t(depth)/2048., 'mask':self.transform_t(mask)}
            
        return data

    def __len__(self):
        
        return len(self.image_paths)


