import math
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import torch.nn as nn
import scipy.misc
import matplotlib.pyplot as plt
import cv2
from torch import optim
from torch.autograd import Variable
import Imath
import array
import matplotlib as mpl
import matplotlib.cm as cm
import argparse
import random
from imageio import imread
import skimage
import skimage.transform
from models.egformer import EGDepthModel
from torch.nn.parallel import DistributedDataParallel as DDP

from models.Panoformer.model import Panoformer as PanoBiT
from scipy.ndimage.filters import maximum_filter
import time
from fvcore.nn import FlopCountAnalysis
from dataset import visualize_a_data
from misc import post_proc
from shapely.geometry import Polygon
import sys

# Layout estimation part in this code is adopted from 'HorizonNet'
# HorizonNet (https://github.com/sunset1995/HorizonNet)

def find_N_peaks(signal, r=29, min_v=0.05, N=None):
    max_v = maximum_filter(signal, size=r, mode='wrap')
    pk_loc = np.where(max_v == signal)[0]

    pk_loc = pk_loc[signal[pk_loc] > min_v]
    if N is not None:
        order = np.argsort(-signal[pk_loc])
        pk_loc = pk_loc[order[:N]]
        pk_loc = pk_loc[np.argsort(pk_loc)]
    return pk_loc, signal[pk_loc]



class Train(object):
    def __init__(self,config, train_loader, gpu, train_sampler,Val_loader):
        self.posenet = None
        self.checkpoint_path = config.checkpoint_path
        self.eval_data_path = config.val_path
        self.model_name = config.model_name
        self.model_path = os.path.join(config.model_name,config.model_path)
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.lr = config.lr 
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.sample_path = os.path.join(self.model_name,config.sample_path)
        self.log_path = os.path.join(self.model_name,'log.txt')
        self.eval_path = os.path.join(self.model_name, config.eval_path)
        self.train_loader = train_loader
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.config = config
        self.backbone = self.config.backbone
        self.depthnet = None
        self.val_loader = Val_loader 
        self.checkpoint_path = config.checkpoint_path
        
        self.crop_ratio = config.eval_crop_ratio
        self.models = {}


        if not os.path.exists(self.model_name):
            os.mkdir(self.model_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if not os.path.exists(self.sample_path):
            os.mkdir(self.sample_path)
        if not os.path.exists(self.eval_path):
            os.mkdir(self.eval_path)

        # DDP settings
        self.gpu = gpu
        self.distributed = config.distributed
        self.train_sampler = train_sampler

        self.build_model()

    def build_model(self):
            

        if self.backbone == 'EGformer':
            from models.egformer import EGDepthModel
            self.depthnet = EGDepthModel(hybrid=False)
 
        elif self.backbone == 'Panoformer':
            self.depthnet = PanoBiT()

        else:
            print("Error")

            
 
        self.g_optimizer = optim.AdamW([{"params": list(self.depthnet.parameters())}],
                                        self.lr,[self.beta1,self.beta2])


        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer, 0.95)

        if not torch.cuda.is_available():
            print(f'Using CPU')
        elif self.distributed:  # Using multi-GPUs
            if self.gpu is not None:
                torch.cuda.set_device(self.gpu)
                self.depthnet.cuda(self.gpu)
                self.depthnet = torch.nn.parallel.DistributedDataParallel(self.depthnet, device_ids=[self.gpu], find_unused_parameters=True)
            else:
                self.depthnet.cuda()
                self.depthnet = torch.nn.parallel.DistributedDataParallel(self.depthnet, find_unused_parameters=True)

        elif self.gpu is not None:  # Not using multi-GPUs
            torch.cuda.set_device(self.gpu)
            self.depthnet = self.depthnet.cuda(self.gpu)
    def is_main_process(self):
        return torch.distributed.get_rank()==0


    def feed_forward(net, x, y_bon, y_cor):
        x = x
        losses = {}

        with autocast():
            y_bon_, y_cor_ = net(x)
            losses['bon'] = F.l1_loss(y_bon_, y_bon)
            losses['cor'] = F.binary_cross_entropy_with_logits(y_cor_, y_cor)
    
        losses['total'] = losses['bon'] + losses['cor']

        return losses


    def to_variable(self,x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def transform(self,input):
        transform = transforms.Compose([
                    transforms.ToTensor()])
        return transform(input)
 
    def reset_grad(self):
        self.depthnet.zero_grad()
        

    def resize(self,input,scale):
        input = nn.functional.interpolate(
                input, scale_factor=scale, mode="bilinear", align_corners=True)   
        return input

    def train(self):
        if os.path.isfile(self.log_path):
            os.remove(self.log_path)  
 
        if self.config.Continue:
            '''
            Load checkpoint of model
            '''
            depthnet_dict = self.depthnet.state_dict()
            pretrained_dict = torch.load((self.config.checkpoint_path), map_location=torch.device("cpu"))

            if self.config.load_convonly:  # This part is not fully verified / there may exist unexpected behaviors.
                print("Loading Convonlution layer weights only")
                for key in list(pretrained_dict.keys()):
                    if self.backbone == 'EGformer':
                        pretrained_dict[key.replace('stage','Tstage')] = pretrained_dict.pop(key) # Make transformer block not to be loaded                   
                    elif self.backbone == 'Panoformer':
                        pretrained_dict[key.replace('encoder','Tencoder').replace('decoder','Tdecoder')] = pretrained_dict.pop(key) # Make transformer block not to be loaded 

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in depthnet_dict}
            print("Loading pre-trained modules below")

            depthnet_dict.update(pretrained_dict)
            self.depthnet.load_state_dict(depthnet_dict,strict=True)

            print('checkpoint weights loaded')

        max_batch_num = len(self.train_loader) - 1

############################# Initial evaluation ##########################
        with torch.no_grad(): 
            eval_name = 'Sample_%d' %(0)
            if self.is_main_process():    
                self.sample(self.eval_data_path,eval_name,self.crop_ratio)
########################################################################

        for epoch in range(self.num_epochs):
            if self.distributed:
                self.train_sampler.set_epoch(epoch)

            for batch_num, data in enumerate(self.train_loader):
                if True:
                    inputs,y_bon,y_cor = data
                    inputs = self.to_variable(inputs)
                    y_bon = self.to_variable(y_bon)
                    y_cor = self.to_variable(y_cor)

                    if self.backbone == 'EGformer':
                        y_bon_predict, y_cor_predict = self.depthnet(inputs)

                    elif self.backbone == 'Panoformer':
                        y_bon_predict, y_cor_predict = self.depthnet(inputs)
                    
                    loss = F.l1_loss(y_bon_predict, y_bon)
                    loss += F.binary_cross_entropy_with_logits(y_cor_predict, y_cor)
    
                    try: 
                        loss.backward()
                        self.g_optimizer.step()
                    
                    except:
                        print('skip_backward')
                        self.g_optimizer.zero_grad()                           

                self.reset_grad()                   
                if (batch_num) % self.log_step == 0:
                    if torch.distributed.get_rank() == 0:
                        try:
                            print('Epoch [%d/%d], Step[%d/%d], loss: %.5f' 
                              %(epoch, self.num_epochs, batch_num, max_batch_num, 
                                loss.item()))
                        except:
                            print("Skip logging") # 

                if (batch_num) % self.sample_step == 0:
                    if torch.distributed.get_rank() == 0:
                        eval_name = 'Sample_%d' %(epoch)
                        with torch.no_grad():
                            try:
                                self.sample(self.eval_data_path,eval_name,self.crop_ratio)
   
                            except:
                                print("layout estimation error")
            if torch.distributed.get_rank() not in [-1,0]:
                torch.distributed.barrier()
            ### Make sure only rank 0 save the model / validation
            # Refer to https://github.com/pytorch/pytorch/issues/54059 
            

            if torch.distributed.get_rank() == 0:
                e_path = os.path.join(self.model_path, self.backbone + '-%d.pkl' % (epoch))
                torch.save(self.depthnet.state_dict(),e_path)
                torch.distributed.barrier()

    def sample(self,root,eval_name,crop_ratio):
        image_list = os.listdir(root)
        eval_image = []
        min_v = None
        for image_name in image_list:
            eval_image.append(os.path.join(root,image_name))
        
        index = 0  
        for image_path in eval_image:
            index = index + 1

            input_image = np.array(Image.open(image_path),np.float32)[...,:3]/255.
            
            H = 512
            W = 1024

            input_image = torch.FloatTensor(input_image.transpose([2,0,1])).cuda().unsqueeze(0)            
        
            if self.backbone == 'EGformer':
                y_bon_, y_cor_ = self.depthnet(input_image)
            
            elif self.backbone == 'Panoformer':
                y_bon_, y_cor_ = self.depthnet(input_image)

            if True:
                vis_out = visualize_a_data(input_image[0].cpu(),
                                   torch.FloatTensor(y_bon_[0].cpu()).cpu(),
                                   torch.FloatTensor(y_cor_[0].cpu().cpu()))
            else:
                vis_out = None
            y_bon_ = y_bon_.cpu().numpy()
            y_cor_ = y_cor_.cpu().numpy()
            y_bon_ = (y_bon_[0] / np.pi + 0.5) * H - 0.5
            y_bon_[0] = np.clip(y_bon_[0], 1, H/2-1)
            y_bon_[1] = np.clip(y_bon_[1], H/2+1, H-2)
            y_cor_ = y_cor_[0, 0]

            # Init floor/ceil plane
            z0 = 50
            _, z1 = post_proc.np_refine_by_fix_z(*y_bon_, z0)
            force_raw = True
            force_cuboid = False
            r = 0.05
            if force_raw:
                # Do not run post-processing, export raw polygon (1024*2 vertices) instead.
                # [TODO] Current post-processing lead to bad results on complex layout.
                cor = np.stack([np.arange(1024), y_bon_[0]], 1)

            else:
                # Detech wall-wall peaks
                if min_v is None:
                    min_v = 0 if force_cuboid else 0.05
                r = int(round(W * r / 2))
                N = 4 if force_cuboid else None
                xs_ = find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0]

                # Generate wall-walls
                cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=force_cuboid)
                if not force_cuboid:
                    # Check valid (for fear self-intersection)
                    xy2d = np.zeros((len(xy_cor), 2), np.float32)
                    for i in range(len(xy_cor)):
                        xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
                        xy2d[i, xy_cor[i-1]['type']] = xy_cor[i-1]['val']
                    if not Polygon(xy2d).is_valid:
                        print(
                            'Fail to generate valid general layout!! '
                            'Generate cuboid as fallback.',
                            file=sys.stderr)
                        xs_ = find_N_peaks(y_cor_, r=r, min_v=0, N=4)[0]
                        cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)

            # Expand with btn coory
            cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])

            # Collect corner position in equirectangular
            cor_id = np.zeros((len(cor)*2, 2), np.float32)           
            for j in range(len(cor)):
                cor_id[j*2] = cor[j, 0], cor[j, 1]
                cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]

            # Normalized to [0, 1]
            cor_id[:, 0] /= W
            cor_id[:, 1] /= H
            if vis_out is not None:
                vis_path = os.path.join(self.eval_path, str(index) + '_' 'layout.png')
                vh, vw = vis_out.shape[:2]
                Image.fromarray(vis_out)\
                     .resize((vw//2, vh//2), Image.LANCZOS)\
                     .save(vis_path)       


