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
from midas_loss import ScaleAndShiftInvariantLoss
from models.egformer import EGDepthModel
from torch.nn.parallel import DistributedDataParallel as DDP

from models.Panoformer.model import Panoformer as PanoBiT

from validation import Validation
import time
from fvcore.nn import FlopCountAnalysis

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
        
        self.scale_loss = ScaleAndShiftInvariantLoss().cuda(gpu)
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

            
        self.Validation = Validation(self.config,self.val_loader)
 
        self.g_optimizer = optim.AdamW([{"params": list(self.depthnet.parameters())}],
                                        self.lr,[self.beta1,self.beta2])


        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer, 0.95)

        if self.config.wandb and self.is_main_process():
            import wandb
            wandb.init(project=self.config.wandb_project,entity=self.config.wandb_entity,name=self.config.wandb_name,settings=wandb.Settings(code_dir='.'))
            wandb.run.log_code('.')


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
                if self.config.wandb:
                    with torch.no_grad():
                        eval_dict = self.Validation.validation(model=self.depthnet)
                        import wandb
                        wandb.log({'abs_rel':eval_dict['abs_rel'],'sq_rel':eval_dict['sq_rel'],'lin_rms_sq':eval_dict['lin_rms_sq'],'log_rms_sq':eval_dict['log_rms_sq'],'d1':eval_dict['d1'],'d2':eval_dict['d2'],'d3':eval_dict['d3'],'params':torch.cuda.FloatTensor([eval_dict['params']]),'flops':torch.cuda.FloatTensor([eval_dict['flops']]),'learning_rate':self.lr},step=0)
 
########################################################################

        for epoch in range(self.num_epochs):
            if self.distributed:
                self.train_sampler.set_epoch(epoch)

            for batch_num, data in enumerate(self.train_loader):
                if True:
                    inputs = self.to_variable(data['color'])
                    gt = self.to_variable(data['depth'])
                    mask = self.to_variable(data['mask'])

                    gt = gt / 8. # Set <1 depth range 

                    if self.backbone == 'EGformer':
                        depth = self.depthnet(inputs)

                    elif self.backbone == 'Panoformer':
                        features = self.depthnet(inputs)
                        depth = features["pred_depth"]
                    
                    loss = self.scale_loss(depth,gt,mask)
                    
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
                            print('Epoch [%d/%d], Step[%d/%d], image_loss: %.5f' 
                              %(epoch, self.num_epochs, batch_num, max_batch_num, 
                                loss.item()))
                        except:
                            print("Skip logging") # 

                if (batch_num) % self.sample_step == 0:
                    if torch.distributed.get_rank() == 0:
                        eval_name = 'Sample_%d' %(epoch)
                        with torch.no_grad():
                            self.sample(self.eval_data_path,eval_name,self.crop_ratio)
            if torch.distributed.get_rank() not in [-1,0]:
                torch.distributed.barrier()
            ### Make sure only rank 0 save the model / validation
            # Refer to https://github.com/pytorch/pytorch/issues/54059 
            

            if torch.distributed.get_rank() == 0:
                e_path = os.path.join(self.model_path, self.backbone + '-%d.pkl' % (epoch))
                torch.save(self.depthnet.state_dict(),e_path)
                torch.distributed.barrier()
                if self.config.wandb: 
                    with torch.no_grad():
                        eval_dict = self.Validation.validation(model=self.depthnet)
                        import wandb
                        wandb.log({'abs_rel':eval_dict['abs_rel'],'sq_rel':eval_dict['sq_rel'],'lin_rms_sq':eval_dict['lin_rms_sq'],'log_rms_sq':eval_dict['log_rms_sq'],'d1':eval_dict['d1'],'d2':eval_dict['d2'],'d3':eval_dict['d3'],'params':torch.cuda.FloatTensor([eval_dict['params']]),'flops':torch.cuda.FloatTensor([eval_dict['flops']]),'learning_rate':self.lr},step=epoch+1)
#            self.lr_scheduler.step() # Not used for egformer report

    def post_process_disparity(self,disp):
        
        disp = disp.cpu().detach().numpy() 
        _, h, w = disp.shape
        l_disp = disp[0,:,:]
        
        return l_disp


    def process_sample(self,sample,height,width):
        map = sample

        original_height = height
        original_width = width

        map = self.post_process_disparity(map.squeeze(1)).astype(np.float32)

        pred_width = map.shape[1]
        map = cv2.resize(map.squeeze(), (original_width, original_height))
        map = map.squeeze()

        vmax = np.percentile(map, 95)
        normalizer = mpl.colors.Normalize(vmin=map.min(), vmax=map.max())
        mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
        
        map = (mapper.to_rgba(map)[:, :, :3] * 255).astype(np.uint8)

        return map

    def sample(self,root,eval_name,crop_ratio):
        image_list = os.listdir(root)
        eval_image = []
        for image_name in image_list:
            eval_image.append(os.path.join(root,image_name))
        
        index = 0  
        for image_path in eval_image:
            index = index + 1
 
            input_image = (imread(image_path).astype("float32")/255.0)
            original_height, original_width, num_channels = input_image.shape
        
            input_height = 512
            input_width = 1024

            input_image = skimage.transform.resize(input_image, [input_height-2 * crop_ratio , input_width])
            input_image = np.pad(input_image, ((crop_ratio,crop_ratio),(0,0),(0,0)), mode='constant')
            input_image = input_image.astype(np.float32)
            
            input_image = torch.from_numpy(input_image).unsqueeze(0).float().permute(0,3,1,2).cuda()
        
            
            if self.backbone == 'EGformer':
                depth = self.depthnet(input_image)
            
            elif self.backbone == 'Panoformer':
                features = self.depthnet(input_image)
                depth = features["pred_depth"]
            
            depth = self.process_sample(depth, original_height, original_width)


            save_name = eval_name + '_'+str(index)+'.png'        

            plt.imsave(os.path.join(self.eval_path,save_name ), depth, cmap='viridis')


