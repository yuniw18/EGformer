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

import time
from fvcore.nn import FlopCountAnalysis


def create_color_palette():
    return np.array([
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),  
       (247, 182, 210),		# desk
       (66, 188, 102), 
       (219, 219, 141),		# curtain
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14), 		# refrigerator
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209), 
       (227, 119, 194),		# bathtub
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)
    ])


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
        self.criterion = nn.CrossEntropyLoss() 

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
    def visualize_label_image(self,pred,GT=False):
        if not GT:
            pred = torch.argmax(pred, dim=1)
        color_palette = create_color_palette() / 255.
        pred_imgs = [color_palette[p] for p in pred.cpu().numpy()]
        
        return torch.from_numpy(pred_imgs[0]).permute(2,0,1)

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
 
########################################################################

        for epoch in range(self.num_epochs):
            if self.distributed:
                self.train_sampler.set_epoch(epoch)

            for batch_num, data in enumerate(self.train_loader):
                if True:
                    inputs = self.to_variable(data['color'])
                    gt = self.to_variable(data['depth'])
                    label_gt = self.to_variable(data['semantic'].squeeze(0)).long()

                    gt = gt / 8. # Set <1 depth range 

                    if self.backbone == 'EGformer':
                        label_pred = self.depthnet(inputs)

                    elif self.backbone == 'Panoformer':
                        features = self.depthnet(inputs)
                        label_pred = features["pred_depth"]
                    
                    loss = self.criterion(label_pred, label_gt)                   
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
                        eval_name = 'Sample_%d.png' %(batch_num)
                        gt_name = 'Sample_gt_%d.png' %(batch_num)

                        pred_imgs = self.visualize_label_image(label_pred)
                        pred_gt = self.visualize_label_image(label_gt,GT=True)

                        torchvision.utils.save_image(pred_imgs.data,os.path.join(self.eval_path, eval_name))
                        torchvision.utils.save_image(pred_gt.data,os.path.join(self.eval_path, gt_name))


            if torch.distributed.get_rank() == 0:
                e_path = os.path.join(self.model_path, self.backbone + '-%d.pkl' % (epoch))
                torch.save(self.depthnet.state_dict(),e_path)
                torch.distributed.barrier()

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

    def validate_epoch(dataloader, model, criterion, epoch, classLabels, validClasses, void=-1, maskColors=None, folder='baseline_run', args=None):
        acc_running = AverageMeter('Accuracy', ':.4e')
        iou = iouCalc(classLabels, validClasses, voidClass = void)
    
        # input resolution
        res = 512*1024    
        # Set model in evaluation mode
        model.eval()
    
        with torch.no_grad():
            for epoch_step, (inputs, labels, filepath) in enumerate(dataloader):
                data_time.update(time.time()-end)
                
                inputs = inputs.float().cuda()
                labels = labels.long().cuda()
    
                # forward
                outputs = model(inputs)
                preds = torch.argmax(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Statistics
                bs = inputs.size(0) # current batch size
                corrects = torch.sum(preds == labels.data)
                nvoid = int((labels==void).sum())
                acc = corrects.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
                acc_running.update(acc, bs)
                # Calculate IoU scores of current batch
                iou.evaluateBatch(preds, labels)
                
                progress.display(epoch_step)
        
            miou = iou.outputScores()
            print('Accuracy      : {:5.3f}'.format(acc_running.avg))
            print('---------------------')

        return acc_running.avg,  miou



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
                label_pred = self.depthnet(input_image)
            
            elif self.backbone == 'Panoformer':
                features = self.depthnet(input_image)
                label_pred = features["pred_depth"]
            
            pred_imgs = self.visualize_label_image(label_pred)

            save_name = eval_name + '_'+str(index)+'.png'        


