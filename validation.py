import torch
import torch.nn.functional as F
import time
import os
import math
import shutil
import os.path as osp
import matplotlib.pyplot as plt
import torchvision
from collections import OrderedDict
import pandas as pd
from fvcore.nn import FlopCountAnalysis
import matplotlib as mpl
import matplotlib.cm as cm
import argparse
import importlib
import numpy as np

# This code is used to log training progress using wandb api

class BadPixelMetric:
    def __init__(self, threshold=1.25, depth_cap=10,data_type='None',align_type='Image'):
        self.__threshold = threshold
        self.__depth_cap = depth_cap
        self.data_type = data_type
        self.align_type = align_type

    # From https://github.com/isl-org/MiDaS
    def compute_scale_and_shift(self, prediction, target, mask):
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)
        
        det = a_00 * a_11 - a_01 * a_01
        
        # A needs to be a positive definite matrix.
        valid = det > 0

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0, x_1

    def __call__(self, prediction, target, mask):

        ##### Column/Image-wise scale-and-shift alignment ####    
        if self.align_type == 'Image':

            prediction = prediction.squeeze(0)
            target = target.squeeze(0)
            mask = mask.squeeze(0)
        elif self.align_type == 'Column':
            prediction = prediction
            target = target
            mask = mask
        else:
            print("align type error")
 

        scale, shift = self.compute_scale_and_shift(prediction, target, mask)

        scale = scale.unsqueeze(0).unsqueeze(0)
        shift = shift.unsqueeze(0).unsqueeze(0)
        
        prediction_aligned = scale * prediction + shift

        depth_cap = self.__depth_cap
        
        prediction_aligned[prediction_aligned > depth_cap] = depth_cap
        gt = target
        pred = prediction_aligned

        
        abs_rel_error = ((pred[mask>0] - gt[mask>0]).abs() / gt[mask>0]).mean()
        sq_rel_error = (((pred[mask>0] - gt[mask>0]) ** 2) / gt[mask>0]).mean()
        lin_rms_sq_error = ((pred[mask>0] - gt[mask>0]) ** 2).mean()
        mask_log = (mask > 0) & (pred > 1e-7) & (gt > 1e-7) # Compute a mask of valid values
        log_rms_sq_error = ((pred[mask_log].log() - gt[mask_log].log()) ** 2).mean()
        d1_ratio = (torch.max(pred[mask>0] / gt[mask>0], gt[mask>0] / pred[mask>0]) < (1.25 ** 1)).float().mean()
        d2_ratio = (torch.max(pred[mask>0] / gt[mask>0], gt[mask>0] / pred[mask>0]) < (1.25 ** 2)).float().mean()
        d3_ratio = (torch.max(pred[mask>0] / gt[mask>0], gt[mask>0] / pred[mask>0]) < (1.25 ** 3)).float().mean()

        err = torch.zeros_like(pred, dtype=torch.float)

        err[mask == 1] = torch.max(
            pred[mask == 1] / target[mask == 1],
            target[mask == 1] / pred[mask == 1],
        )

        err[mask == 1] = (err[mask == 1] > self.__threshold).float()

        p = torch.sqrt(((pred[mask>0] - gt[mask>0]) ** 2).mean())

        return p, abs_rel_error, sq_rel_error,lin_rms_sq_error,log_rms_sq_error,d1_ratio,d2_ratio,d3_ratio

# From https://github.com/fyu/drn
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {
            'val': self.val,
            'sum': self.sum,
            'count': self.count,
            'avg': self.avg
        }

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']

class Validation(object):

    def __init__(self,
                 config,
                 val_dataloader
                 ):

        self.val_dataloader = val_dataloader
        self.config = config
        # Some timers
        self.batch_time_meter = AverageMeter()
        # Some trackers
        self.epoch = 0

        # Accuracy metric trackers
        self.rmse_error_meter = AverageMeter()
        self.abs_rel_error_meter = AverageMeter()
        self.sq_rel_error_meter = AverageMeter()
        self.lin_rms_sq_error_meter = AverageMeter()
        self.log_rms_sq_error_meter = AverageMeter()
        self.d1_inlier_meter = AverageMeter()
        self.d2_inlier_meter = AverageMeter()
        self.d3_inlier_meter = AverageMeter()

        # List of length 2 [Visdom instance, env]
        
    def validation(self,model):

        print('Validating....')

        # Put the model in eval mode
        
        self.use_hybrid = False

        self.net = model
        self.net.eval()
        # Reset meter
        self.reset_eval_metrics()
        
        # start / end model
        
        # Load data
        s = time.time()
        max_batch_num = len(self.val_dataloader)-1
        with torch.no_grad():
            for batch_num, data in enumerate(self.val_dataloader):
                print(
                    'Evaluating {}/{}'.format(batch_num, len(
                        self.val_dataloader)),
                    end='\r')
                
                if self.config.val_data == 'S3D':
                    inputs = data['color'].float().cuda()
                    gt = data['depth'].float().cuda()

                    gt = gt / gt.max()
                    gt = gt * 10.

                    self.input_shape = torch.zeros(inputs.shape)  # Used for calculating # params and MACs
                
                elif self.config.val_data == 'Pano3D':
                    inputs = data['color'].float().cuda()

                    gt = data['depth'].float().cuda()

                    gt = gt / gt.max()
                    gt = gt * 10.
                    
                    self.input_shape = torch.zeros(inputs.shape)  # Used for calculating # params and MACs

                if self.config.backbone == 'EGformer':
                    features = self.net.module(inputs)
   
                elif self.config.backbone == 'Panoformer':
                    features = self.net.module(inputs)
                    features = features["pred_depth"]
                output = features


                self.compute_eval_metrics(output, gt)

        # Print a report on the validation results
        
        abs_rel = self.abs_rel_error_meter.avg
        sq_rel = self.sq_rel_error_meter.avg
        lin_rms_sq = math.sqrt(self.lin_rms_sq_error_meter.avg)
        log_rms_sq = math.sqrt(self.log_rms_sq_error_meter.avg)
        d1 = self.d1_inlier_meter.avg 
        d2 = self.d2_inlier_meter.avg
        d3 = self.d3_inlier_meter.avg 
        rmse = self.rmse_error_meter.avg
        params,flops = self.print_param_MACs()
        eval_dict={}
        eval_dict['abs_rel']=abs_rel
        eval_dict['sq_rel']=sq_rel
        eval_dict['lin_rms_sq']=lin_rms_sq
        eval_dict['log_rms_sq']=log_rms_sq
        eval_dict['d1']=d1
        eval_dict['d2']=d2
        eval_dict['d3']=d3
        eval_dict['params']=params
        eval_dict['flops']= flops

        self.net.train()
        return eval_dict


    def reset_eval_metrics(self):
        '''
        Resets metrics used to evaluate the model
        '''
        self.rmse_error_meter.reset()
        self.abs_rel_error_meter.reset()
        self.sq_rel_error_meter.reset()
        self.lin_rms_sq_error_meter.reset()
        self.log_rms_sq_error_meter.reset()
        self.d1_inlier_meter.reset()
        self.d2_inlier_meter.reset()
        self.d3_inlier_meter.reset()
        
        self.is_best = False

    def compute_eval_metrics(self, output, gt,AU=None,EU=None,do_log=False):
        '''
        Computes metrics used to evaluate the model
        '''
        depth_pred = output

        if self.config.val_data == 'S3D':
            gt_depth = gt
            depth_mask = (gt>0).cuda()
        
        elif self.config.val_data == 'Pano3D':
            gt_depth = gt
            depth_mask = (gt<10.).cuda()
        
        else:      
            gt_depth = gt[0]
            depth_mask = gt[1]

      
       
        Bmetric = BadPixelMetric(depth_cap=100,data_type=self.config.val_data,align_type = self.config.align_type)
        Bloss = Bmetric(depth_pred,gt_depth,depth_mask)
        
        N = depth_mask.sum()

        RMSE = Bloss[0]
        abs_rel = Bloss[1]
        sq_rel = Bloss[2]
        rms_sq_lin = Bloss[3]
        rms_sq_log = Bloss[4]
        d1 = Bloss[5]
        d2 = Bloss[6]
        d3 = Bloss[7]
        
        self.rmse_error_meter.update(RMSE, N)       
        self.abs_rel_error_meter.update(abs_rel, N)
        self.sq_rel_error_meter.update(sq_rel, N)
        self.lin_rms_sq_error_meter.update(rms_sq_lin, N)
        self.log_rms_sq_error_meter.update(rms_sq_log, N)
        self.d1_inlier_meter.update(d1, N)
        self.d2_inlier_meter.update(d2, N)
        self.d3_inlier_meter.update(d3, N)


    def print_param_MACs(self):
        with torch.no_grad():
            # Calculate total number of parameters
            params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

            print(f'Total params : {params}')
            
            self.input_shape = self.input_shape.cuda()
            # Calculate total number of MACs
            flopss = FlopCountAnalysis(self.net, self.input_shape)
            flopss.unsupported_ops_warnings(False)
            flopss.uncalled_modules_warnings(False)
            flops = flopss.total()
            print(f'Total FLOPs : {flops}')
        return int(params), int(flops)


def format_size(x: int) -> str:
    return str(x)

        
