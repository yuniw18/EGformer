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

## EGformer
from models.egformer import EGDepthModel

## Panoformer
from models.Panoformer.model import Panoformer as PanoBiT


import matplotlib as mpl
import matplotlib.cm as cm
import argparse
import importlib
import numpy as np

from helpers import iouCalc

# Semantic segmantation part in this code is adopted from 'HoHoNet'
# HoHoNet (https://github.com/sunset1995/HoHoNet)

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

class Evaluation(object):

    def __init__(self,
                 config,
                 val_dataloader,
                 gpu):

        self.val_dataloader = val_dataloader
        self.config = config
        self.gpu = gpu
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
        self.acc = AverageMeter() 

        self.validClasses = np.arange(0,41)
        self.evalClasses = np.arange(0,41)
        # List of length 2 [Visdom instance, env]
        self.confMatrix     = np.zeros(shape=(len(self.validClasses),len(self.validClasses)),dtype=np.ulonglong)
        # Loss trackers
        self.loss = AverageMeter()
    def post_process_disparity(self,disp):
        disp = disp.cpu().detach().numpy()
        return disp   
                
    def visualize_label_image(self,pred,GT=False):
        color_palette = create_color_palette() / 255.
        pred_imgs = [color_palette[p] for p in pred.cpu().numpy()]
        
        return torch.from_numpy(pred_imgs[0]).permute(2,0,1)

    def evaluate_panoformer(self):

        print('Evaluating Panoformer')
        # Put the model in eval mode
        torch.cuda.set_device(self.gpu)
        
        self.net = PanoBiT()
        self.net.cuda(self.gpu).eval()

        self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[self.gpu], find_unused_parameters=True)
 

        self.net.load_state_dict(torch.load(self.config.checkpoint_path),strict=True)


        # Reset meter
        self.reset_eval_metrics()

        iou = iouCalc(classLabels = self.validClasses, validClasses = self.validClasses, voidClass = -1)
        res = 512 * 1024
 
        # Load data
        s = time.time()
        with torch.no_grad():
            for batch_num, data in enumerate(self.val_dataloader):
                print(
                    'Evaluating {}/{}'.format(batch_num, len(
                        self.val_dataloader)),
                    end='\r')
               
                if self.config.eval_data == 'Structure3D':
                    inputs = data[0].float().cuda()
 
                    gt = data[1].float().cuda()
                    gt =gt.squeeze(0).long()

                    self.input_shape = torch.zeros(inputs.shape)  # Used for calculating # params and MACs
                features = self.net(inputs)
                label_pred = features["pred_depth"]
                label_pred = torch.argmax(label_pred,1)               

                bs = inputs.size(0)
                corrects = torch.sum(label_pred == gt.data)

                acc = corrects.double()/(bs * res)
                self.acc.update(acc, bs)
                
                iou_pred = iou.evaluateBatch(label_pred , gt.squeeze(1))
                if False:    
                    eval_name = 'Sample_%d.png' %(batch_num)
                    gt_name = 'Sample_gt_%d.png' %(batch_num)

                    pred_imgs = self.visualize_label_image(label_pred)
                    pred_gt = self.visualize_label_image(gt,GT=True)
                
                    torchvision.utils.save_image(pred_imgs.data,os.path.join(self.config.output_path, eval_name))
                    torchvision.utils.save_image(pred_gt.data,os.path.join(self.config.output_path, gt_name))



            miou = iou.outputScores()
            print('mAcc: {:.4f}\n\n'.format(self.acc.avg))
                


    def evaluate_egformer(self):

        print('Evaluating EGformer')

        # Put the model in eval mode
        
        self.use_hybrid = False
        torch.cuda.set_device(self.gpu)
        
        self.net = EGDepthModel(hybrid=self.use_hybrid)

        self.net.cuda(self.gpu)
        self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[self.gpu], find_unused_parameters=True)

        self.net.load_state_dict(torch.load(self.config.checkpoint_path),strict=False)
        self.net.eval()

        # Reset meter
        self.reset_eval_metrics()

        iou = iouCalc(classLabels = self.validClasses, validClasses = self.validClasses, voidClass = -1)
        res = 512 * 1024
      
        # start / end model
        
        # Load data
        s = time.time()
        with torch.no_grad():
            for batch_num, data in enumerate(self.val_dataloader):
                print(
                    'Evaluating {}/{}'.format(batch_num, len(
                        self.val_dataloader)),
                    end='\r')
                if self.config.eval_data == 'Structure3D':
                    inputs = data[0].float().cuda()
                    gt = data[1].float().cuda()
                    gt =gt.squeeze(0).long()
    
                self.input_shape = torch.zeros(inputs.shape)  # Used for calculating # params and MACs

                label_pred = self.net(inputs)
                label_pred = torch.argmax(label_pred,1)               

                bs = inputs.size(0)
                corrects = torch.sum(label_pred == gt.data)

                acc = corrects.double()/(bs * res)
                self.acc.update(acc, bs)
                
                iou_pred = iou.evaluateBatch(label_pred , gt.squeeze(1))
                if False:    
                    input_name = 'Input_%d.png' %(batch_num)
                    eval_name = 'Sample_%d.png' %(batch_num)
                    gt_name = 'Sample_gt_%d.png' %(batch_num)

                    pred_imgs = self.visualize_label_image(label_pred)
                    pred_gt = self.visualize_label_image(gt,GT=True)
                
                    torchvision.utils.save_image(inputs.data,os.path.join(self.config.output_path, input_name))
                    torchvision.utils.save_image(pred_imgs.data,os.path.join(self.config.output_path, eval_name))
                    torchvision.utils.save_image(pred_gt.data,os.path.join(self.config.output_path, gt_name))


            miou = iou.outputScores()
            print('mAcc: {:.4f}\n\n'.format(self.acc.avg))
 


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
    
    def getIouScoreForLabel(self, label):
        # Calculate and return IOU score for a particular label (train_id)
        if label == self.voidClass:
            return float('nan')
    
        # the number of true positive pixels for this label
        # the entry on the diagonal of the confusion matrix
        tp = np.longlong(self.confMatrix[label,label])
    
        # the number of false negative pixels for this label
        # the row sum of the matching row in the confusion matrix
        # minus the diagonal entry
        fn = np.longlong(self.confMatrix[label,:].sum()) - tp
    
        # the number of false positive pixels for this labels
        # Only pixels that are not on a pixel with ground truth label that is ignored
        # The column sum of the corresponding column in the confusion matrix
        # without the ignored rows and without the actual label of interest
        notIgnored = [l for l in self.validClasses if not l == self.voidClass and not l==label]
        fp = np.longlong(self.confMatrix[notIgnored,label].sum())
    
        # the denominator of the IOU score
        denom = (tp + fp + fn)
        if denom == 0:
            return float('nan')
    
        # return IOU
        return float(tp) / denom
    
    def evaluateBatch(self, predictionBatch, groundTruthBatch):
        # Calculate IoU scores for single batch
        assert predictionBatch.size(0) == groundTruthBatch.size(0), 'Number of predictions and labels in batch disagree.'
        
        # Load batch to CPU and convert to numpy arrays
        predictionBatch = predictionBatch.cpu().numpy()
        groundTruthBatch = groundTruthBatch.cpu().numpy()

        for i in range(predictionBatch.shape[0]):
            predictionImg = predictionBatch[i,:,:]
            groundTruthImg = groundTruthBatch[i,:,:]
            
            # Check for equal image sizes
            assert predictionImg.shape == groundTruthImg.shape, 'Image shapes do not match.'
            assert len(predictionImg.shape) == 2, 'Predicted image has multiple channels.'
        
            imgWidth  = predictionImg.shape[0]
            imgHeight = predictionImg.shape[1]
            nbPixels  = imgWidth*imgHeight
             # Evaluate images
            encoding_value = max(groundTruthImg.max(), predictionImg.max()).astype(np.int32) + 1
            encoded = (groundTruthImg.astype(np.int32) * encoding_value) + predictionImg
        
            values, cnt = np.unique(encoded, return_counts=True) # number of values = cnt
        
            for value, c in zip(values, cnt):
                pred_id = value % encoding_value
                gt_id = int((value - pred_id)/encoding_value)
                if not gt_id in self.validClasses:
                    printError('Unknown label with id {:}'.format(gt_id))
                self.confMatrix[gt_id][pred_id] += c
        
            # Calculate pixel accuracy
            notIgnoredPixels = np.in1d(groundTruthImg, self.evalClasses, invert=True).reshape(groundTruthImg.shape)
            erroneousPixels = np.logical_and(notIgnoredPixels, (predictionImg != groundTruthImg))
            nbNotIgnoredPixels = np.count_nonzero(notIgnoredPixels)
            nbErroneousPixels = np.count_nonzero(erroneousPixels)
            self.perImageStats.append([nbNotIgnoredPixels, nbErroneousPixels])
            
            self.nbPixels += nbPixels
            
        return
            
    def outputScores(self):
        # Output scores over dataset
        assert self.confMatrix.sum() == self.nbPixels, 'Number of analyzed pixels and entries in confusion matrix disagree: confMatrix {}, pixels {}'.format(self.confMatrix.sum(),self.nbPixels)
    
        # Calculate IOU scores on class level from matrix
        classScoreList = []
            
        # Print class IOU scores
        outStr = 'classes           IoU\n'
        outStr += '---------------------\n'
        for c in self.evalClasses:
            iouScore = self.getIouScoreForLabel(c)
            classScoreList.append(iouScore)
            outStr += '{:<14}: {:>5.3f}\n'.format(c, iouScore)
        miou = getScoreAverage(classScoreList)
        outStr += '---------------------\n'
        outStr += 'Mean IoU      : {avg:5.3f}\n'.format(avg=miou)
        outStr += '---------------------'
        
        print(outStr)
        
        return miou
 



    def print_validation_report(self):
        '''
        Prints a report of the validation results
        '''
        print('Epoch: {}\n'
              '  Avg. Abs. Rel. Error: {:.4f}\n'
              '  Avg. Sq. Rel. Error: {:.4f}\n'
              '  Avg. Lin. RMS Error: {:.4f}\n'
              '  Avg. Log RMS Error: {:.4f}\n'
              '  Inlier D1: {:.4f}\n'
              '  Inlier D2: {:.4f}\n'
              '  Inlier D3: {:.4f}\n'
              '  RMSE: {:.4f}\n\n'.format(
                  self.epoch + 1, self.abs_rel_error_meter.avg,
                  self.sq_rel_error_meter.avg,
                  math.sqrt(self.lin_rms_sq_error_meter.avg),
                  math.sqrt(self.log_rms_sq_error_meter.avg),
                  self.d1_inlier_meter.avg, self.d2_inlier_meter.avg,
                  self.d3_inlier_meter.avg, self.rmse_error_meter.avg))
        self.print_param_MACs()

    def print_param_MACs(self):
        with torch.no_grad():
            # Calculate total number of parameters
            params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

            params = format_size(params)
            print(f'Total params : {params}')

            self.input_shape = self.input_shape.cuda()
            # Calculate total number of MACs
            flopss = FlopCountAnalysis(self.net, self.input_shape)
            flopss.unsupported_ops_warnings(False)
            flopss.uncalled_modules_warnings(False)
            flops = format_size(flopss.total())
            print(f'Total FLOPs : {flops}')



def format_size(x: int) -> str:
    if x > 1e8:
        return "{:.1f}G".format(x / 1e9)
    if x > 1e5:
        return "{:.1f}M".format(x / 1e6)
    if x > 1e2:
        return "{:.1f}K".format(x / 1e3)
    return str(x)

        
