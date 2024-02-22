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
from PIL import Image
## EGformer
from models.egformer import EGDepthModel

## Panoformer
from models.Panoformer.model import Panoformer as PanoBiT
import json
from dataset import cor_2_1d
import matplotlib as mpl
import matplotlib.cm as cm
import argparse
import importlib
import numpy as np
from shapely.geometry import Polygon
from dataset import visualize_a_data
from misc import post_proc

# Layout estimation part in this code is adopted from 'HorizonNet'
# HorizonNet (https://github.com/sunset1995/HorizonNet)

def layout_2_depth(cor_id, h, w, return_mask=False):
    # Convert corners to per-column boundary first
    # Up -pi/2,  Down pi/2
    vc, vf = cor_2_1d(cor_id, h, w)
    vc = vc[None, :]  # [1, w]
    vf = vf[None, :]  # [1, w]
    assert (vc > 0).sum() == 0
    assert (vf < 0).sum() == 0

    # Per-pixel v coordinate (vertical angle)
    vs = ((np.arange(h) + 0.5) / h - 0.5) * np.pi
    vs = np.repeat(vs[:, None], w, axis=1)  # [h, w]

    # Floor-plane to depth
    floor_h = 1.6
    floor_d = np.abs(floor_h / np.sin(vs))

    # wall to camera distance on horizontal plane at cross camera center
    cs = floor_h / np.tan(vf)

    # Ceiling-plane to depth
    ceil_h = np.abs(cs * np.tan(vc))      # [1, w]
    ceil_d = np.abs(ceil_h / np.sin(vs))  # [h, w]

    # Wall to depth
    wall_d = np.abs(cs / np.cos(vs))  # [h, w]

    # Recover layout depth
    floor_mask = (vs > vf)
    ceil_mask = (vs < vc)
    wall_mask = (~floor_mask) & (~ceil_mask)
    depth = np.zeros([h, w], np.float32)    # [h, w]
    depth[floor_mask] = floor_d[floor_mask]
    depth[ceil_mask] = ceil_d[ceil_mask]
    depth[wall_mask] = wall_d[wall_mask]

    assert (depth == 0).sum() == 0
    if return_mask:
        return depth, floor_mask, ceil_mask, wall_mask
    return depth


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


def test_general(dt_cor_id, gt_cor_id, w, h, losses):
    gt_cor_id = gt_cor_id.cpu().numpy()
   
    dt_floor_coor = dt_cor_id[1::2]
    dt_ceil_coor = dt_cor_id[0::2]
    gt_floor_coor = gt_cor_id[1::2]
    gt_ceil_coor = gt_cor_id[0::2]
    assert (dt_floor_coor[:, 0] != dt_ceil_coor[:, 0]).sum() == 0
    assert (gt_floor_coor[:, 0] != gt_ceil_coor[:, 0]).sum() == 0

    # Eval 3d IoU and height error(in meter)
    N = len(dt_floor_coor)
    ch = -1.6
    dt_floor_xy = post_proc.np_coor2xy(dt_floor_coor, ch, 1024, 512, floorW=1, floorH=1)
    gt_floor_xy = post_proc.np_coor2xy(gt_floor_coor, ch, 1024, 512, floorW=1, floorH=1)
    dt_poly = Polygon(dt_floor_xy)
    gt_poly = Polygon(gt_floor_xy)
    if not gt_poly.is_valid:
        print('Skip ground truth invalid (%s)')

        return

    # 2D IoU
    try:
        area_dt = dt_poly.area
        area_gt = gt_poly.area
        area_inter = dt_poly.intersection(gt_poly).area
        iou2d = area_inter / (area_gt + area_dt - area_inter)
    except:
        iou2d = 0

    # 3D IoU
    try:
        cch_dt = post_proc.get_z1(dt_floor_coor[:, 1], dt_ceil_coor[:, 1], ch, 512)
        cch_gt = post_proc.get_z1(gt_floor_coor[:, 1], gt_ceil_coor[:, 1], ch, 512)
        h_dt = abs(cch_dt.mean() - ch)
        h_gt = abs(cch_gt.mean() - ch)
        area3d_inter = area_inter * min(h_dt, h_gt)
        area3d_pred = area_dt * h_dt
        area3d_gt = area_gt * h_gt
        iou3d = area3d_inter / (area3d_pred + area3d_gt - area3d_inter)
    except:
        iou3d = 0

    # rmse & delta_1
    if True:
        gt_layout_depth = layout_2_depth(gt_cor_id, h, w)
        try:
            dt_layout_depth = layout_2_depth(dt_cor_id, h, w)
        except:
            dt_layout_depth = np.zeros_like(gt_layout_depth)
        rmse = ((gt_layout_depth - dt_layout_depth)**2).mean() ** 0.5
        thres = np.maximum(gt_layout_depth/dt_layout_depth, dt_layout_depth/gt_layout_depth)
        delta_1 = (thres < 1.25).mean()

    # Add a result
    n_corners = len(gt_floor_coor)
    if n_corners % 2 == 1:
        n_corners = 'odd'
    elif n_corners < 10:
        n_corners = str(n_corners)
    else:
        n_corners = '10+'
    print("n_corners:" + n_corners)
    losses[n_corners]['2DIoU'].append(iou2d)
    losses[n_corners]['3DIoU'].append(iou3d)
    losses[n_corners]['rmse'].append(rmse)
    losses[n_corners]['delta_1'].append(delta_1)
    losses['overall']['2DIoU'].append(iou2d)
    losses['overall']['3DIoU'].append(iou3d)
    losses['overall']['rmse'].append(rmse)
    losses['overall']['delta_1'].append(delta_1)


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
                
    def visualize_label_image(self,pred,GT=False):
        color_palette = create_color_palette() / 255.
        pred_imgs = [color_palette[p] for p in pred.cpu().numpy()]
        
        return torch.from_numpy(pred_imgs[0]).permute(2,0,1)

    def evaluate_panoformer(self):

        print('Evaluating Panoformer')
        # Put the model in eval mode
        torch.cuda.set_device(0)
        self.gpu = int(self.config.gpu) 
        
        self.net = PanoBiT()
        self.net.cuda(self.gpu).eval()

        self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[self.gpu], find_unused_parameters=True)
 

        self.net.load_state_dict(torch.load(self.config.checkpoint_path),strict=True)


        # Reset meter
        self.reset_eval_metrics()

        res = 512 * 1024
 
        # Load data
        losses = dict([
            (n_corner, {'2DIoU': [], '3DIoU': [], 'rmse': [], 'delta_1': []})
            for n_corner in ['4', '6', '8', '10+', 'odd', 'overall']
        ])
      
        # start / end model
        
        # Load data
        s = time.time()
        with torch.no_grad():
            for batch_num, data in enumerate(self.val_dataloader):
                print(
                    'Evaluating {}/{}'.format(batch_num, len(
                        self.val_dataloader)),
                    end='\r')
   
                inputs, y_bon, gt_cor_id,y_cor =data
                inputs = inputs.cuda()
                y_bon = y_bon.cpu()
                
                self.input_shape = torch.zeros(inputs.shape)  # Used for calculating # params and MACs
                H = 512
                W = 1024
                y_bon_, y_cor_ = self.net(inputs)
                if True:
                    vis_out = visualize_a_data(inputs[0].cpu(),
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

                dt_cor_id = cor_id
        
                test_general(dt_cor_id, gt_cor_id.squeeze(0), W, H, losses)

                for k, result in losses.items():
                    iou2d = np.array(result['2DIoU'])
                    iou3d = np.array(result['3DIoU'])
                    rmse = np.array(result['rmse'])
                    delta_1 = np.array(result['delta_1'])
                    if len(iou2d) == 0:
                        continue
                    print('GT #Corners: %s  (%d instances)' % (k, len(iou2d)))
                    print('    2DIoU  : %.2f' % (iou2d.mean() * 100))
                    print('    3DIoU  : %.2f' % (iou3d.mean() * 100))
                    print('    RMSE   : %.2f' % (rmse.mean()))
                    print('    delta^1: %.2f' % (delta_1.mean()))
        
                if False:   # Set True, if you want to save predicted layout image (before post-processing) 
                    input_name = 'Input_%d.png' %(batch_num)
                    eval_name = 'Sample_%d.png' %(batch_num)
                    gt_name = 'Sample_gt_%d.png' %(batch_num)

                    pred_imgs = self.visualize_label_image(label_pred)
                    pred_gt = self.visualize_label_image(gt,GT=True)
                
                    torchvision.utils.save_image(inputs.data,os.path.join(self.config.output_path, input_name))
                    torchvision.utils.save_image(pred_imgs.data,os.path.join(self.config.output_path, eval_name))
                    torchvision.utils.save_image(pred_gt.data,os.path.join(self.config.output_path, gt_name))

                if False:    # save layout results using json format
                    with open(os.path.join(self.config.output_json_path, str(batch_num) + '.json'), 'w') as f:
                        json.dump({
                            'z0': float(z0),
                            'z1': float(z1),
                            'uv': [[float(u), float(v)] for u, v in cor_id],
                        }, f)

                   
                    if vis_out is not None:
                        vis_path = os.path.join(self.config.output_path, str(batch_num) + '_' 'layout.png')
                        vh, vw = vis_out.shape[:2]
                        Image.fromarray(vis_out)\
                         .resize((vw//2, vh//2), Image.LANCZOS)\
                         .save(vis_path)       



    def evaluate_egformer(self):

        print('Evaluating EGformer')

        # Put the model in eval mode
        
        self.use_hybrid = False
        
        torch.cuda.set_device(self.gpu)
        self.net = EGDepthModel(hybrid=self.use_hybrid)

        self.net.cuda(self.gpu)
#        self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[self.gpu], find_unused_parameters=True)
        self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[self.gpu])


        self.net.load_state_dict(torch.load(self.config.checkpoint_path),strict=True)
        self.net.eval()

        # Reset meter

        res = 512 * 1024
   
        losses = dict([
            (n_corner, {'2DIoU': [], '3DIoU': [], 'rmse': [], 'delta_1': []})
            for n_corner in ['4', '6', '8', '10+', 'odd', 'overall']
        ])
      
        # start / end model
        
        # Load data
        s = time.time()
        with torch.no_grad():
            for batch_num, data in enumerate(self.val_dataloader):
                print(
                    'Evaluating {}/{}'.format(batch_num, len(
                        self.val_dataloader)),
                    end='\r')
                
                inputs, y_bon, gt_cor_id,y_cor =data
                inputs = inputs.cuda()
                y_bon = y_bon.cpu()
                y_cor = y_cor.cpu()
          
                self.input_shape = torch.zeros(inputs.shape)  # Used for calculating # params and MACs
                H = 512
                W = 1024
                y_bon_, y_cor_ = self.net(inputs)
                if True:
                    vis_out = visualize_a_data(inputs[0].cpu(),
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

                dt_cor_id = cor_id
        
                test_general(dt_cor_id, gt_cor_id.squeeze(0), W, H, losses)

                for k, result in losses.items():
                    iou2d = np.array(result['2DIoU'])
                    iou3d = np.array(result['3DIoU'])
                    rmse = np.array(result['rmse'])
                    delta_1 = np.array(result['delta_1'])
                    if len(iou2d) == 0:
                        continue
                    print('GT #Corners: %s  (%d instances)' % (k, len(iou2d)))
                    print('    2DIoU  : %.2f' % (iou2d.mean() * 100))
                    print('    3DIoU  : %.2f' % (iou3d.mean() * 100))
                    print('    RMSE   : %.2f' % (rmse.mean()))
                    print('    delta^1: %.2f' % (delta_1.mean()))
        
                if False:   # Set True, if you want to save predicted layout image (before post-processing) 
                    input_name = 'Input_%d.png' %(batch_num)
                    eval_name = 'Sample_%d.png' %(batch_num)
                    gt_name = 'Sample_gt_%d.png' %(batch_num)

                    pred_imgs = self.visualize_label_image(label_pred)
                    pred_gt = self.visualize_label_image(gt,GT=True)
                
                    torchvision.utils.save_image(inputs.data,os.path.join(self.config.output_path, input_name))
                    torchvision.utils.save_image(pred_imgs.data,os.path.join(self.config.output_path, eval_name))
                    torchvision.utils.save_image(pred_gt.data,os.path.join(self.config.output_path, gt_name))

                if False: # save layout results using json format
                    with open(os.path.join(self.config.output_json_path, str(batch_num) + '.json'), 'w') as f:
                        json.dump({
                            'z0': float(z0),
                            'z1': float(z1),
                            'uv': [[float(u), float(v)] for u, v in cor_id],
                        }, f)

                   
                    if vis_out is not None: # vis_out -> post-processed layout image
                        vis_path = os.path.join(self.config.output_path, str(batch_num) + '_' 'layout.png')
                        vh, vw = vis_out.shape[:2]
                        Image.fromarray(vis_out)\
                         .resize((vw//2, vh//2), Image.LANCZOS)\
                         .save(vis_path)       

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
 




        
