import argparse
import os
from trainer import Train
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from data_load import S3D_loader
from torchvision import transforms
import torch
from dataset import PanoCorBonDataset
#from pano_loader.pano_loader import Pano3D

def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
    cudnn.benchmark = True

    torch.manual_seed(159111236)
    torch.cuda.manual_seed_all(4099049123103886)    

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if config.multiprocessing_distributed:
        config.world_size = ngpus_per_node * config.world_size
        torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        main_worker(config.gpu, ngpus_per_node, config)

def main_worker(gpu, ngpus_per_node, config):
    if config.gpu is not None:
        print(f'Use GPU: {gpu} for training')

    if config.distributed:
        if config.dist_url == "envs://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            config.rank = config.rank * ngpus_per_node + gpu
        torch.distributed.init_process_group(backend=config.dist_backend, init_method=config.dist_url, world_size=config.world_size, rank=config.rank)

    transform_s3d = transforms.Compose([
                    transforms.ToTensor()
                    ])

    Data_sampler = PanoCorBonDataset(
            root=config.S3D_path,
            flip=False,  rotate=False, gamma=False,
            stretch=False)

    if config.val_data == 'S3D':
        val_loader =  S3D_loader(config.valid_path,transform = transform_s3d,transform_t = transform_s3d)
    elif config.val_data == 'Pano3D':
        val_loader = Pano3D(
                root = config.pano3d_root,
                part=config.pano3d_part,
                split=config.pano3d_split,
                types=config.pano3d_types,
                )

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(Data_sampler)
    else:
        train_sampler = None

    config.batch_size = int(config.batch_size / ngpus_per_node)
    config.num_workers = int((config.num_workers + ngpus_per_node - 1) / ngpus_per_node)
    
    Train_loader = DataLoader(Data_sampler,batch_size=config.batch_size,num_workers=config.num_workers,pin_memory=True, sampler=train_sampler)
    
    Val_loader = torch.utils.data.DataLoader(
            val_loader,
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False)
    train = Train(config,Train_loader, gpu, train_sampler,Val_loader)
    train.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_set",
                                 type=str,
                                 help="Dataset for training. Concat option uses both dataset",
                                 choices=["Pano3D", "S3D","Concat"],
                                 default="S3D")
    parser.add_argument("--val_data",
                                 type=str,
                                 help="Dataset for evaluation",
                                 choices=["Pano3D", "S3D"],
                                 default="S3D") # Validating Pano3D dataset is not tested yet
    parser.add_argument('--valid_path', type=str, default='../Structure3D/val') # file path which contains images to validated

    parser.add_argument("--align_type",
                                 type=str,
                                 help="Align types for measuring errors",
                                 choices=["Column", "Image"],
                                 default="Image")
    parser.add_argument("--backbone",
                                 type=str,
                                 help="model to be used",
                                 choices=["EGformer","Panoformer"],
                                 default="EGformer")
 
    ####### Data directory ###### 
    parser.add_argument('--S3D_path',help='Folder containing Structure3D dataset', type=str,default='../Structure3D/train') # Structure3D dataset path
   
    parser.add_argument("--pano3d_root", type=str, help="Path to the root folder containing the Pano3D extracted data.",default='YOUR_DATA_PATH/datasets/Pano3D')
    parser.add_argument("--pano3d_part", type=str, help="The Pano3D subset to load.",default='M3D_high')
    parser.add_argument("--pano3d_split", type=str, help="The Pano3D split corresponding to the selected subset that will be loaded.",default='./pano_loader/Pano3D/splits/M3D_v1_train.yaml')
    parser.add_argument('--pano3d_types', default=['color','depth','mask'], nargs='+',
            choices=[
                'color', 'depth', 'normal', 'semantic', 'structure', 'layout',
                'color_up', 'depth_up', 'normal_up', 'semantic_up', 'structure_up', 'layout_up'
                'color_down', 'depth_down', 'normal_down', 'semantic_down', 'structure_down', 'layout_down'
                'color_left', 'depth_left', 'normal_left', 'semantic_left', 'structure_left', 'layout_left'
                'color_right', 'depth_right', 'normal_right', 'semantic_right', 'structure_right', 'layout_right'
            ],
            help='The Pano3D data types that will be loaded, one of [color, depth, normal, semantic, structure, layout], potentially suffixed with a stereo placement from [up, down, left, right].'
        )
    parser.add_argument('--val_path', type=str, default='./EVAL_SAMPLE') # file path which contains images to sampled

    ##### Starting options #####
    parser.add_argument('--Continue', help=' Strat training from the checkpoint', action='store_true')
    parser.add_argument('--load_convonly', help=' Strat training from the checkpoint, but only initialize convolutional layers', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, help='path of pretrained weight', default='')


    ##### hyper parameters #####
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)        
    parser.add_argument('--beta2', type=float, default=0.999)
    
    ############## Directory ############## 
    parser.add_argument('--model_name',help='path where models to be saved' , type=str, default='./checkpoints/default') 
    parser.add_argument('--model_path', type=str, default='models')
    parser.add_argument('--sample_path', type=str, default='samples')
    parser.add_argument('--eval_path', type=str, default='evaluate')


    ############ Set log step ############
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_step', type=int , default=500)
    parser.add_argument('--eval_crop_ratio', type=int , default=0)
    
    ############ Distributed Data Parallel (DDP) ############
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=-1)
    parser.add_argument('--gpu', type=str, default="0,1,2,3")
    parser.add_argument('--dist-url', type=str, default="tcp://127.0.0.1:7777")
    parser.add_argument('--dist-backend', type=str, default="nccl")
    parser.add_argument('--multiprocessing_distributed', default=True)
    
    ########### wandb option ############
    parser.add_argument('--wandb', help='Enable wandb logging', action='store_true')
    parser.add_argument('--wandb_name', help='wandb name',type=str)
    parser.add_argument('--wandb_project', help='wandb project',type=str)
    parser.add_argument('--wandb_entity', help='wandb entity',type=str)





    config = parser.parse_args()
    
    if not os.path.exists(config.model_name):
        os.mkdir(config.model_name)
    config_path = os.path.join(config.model_name,'config.txt')
    f = open(config_path,'w')
    print(config,file=f)
    f.close()
 
    print(config)
    main(config)
