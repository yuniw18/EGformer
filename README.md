# EGformer

This repository contains the code of the paper 
>["EGformer: Equirectangular geometry-biased transformer for 360 depth estimation."](https://arxiv.org/abs/2304.07803)  
>
>Ilwi Yun, Chanyong Shin, Hyunku Lee, Hyuk-Jae Lee, Chae Eun Rhee.
>
>ICCV2023


Our codes are based on the following repositories: [CSWin Transformer](https://github.com/microsoft/CSWin-Transformer), [Panoformer](https://github.com/zhijieshen-bjtu/PanoFormer), [MiDaS](https://github.com/isl-org/MiDaS), [Pano3D](https://github.com/VCL3D/Pano3D) and others.

We'd like to thank the authors providing the codes.


## Changelog

[23/09/27] Initialize repo & upload experiment report.<br> 
[23/10/18] Upload inference/evaluation codes.

## :blue_book: Experiment Report 
To check the reproducibility, we re-trained some models in the paper under slightly different environment, and log the training progress of them. 
Some additional experiments are also conducted for further analysis, which can be found in this [:blue_book: ***EGformer Report***](https://api.wandb.ai/links/yuniw/21nqqyl8). 

## 1. Setup

These codes are tested under PyTorch (1.8) with a 4 NVIDIA v100 GPU.

#### 1) Set up dependencies
By using Anaconda pacakge, environment can be set. Open the `depth_1.8.ymal` file and modify the 'prefix' according to your environment. Then, do the followings.

~~~bash
conda env create --file depth_1.8.yaml
conda activate depth_1.8
~~~

'openexr' package in `depth_1.8.ymal` may not be required (not tested though).

#### 2) Download the pretrained models.
[Download link](https://drive.google.com/drive/folders/15L5RviWqd63TS1L3gwxqggiEgkZS5Yjg?usp=drive_link) 

Then, move the downloaded files (e.g., EGformer_pretrained.pkl) in `pretrained_models` folder.

🔔 NOTE: Currently, some unnecessary parameters (~4mb) are included in EGformer pretrained model. Because they do not affect the other metrics (depth,FLOPs), you can use this version until we provide the clean one.

#### 3) Download the dataset
We use Structure3D and Pano3D dataset for experiments (refer to [Technical Appendix](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Yun_EGformer_Equirectangular_Geometry-biased_ICCV_2023_supplemental.pdf) for more details). 
In our opinion, it looks like Structure3D is the most appropriate dataset to evaluate 360 depths in temrs of quality and quantity. 

Download each dataset and pre-process them. 

* [Structure3D](https://github.com/bertjiazheng/Structured3D) : Create train/val/test split following the instructuions in their github repo. Data folder should be constructed as below.  We use rgb/raw_light samples. Refer to their github repositoiries for more details.

```bash
├── Structure3D
   ├── scene_n
        ├── 2D_rendering
            ├── room_number
                ├── panorama
                    ├── full
                        ├── rgb_rawlight.png
                        ├── depth.png
``` 

* [Pano3D](https://github.com/VCL3D/Pano3D) : Create train/val/test split following instructuions in their github repo. We use `Matterport3D Train & Test (/w Filmic) High Resolution (1024 x 512)` dataset for training & evaluation.
`M3D_high` folder should be located in another folder as below.

```bash
├── Pano3D_folder
   ├── M3D_high
``` 
#### 4) Setup wandb environment (optional)
To get the experiment report as above, you should set up the [wandb](https://wandb.ai/site) environment.
If not required, skip this part.

## 2. Quick start (Inference)

#### 1) Go to evaluate folder & run the following command
~~~bash
cd evaluate
mkdir test_result
bash scripts/inference_scripts
~~~

Then, check if depth results are generated in `test_result` folder.

#### 2) To extract the depths of custom images, run the following command
Put "folders" containing custom images in `INFER_SAMPLE` folder & run the script below.
~~~bash
cd evaluate
mv Custom_image_folder INFER_SAMPLE
bash scripts/inference_scripts
~~~

Then, the depth results will be saved in `test_result` folder. More details about configurations can be found in `evaluate_main.py`

🔔 NOTE: Input resolution is fixed to 512x1024. If input image resolution is not 512x1024, you should resize them manually.


## 3.  Evaluation
To reproduce the experimental results below, follow the instructions. 

| Model               | Testset | Training Set | abs. rel. |Sq.rel |RMS | delta < 1.25  |
|---------------------|--------------------------|--------------------------|-----------------|------|------|----------------|
| EGformer_pretrained     | [Structure3D](https://github.com/bertjiazheng/Structured3D)|  [Structure3D](https://github.com/bertjiazheng/Structured3D) + [Pano3D](https://github.com/VCL3D/Pano3D) | 0.0342    | 0.0279     | 0.2756 |0.9810|
| Panoformer_pretrained    |[Structure3D](https://github.com/bertjiazheng/Structured3D) | [Structure3D](https://github.com/bertjiazheng/Structured3D) + [Pano3D](https://github.com/VCL3D/Pano3D) | 0.0394  | 0.0346     | 0.2960|0.9781|

| Model               | Testset | Training Set | abs. rel. |Sq.rel |RMS | delta < 1.25  |
|---------------------|--------------------------|--------------------------|-----------------|------|------|----------------|
| EGformer_pretrained     | [Pano3D](https://github.com/VCL3D/Pano3D)  |  [Structure3D](https://github.com/bertjiazheng/Structured3D) + [Pano3D](https://github.com/VCL3D/Pano3D) | 0.0660    | 0.0428     | 0.3874 |0.9503|
| Panoformer_pretrained    |[Pano3D](https://github.com/VCL3D/Pano3D) | [Structure3D](https://github.com/bertjiazheng/Structured3D) + [Pano3D](https://github.com/VCL3D/Pano3D) | 0.0699  | 0.0494     | 0.4046|0.9436|


####  1) Go to evaluate folder & run the script
Go to evalute folder & modify the `scripts/eval_script` file based on your purpose and environment.

For example, redefine `--S3D_path` configurations according to your environment. 

Run the scripts below, and you can get the results above.

~~~bash
cd evaluate
bash scripts/eval_script
~~~

When using Pano3D dataset, use OPENCV_IO_ENABLE_OPENEXR=1 option if required.

~~~bash
cd evaluate
OPENCV_IO_ENABLE_OPENEXR=1 bash scripts/eval_script
~~~

🔔 NOTE: Sigmoid activation function is added at the output layer of Panoformer unlike the original version. Details can be found in [:blue_book: ***EGformer Report***](https://api.wandb.ai/links/yuniw/21nqqyl8).




## To do list
- [x] Experiment report  
- [x] Code for inference  
- [x] Code for evaluation 
- [ ] Code for training

## Citation
```
@InProceedings{Yun_2023_ICCV,
    author    = {Yun, Ilwi and Shin, Chanyong and Lee, Hyunku and Lee, Hyuk-Jae and Rhee, Chae Eun},
    title     = {EGformer: Equirectangular Geometry-biased Transformer for 360 Depth Estimation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {6101-6112}
}
```
## License
Our contributions on codes are released under the MIT license. For the codes of the otehr works, refer to their repositories.
