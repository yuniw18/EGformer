# Semantic segmentation
For the users who want to test semantic segmentation under our experimental setup, we provide sample codes.

🔔 NOTE: The purpose of this repo is just to show an example of applying semantic segmentation codes to our experimental setup. Therefore, before using these codes for your experiments, you need to check the codes carefully whether there exist errors which may impair performances. 

🔔 NOTE: To make output format suitable for semantic segmentation, we modify the output layer of EGformer/Panoformer sligntly.

## 1. Setup
#### 1) Prepare Structure3D dataset by following the instructions in depth estimation repo. 

Check whether there exists semantic.png file as below. 

```bash
├── Structure3D
   ├── scene_n
        ├── 2D_rendering
            ├── room_number
                ├── panorama
                    ├── full
                        ├── rgb_rawlight.png
                        ├── semantic.png
                        ├── depth.png
                 

``` 
#### 2) Prepare depth pretrained models by follwing the instructions in depth estimation repo. 
We utilize depth pre-trained models to initizliae the convolution weight parameters when training each model.

## 2. Training

#### 1) Run the scripts in 'train_scripts' folder to train EGformer/Panoformer for layout estimation


To train EGformer, do 

~~~bash
bash train_scripts/script_EGformer
~~~

We utilize depth pre-trained models to initizliae the convolution weight parameters.

## 3.  Evaluation

####  1) Go to evaluate folder & run the script
Go to evalute folder & run the script in `eval_script` after the training is done.  

~~~bash
cd evaluate
bash scripts/eval_script
~~~

When we train EGformer\Panopformer using scripts in 'train_scripts' above, we get results below.
Under this training setup, Panoformer provides slightly better semantic segmantation results.

| Model               | mIoU | mACC | 
|---------------------|--------------------------|---|
| EGformer     | 60.9 | 92.77 | 
| Panoformer     | 61.3 | 92.89 | 

<img src="Semantic_segmentation.PNG"  >




