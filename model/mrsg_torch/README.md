# Main changes from the original project 

1. Combine the 3 datasets into one big dataset (ALL.yaml): DRIVE + STARE + CHASE  
2. Train model by ALL.yaml   
3. inference.py - Can output combined result image (original + prediction). Controlled by cfg['COMBINE']    
4. inference.py - Resize input image to training size. We found the model prediction is sensative to image size.    

# Retinal Vessel Segmentation with Pixel-wise Adaptive Filters (ISBI 2022)

# Introduction
![image](https://github.com/Limingxing00/Retinal-Vessel-Segmentation-ISBI2022/blob/main/figure/framework.png)  

This is the official code of **Retinal Vessel Segmentation with Pixel-wise Adaptive Filters and Consistency Training  (ISBI 2022)**. We evaluate our methods on three datasets, DRIVE, CHASE_DB1 and STARE.

# Datesets and results
You can download the three datasets and our results from [Google drive](https://drive.google.com/drive/folders/1OJBy8uNSg-agk6Rz_xV9fTZGb_KC0j_V?usp=sharing).  
Of course, you can download the dataset from [DRIVE](http://www.isi.uu.nl/Research/Databases/DRIVE/), [CHASE_DB1](https://blogs.kingston.ac.uk/retinal/chasedb1/) and [STARE](http://cecas.clemson.edu/~ahoover/stare/) respectively.

# Quick start
## Requirement
1. Refer to [Pytorch](https://pytorch.org/get-started/previous-versions/) to install `Pytorch >= 1.1`.
2. `pip install -r requirements.txt`


## Config file

```
DATASET: "DRIVE"

TRAIN_DATA_PATH: ".../training/images" # modify it to your own path
TRAIN_LABEL_PATH: ".../training/1st_manual"


TEST_DATA_PATH: ".../test/images"
TEST_PRED_PATH: "results/test/DRIVE/prediction"
TEST_LABEL_PATH: ".../test/label/1st_manual"

# view
#VAL_PICTURE_PATH: "/gdata1/limx/mx/dataset/Drive19/visualization"
#VIEW_VAL_PATH: "results/val_view"
#VIEW_TRAIN_PATH: "results/train_view"

MODEL_PATH: "results/test/DRIVE/model"
LOG_PATH: "results/test/DRIVE/logging.txt"

# train
LEARNING_RATE: 0.005
BATCH_SIZE: 5
EPOCH: 6000
CHECK_BATCH: 50
multi_scale: [0.3]
INPUT_CHANNEL: 3
MAX_AFFINITY: 5
RCE_WEIGHT: 1
RCE_RATIO: 0.1

# inference
MODEL_NUMBER: "epoch_2750_f1_0.8261"
# load breakpoint
IS_BREAKPOINT: False
BREAKPOINT: ""


```

Please modify ```TRAIN_DATA_PATH```, ```TRAIN_LABEL_PATH```, ```TEST_DATA_PATH``` and ```TEST_LABEL_PATH```.  

## Training
Please specify the configuration file.  
For example, you can run ```.sh``` file to train the specific dataset.
```python
cd rootdir
sh pbs/DRIVE_RUN.sh
```
After finishing the training stage, you will obtain the ```/results/test/DRIVE/logging.txt```. The logging.txt file can log the metrics, like model number, f1, auc, acc, specificity, precision, sensitivity.

  `python train.py --cfg lib/All.yaml`

If you want to start with pretrained weights, modify the .yaml file as: 
```python
  # load breakpoint / pretrained weights  
  IS_BREAKPOINT: True  
  BREAKPOINT: "results - 20221118 - continue/test/ALL/model/  epoch_450_f1_0.5003.pth"
```

## Testing
Please select the best model in loggging.txt and modify the ```MODEL_NUMBER``` in configuration file.
```python
cd rootdir
python inference.py --lib/DRIVE.yaml 
```


## Evaluation
To evalutate the results offline bewteen `cfg['TEST_PRED_PATH']` and `cfg['TEST_LABEL_PATH']`. Your can run the code like it.
```python
cd rootdir
python eval.py --lib/DRIVE.yaml 
```

# Citation
If you find this work or code is helpful in your research, please cite:
```
@inproceedings{li2022retinal,
  title={Retinal Vessel Segmentation with Pixel-Wise Adaptive Filters},
  author={Li, Mingxing and Zhou, Shenglong and Chen, Chang and Zhang, Yueyi and Liu, Dong and Xiong, Zhiwei},
  booktitle={2022 IEEE 19th International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2022},
  organization={IEEE}
}
```
