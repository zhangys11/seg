This is a collection of image segmentation projects adjusted for fundus blood vessel segmentation.  

# Run

Provide four flavors: 

| Module  | How to run | Model weights location | Notebooks |
| ------------ |:-------------------:|:-------------------:|:-------------------:|
| model.unet_keras | run_training.py, run_testing.py | test/test_best_weights.h5 | 1. U-Net - Introduction.ipynb, 2. Fundus Blood Vessel Segmentation.ipynb   |
| model.unet_torch | train.py, test.py | weights/checkpoint.pth | demo.ipynb |
| model.multiple_torch | all codes are inside .ipynb files | best_binclass_model.h5, best_multiclass_model.h5 |  1. binary segmentation (camvid).ipynb and 2. multiclass segmentation (camvid).ipynb |
| model.mrsg_torch  | python train.py --cfg lib/All.yaml, python inference.py --lib/DRIVE.yaml | results/test/ALL/model/*.pth | demo.ipynb |

# Credits
The following github projects are used:  

| Module  | based on | url |
| ------------ |:-------------------:|:-------------------:|
| model.unet_keras        | Retina blood vessel segmentation with a convolution neural network (U-net)   | https://github.com/orobix/retina-unet        |
| model.unet_torch        | Retina-Blood-Vessel-Segmentation-in-PyTorch | https://github.com/nikhilroxtomar/Retina-Blood-Vessel-Segmentation-in-PyTorch |
| model.multiple_torch        | Python library with Neural Networks for Image Segmentation based on Keras and TensorFlow. | https://github.com/qubvel/segmentation_models |  
| model.mrsg_torch        | Retinal Vessel Segmentation with Pixel-wise Adaptive Filters (ISBI 2022) | https://github.com/Limingxing00/Retinal-Vessel-Segmentation-ISBI2022/ |


# Todo
make a thorough refactor