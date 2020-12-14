## Setup

### Dependencies
Ubuntu 20.04, Intel Core i7 3770 and RTX 2080 Ti were used to run the experiments. 

Dependencies can be installed by:
```bash
pip install -r requirements.txt
```

### Download and setup pretrained models
1. Clone this repository

2. Download all files from [here](https://drive.google.com/drive/folders/1nT7nfzqYbrffRwJWhdGFsGjH_wi6w-Pd?usp=sharing) to your local computer

3. Move all .pth files to './models' directory

4. Move FaceEnhancement-checkpoints.zip to './Face_Enhancement' directory, then extract it

5. Move Global-checkpoints.zip to './Global directory', then extract it

### Colorization
1. Training dataset and test dataset (legacy images) can be prepared by:
    - Download test dataset with image_scraper.ipynb
    - Download ImageNet dataset from [here](http://image-net.org/download-images) for training
    - Run preprocess.ipynb to resize test images and create BW images
   
2. Best results for running and fine tuning DeOldify colorization model can be viewed by running inference using 'Colorizer_GANFineTune_bestmodel.pth'. Example inference is shown in 'colorize_test.ipynb'

3. Results for MSGAN training can be viewed by running the 'MSGAN_training.ipynb' file

### Joint with other restoration
The adopted final pipeline consists of preprocessing using Wan et al's (2020) method from [here](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/tree/master/Global) and feeding the result to the colorization model.

To test it, run 'joint_restoration_test.ipynb'

### Acknowledgement

The final code implemented here are heavily based on the implementation from:
1. The original deoldify repository [here](https://github.com/jantic/DeOldify), and
2. Wan et al's joint restoration [here](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/tree/master/Global)# 8755-project
# 8755-project
# 8755-project
