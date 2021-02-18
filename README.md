Thank you for visiting this repository

This repo is the artefact of the project _"Experiments on Restoring Color and Visual Fidelity to Legacy Photographs"_, undertaken by me and supervised by Dr.Salman Khan. Please contact me or Dr.Salman if you need the full paper.

### Introduction
With a multitude of applications, machine-automated image restoration tasks enjoy rapid and numerous research. For experimental purposes, one interesting problem domain lies in the restoration of random old and degraded photos, which are more difficult to generalize due to the possible variation in the degradation type and strength, and the possibility of compounding degradations (such as noise, scratch/torn, occlusion, low-resolution, etc).

While published image restoration models are usually benchmarked on datasets with similar degradation type, we will perform experiments on real old and degraded photos scraped from the internet.

### Results Summary
1. Attempt to induce multimodality in the produced colors, by implementing [this paper](https://arxiv.org/abs/1903.05628) on mode-seeking GAN regularization (Mao et al. 2019):

> <img src="https://github.com/rbdm/8755-project/blob/main/wiki/MSGAN_b.png" width=50%>

2. Attempt to enhance the colorfulness of known robust colorization training (DeOldify) by fine-tuning the GAN training:

> <img src="https://github.com/rbdm/8755-project/blob/main/wiki/CL.png" width=50%>

with highest colorfulness score:

> <img src="https://github.com/rbdm/8755-project/blob/main/wiki/colorfulness.png" width=50%>

3. We then implement some of the latest papers on image restoration with state-of-the-art results, choose the best performing one, and implement a pipeline for end-to-end restoration:

> <img src="https://github.com/rbdm/8755-project/blob/main/wiki/final.png" width=50%>

### Results Summary
In summary, our fine tuning result (2) is able to produce more colorful result than baseline model and the latest image colorization models, both qualitatively and quantitatively. Our multimodal colorization (1) is able to produce interesting and diverse colors, but we have yet to be able to fine-tune the model to produce realistic result. Our adopted end-to-end pipeline (3) shows the capability to generalize to varying restoration requirements.


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

3. Move both `.pth` files to `./models` directory

4. Move `FaceEnhancement-checkpoints.zip` to `./Face_Enhancement` directory, then extract it

5. Move `Global-checkpoints.zip` to `./Global` directory, then extract it

### Prepare dataset
1. Legacy-Large dataset can be downloaded from [here](https://drive.google.com/drive/folders/1nT7nfzqYbrffRwJWhdGFsGjH_wi6w-Pd?usp=sharing), at `Dataset` folder.

2. Alternatively, both training dataset and test dataset (legacy images) can be prepared by:
    - Run `image_scraper.ipynb`
    - Download ImageNet dataset from [here](http://image-net.org/download-images) for training
    - Run `preprocess.ipynb` to resize test images and create BW images
   
### Colorization
1. Best results for training and fine tuning can be viewed by running inference using `Colorizer_GANFineTune_bestmodel.pth`. Example inference is shown in `colorize_test.ipynb`

2. Results for Mode-Seeking GAN training can be viewed by running the `MSGAN_training.ipynb` file

### Joint with other restoration
The adopted final pipeline consists of preprocessing using Wan et al's (2020) method from [here](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/tree/master/Global) and feeding the result to the colorization model.

To test it, run `joint_restoration_test.ipynb`

### Calculate colorfulness
To calculate colorfulness using [Hasler et al's measurement](https://www.researchgate.net/publication/243135534_Measuring_Colourfulness_in_Natural_Images), run `calculate_colorfulness.ipynb`

### Acknowledgement

The final code implemented here are for research purposes only, and heavily based on the implementation from other repositories:
1. The original deoldify repository (Antic 2019) [here](https://github.com/jantic/DeOldify), and
2. The pre-colorization step are from Wan et al's joint restoration (2020) [here](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/tree/master/Global)
