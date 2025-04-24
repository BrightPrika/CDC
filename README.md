<p align="center">
    <h1 align="center">CycleGAN and Diffusion-Based Joint Correction for Over- and Under-Exposed
Images</h1>
    <p align="center">
        <a href="https://yiyulics.github.io/">Bright Prika</a>
</p>

<div align="center">


[![arxiv](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2405.17725)
[![cvf](https://img.shields.io/badge/Paper-CVF-%23357DBD)](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Color_Shift_Estimation-and-Correction_for_Image_Enhancement_CVPR_2024_paper.pdf)
[![LCDP](https://img.shields.io/badge/Dataset-LCDP-%23cda6c3)](https://github.com/onpix/LCDPNet/tree/main)
[![MSEC](https://img.shields.io/badge/Dataset-MSEC-%23cda6c3)](https://github.com/mahmoudnafifi/Exposure_Correction)
[![Pretrained Model](https://img.shields.io/badge/Pretrained-Model-%2380f69a)](https://drive.google.com/drive/folders/1SEQu3f2IdNnLlFH1OLUGyny5Xy-0TGzb?usp=sharing)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/color-shift-estimation-and-correction-for/image-enhancement-on-exposure-errors)](https://paperswithcode.com/sota/image-enhancement-on-exposure-errors?p=color-shift-estimation-and-correction-for)


</div>



**Abstract**: Real-world images often suffer from both over- and under-exposure, leading to color distortions and loss of detail. While existing methods address these issues separately, they struggle with scenes exhibiting simultaneous over- and under-exposed regions. Inspired by the success of CycleGAN and Diffusion Models in image-to-image translation and generative tasks, we propose a novel framework that leverages these techniques to correct exposure and color shifts. Our approach integrates a CycleGAN-inspired architecture to model bidirectional mappings between poorly exposed and well-exposed images, ensuring cycle consistency for robust enhancement. Additionally, we employ a Diffusion Model to iteratively refine the generated pseudo-normal features, capturing complex color distributions and mitigating artifacts. The proposed method consists of two key modules: a **Color Shift Estimation (COSE) module, which uses deformable convolutions extended to the color space to estimate and correct exposure-specific shifts, and a Color Modulation (COMO) module, which employs cross-attention to harmonize the enhanced regions. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches on benchmark datasets, achieving superior visual quality with fewer parameters. The integration of CycleGAN’s adversarial training and the Diffusion Model’s iterative refinement enables our framework to handle extreme exposure conditions effectively, even in the absence of reference normal-exposed pixels. Limitations arise in cases of fully saturated regions, suggesting future directions for incorporating generative priors to further improve robustness.


## :mega: News
- [2025/04/23] Update Google Drive link for the paper and README.
- [2025/04/23] Add environment yaml file.


## :wrench: Installation
To get started, clone this project, create a conda virtual environment using Python 3.9 (or higher versions may do as well), and install the requirements:
```
git clone https://github.com/BrightPrika/CDC.git
cd CDC

conda create -n cdc python=3.9
conda activate cdc

# Change the following line to match your environment
# Refer to https://pytorch.org/get-started/previous-versions/#v1121
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

pip install pytorch_lightning==1.7.6
pip install -r requirements.txt
```


## :computer: Running the Code

### Evaluation

To evaluate the trained model, you'll need to do the following steps:
- Get the [pretrained models](https://drive.google.com/drive/folders/1SEQu3f2IdNnLlFH1OLUGyny5Xy-0TGzb?usp=sharing) (or you can use your own trained weights) and put them in the `pretrained/` folder.
- Modify the path to the test dataset in `src/config/ds/test.yaml` (if you don't need ground truth images for testing, just leave the `GT` value as `none`).
- Run the following command:
    ```
    python src/test.py checkpoint_path=/path/to/checkpoint/filename.ckpt
    ```
- Then under the folder `/path/to/checkpoint/`, a new folder named `test_result/` will be created, and all the final enhanced images (`*.png` images) will be saved in this folder. Other intermediate results of each image will also be saved in the subfolders of `test_result/` (e.g., `test_result/normal/` for pseudo-normal images, etc.)


### Training

To train your own model from scratch, you'll need to do the following steps:
- Prepare the training dataset. You can use the [LCDP dataset](https://github.com/onpix/LCDPNet/tree/main) or [MSEC dataset](https://github.com/mahmoudnafifi/Exposure_Correction) (or you can use your own paired data).
- Modify the path to the training dataset in `src/config/ds/train.yaml`.
- Modify the path to the validation dataset in `src/config/ds/valid.yaml` (if have any).
- Run the following command:
    ```
    python src/train.py name=your_experiment_name
    ```
- The trained models and intermediate results will be saved in the `log/` folder.

#### OOM Errors

You may need to reduce the batch size in `src/config/config.yaml` to avoid out of memory errors. If you do this, but want to preserve quality, be sure to increase the number of training iterations and decrease the learning rate by whatever scale factor you decrease batch size by.



## :postbox: Citation
If you find our work helpful, please cite our paper as:
```
@inproceedings{li_2024_cvpr_csec,
    title       =   {Color Shift Estimation-and-Correction for Image Enhancement},
    author      =   {Yiyu Li and Ke Xu and Gerhard Petrus Hancke and Rynson W.H. Lau},
    booktitle   =   {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year        =   {2024}
}
```
Should you have any questions, feel free to post an issue or contact me at [yiyuli.cs@my.cityu.edu.hk](mailto:yiyuli.cs@my.cityu.edu.hk).


## :sparkles: Acknowledgements
The project is largely based on [LCDPNet](https://github.com/onpix/LCDPNet.git). Many thanks to the project for their excellent contributions!


