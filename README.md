# [TCSVT'25] Semantics-Oriented Multitask Learning for DeepFake Detection: A Joint Embedding Approach (SJEDD)

This repository contains the official PyTorch implementation of the paper **"[Semantics-Oriented Multitask Learning for DeepFake Detection: A Joint Embedding Approach](https://ieeexplore.ieee.org/document/11010889)"** by Mian Zou, Baosheng Yu, Yibing Zhan, Siwei Lyu, and Kede Ma.

☀️ If you find this work useful for your research, please kindly star our repo and cite our paper! ☀️

- [x] Release [arXiv paper](https://arxiv.org/abs/2408.16305)
- [x] Release inference codes
- [x] Release checkpoints 
- [x] Release datasets
- [x] Release training codes

## 📁 Datasets
Follow the links below to download the datasets (🛡️ Copyright of the datasets belongs to their original providers, and you may be asked to fill out some forms before downloading):

|  [FF++](https://github.com/ondyari/FaceForensics) | [CDF(v2)](https://github.com/yuezunli/celeb-deepfakeforensics)| [FSh](https://github.com/ondyari/FaceForensics/blob/master/dataset/FaceShifter/README.md) | [DF-1.0](https://github.com/EndlessSora/DeeperForensics-1.0/tree/master) | 
|:-:|:-:|:-:|:-:|
| [FFSC](https://github.com/MZMMSEC/SO-DFD) | [DiffusionFace (excluding the purely synthesized faces)](https://github.com/Rapisurazurite/DiffFace)| [DiFF (excluding the purely synthesized faces)](https://github.com/xaCheng1996/DiFF) |[DFDC (test set of the full version, not the Preview)](https://ai.meta.com/datasets/dfdc/) |

**Note**: For FF++ and FFSC, please download the full dataset for both training and testing. For the other datasets, if a separate test set is explicitly provided, please download the test set. Otherwise, if no specific split is mentioned, download the full dataset. We also provide [download link](https://pan.baidu.com/s/1Otk8pNiGVXeF5o7ZIjO1NA?pwd=t8d5) for Diffusion-based Face Swapping, i.e., DiffusionFace and DiFF, in which the face images have been organized and pre-processed for direct use in testing.

### Preprocessing (see [instructions](https://github.com/MZMMSEC/SJEDD/tree/main/preprocessing))

1) Extract the frames, and then detect and crop the faces (Optional for video datasets)

2) Rearrange the data for the test experiments


## 🚀 Quick Start

### 1. Installation of base reqiurements
 - python == 3.8
 - PyTorch == 1.13
 - Miniconda
 - CUDA == 11.7

### 2. Download the pretrained model and our model

|      Model       |    Training Dataset   |                                                        Download                                                                | |
|:----------------:|:----------------:|:-------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------:|
| SJEDD | [FF++](https://github.com/ondyari/FaceForensics)  | [Google Drive](https://drive.google.com/drive/folders/17Wqxlf7hWbdC3WRljXryDK3VPE0bFoTp?usp=sharing) |✅|
| SJEDD | [FFSC](https://github.com/MZMMSEC/SO-DFD/tree/main)  | [Google Drive](https://drive.google.com/drive/folders/17Wqxlf7hWbdC3WRljXryDK3VPE0bFoTp?usp=sharing) |✅|

After downloading these checkpoints, put them into the folder ``pretrained``.

### 3. Inference on the test sets
**For FF++**
```
CUDA_VISIBLE_DEVICES=5 python SJEDD_test.py --auc --test_log True --dataset [e.g., CDF, FSh, Deeper, DFDC, FFSC] \
--batch_size 1 --n_frames 64 --resume [path to checkpoints, e.g., ./pretrained/ckpt_best.pth]
```
**For FFSC**
```
CUDA_VISIBLE_DEVICES=5 python SJEDD_test.py --auc --test_log True --dataset Deeper --datapath /data0/mian2/DeeperForensics/ \
--output ./output/test/SO_ViT/deeper/ --batch_size 1 --n_frames 32 --train_dataset ffsc \
--resume [path to checkpoints, e.g., ./pretrained/ckpt_best_FFSC.pth]
```

### 4. Training (For reference)
```
CUDA_VISIBLE_DEVICES=4 python SJEDD_train_FFpp.py --name lr6e-7_lambdaInit_1.0_lambdaLr1e-3_aug0.3_bz32 --aug --aug_probs 0.3 \
--batch_size 32 --num_epoch 20 --output ./output/train_process/autoL-FFppHalf-SO-unnorm-noAnton \
--txt_path_train [path to train list txt file] \
--txt_path_val [path to val list txt file] \
--weighting_method auto-l --task binary \
--weight autol --gpu 0 --autol_lr 1e-3 --autol_init 1.0 --initial_lr 6e-7
```

## 📜 License 
This project is released for **academic and research purposes only**, and is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

## Citation
If you find this repository useful in your research, please consider citing the following paper:
```
@article{zou2025sjedd,
  title={Semantics-oriented multitask learning for DeepFake detection: A joint embedding approach},
  author={Zou, Mian and Yu, Baosheng and Zhan, Yibing and Lyu, Siwei and Ma, Kede},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025}
}
@article{zou2025semantic,
  title={Semantic contextualization of face forgery: A new definition, dataset, and detection method},
  author={Zou, Mian and Yu, Baosheng and Zhan, Yibing and Lyu, Siwei and Ma, Kede},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2025},
  pages={4512-4524},
  vol={20}
}
```
