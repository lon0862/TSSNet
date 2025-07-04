# TSSNet

## Getting Started in Argoverse 1

1\. Clone this repository:
```
https://github.com/lon0862/TSSNet.git
```

2\. Create a conda environment:
```
conda create -n TSSNet python=3.8
conda activate TSSNet
```

3\. Install the dependencies:
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install torch_geometric==2.3.1
conda install pytorch-lightning==2.0.3
```

4\. Download [Argoverse Motion Forecasting Dataset v1.1](https://www.argoverse.org/av1.html). After downloading and extracting the tar.gz files, the dataset directory should be organized as follows:
```
/path/to/dataset_root/
├── train/
|   └── data/
|       ├── 1.csv
|       ├── 2.csv
|       ├── ...
└── val/
    └── data/
        ├── 1.csv
        ├── 2.csv
        ├── ...
```

5\. Install [Argoverse 1 API](https://github.com/argoai/argoverse-api):
**Note**: Replace setup.py's sklearn with scikit-learn. pip should < 24.1, can use following code:
```
python -m pip install pip==24.0
```

6\. Load pretrained_results from proposals

**Note**: In this repository, you need to first download [h5 files](https://drive.google.com/drive/folders/1atzqDjC10pbXqZTqo_49GUzSuTdKXBNt?usp=sharing) of predictions for all combinations of **split** and **model_name**, and please make sure to run the following code for all combinations, too.
- **split** = [train, val, test]
- **model_name** = [HPNet, DGFNet, SmartRefine, SIMPL]
```
cd TSSNet
python load_h52pkl.py --pred_path_root pred_results_models --split val --data_name av1 --model_name SIMPL
```

(optional) Possible error and how to solve:
```
numba failed without raising an exception
=> pip install numba==0.56.4

SystemError: initialization of _internal failed without raising an exception
=> pip uninstall numpy
=> pip install numpy==1.23.4

If conda install too long
=> conda update -n base conda
=> conda install -n base conda-libmamba-solver
=> conda config --set solver libmamba
```

### Training 
```
python train.py --init_pred_root pred_results_models/pkl --train_batch_size 16 --val_batch_size 16 --max_epochs 24 --T_max 24 --iter_num 2 --recurr_num 10
```

### Evaluation
```
python val.py --init_pred_root ./pred_results_models/pkl --ckpt_path pretrained_checkpoints/av1_checkpoint.ckpt --val_batch_size 16 --iter_num 2 --recurr_num 10
```

### Testing 
```	
python test.py --init_pred_root ./pred_results_models/pkl --ckpt_path pretrained_checkpoints/av1_checkpoint.ckpt --test_batch_size 16 --iter_num 2 --recurr_num 10
```

### Quantitative Results

For this repository, the expected performance on Argoverse 1 validation set is:
| Models | minADE | minFDE | MR |
| :--- | :---: | :---: | :---: |
| TSSNet | 0.611 | 0.824 | 0.063 |

## Getting Started in ITRI dataset

We extra apply our model in [ITRI dataset](https://drive.google.com/drive/folders/1vpsz5rH1DYWPHQIJiQyh3-I6u0AapjZc?usp=sharing), which collect by Industrial Technology Research Institute in Taiwain.

1\. Preprocess ITRI to av1 format

**Note**: We split the ITRI dataset using a sliding window of size 1 and selected **car** as the target agent type.
```
python itri2av1_track.py --split_type continuous
python itri2av1_map.py
```

2\. Load pretrained_results from proposals

**Note**: In this repository, you need to first download [h5 files](https://drive.google.com/drive/folders/14eZkzi5JQYNUzOLtls4zw1N2nch-l3hB?usp=sharing) of predictions for all combinations of **split** and **model_name**, and please make sure to run the following code for all combinations, too.
- **split** = [train, val]
- **model_name** = [HPNet, SmartRefine]
```
python load_h52pkl.py --pred_path_root pred_results_models_itri --data_name itri --split_type continuous --obj_type car --model_name SIMPL --split val 
```

### Training
```
python train.py --root ../ITRI_Dataset/av1_dataset_continuous/car --processed_root ../ITRI_Dataset/av1_processed_continuous/car --init_pred_root ./pred_results_models_itri/pkl_continuous/car 
--train_batch_size 16 --val_batch_size 16 --max_epochs 24 --T_max 24 --iter_num 2 --recurr_num 10
```

### Evaluation
```
python val.py --root ../ITRI_Dataset/av1_dataset_continuous/car --processed_root ../ITRI_Dataset/av1_processed_continuous/car --init_pred_root ./pred_results_models_itri/pkl_continuous/car 
--ckpt_path pretrained_checkpoints/itri_checkpoint.ckpt --val_batch_size 16 --iter_num 2 --recurr_num 10
```
