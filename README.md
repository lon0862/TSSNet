# TSSNet

## Getting Started

Step 1: clone this repository:

- Initialize virtual environment:

	conda create -n TSSNet python=3.8
	conda activate TSSNet

- Install agoverse api. 

	cd ..
	git clone https://github.com/argoai/argoverse-api
	=> replace setup.py's sklearn with scikit-learn
	pip install -e argoverse-api(pip<24.1)
	=> python -m pip install pip==24.0
	cd TSSNet
	
	if numba failed without raising an exception
	=> pip install numba==0.56.4

	if SystemError: initialization of _internal failed without raising an exception
	=> pip uninstall numpy
	=> pip install numpy==1.23.4

- Install the [pytorch](https://pytorch.org/). 

	conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

- If conda install too long 
	
	conda update -n base conda
	conda install -n base conda-libmamba-solver
	conda config --set solver libmamba

- Install others

	pip install torch_geometric==2.3.1
	conda install pytorch-lightning==2.0.3

- load pretrained_results from proposals
	
	python load_h52pkl.py --pred_path_root pred_results_models \
	--split val --data_name av1 --model_name SIMPL

- Train
	python train.py --init_pred_root pred_results_models/pkl \
	--train_batch_size 16 --val_batch_size 16 \
	--max_epochs 24 --T_max 24 \
	--iter_num 2 --recurr_num 10

- Eval 
	python val.py --init_pred_root ./pred_results_models/pkl \
	--ckpt_path pretrained_checkpoints/av1_checkpoint.ckpt \
	--val_batch_size 16 --iter_num 2 --recurr_num 10

- Test 
	python test.py --init_pred_root ./pred_results_models/pkl \
	--ckpt_path pretrained_checkpoints/av1_checkpoint.ckpt \
	--test_batch_size 16 --iter_num 2 --recurr_num 10

- ITRI dataset
- ITRI2av1 format
	python itri2av1_track.py --split_type continuous
	python itri2av1_map.py

- load pretrained_results from proposals
	python load_h52pkl.py --pred_path_root pred_results_models_itri \
	--data_name itri --split_type continuous --obj_type car \
	--model_name SIMPL --split val 

- Train 
	python train.py 
	--root ../ITRI_Dataset/av1_dataset_continuous/car \
	--processed_root ../ITRI_Dataset/av1_processed_continuous/car \
	--init_pred_root ./pred_results_models_itri/pkl_continuous/car \
	--train_batch_size 16 --val_batch_size 16 \
	--max_epochs 24 --T_max 24 \
	--iter_num 2 --recurr_num 10
	

- Eval 
	python val.py --root ../ITRI_Dataset/av1_dataset_continuous/car \
	--processed_root ../ITRI_Dataset/av1_processed_continuous/car \
	--init_pred_root ./pred_results_models_itri/pkl_continuous/car \
	--ckpt_path pretrained_checkpoints/itri_checkpoint.ckpt \
	--val_batch_size 16 --iter_num 2 --recurr_num 10 
