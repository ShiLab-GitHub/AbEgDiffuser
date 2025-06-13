# AbEgDiffuser
This repository contains the data and python script in support of the manuscript: AbEgDiffuser: Antibody sequence-structure co-design with equivariant graph neural networks and diffusion models.
_ _ _ _

## 1.  Requirement

### 1) System requirements
This tool is supported for Linux.

### 2) Hardware requirements
We ran the demo using the following specs:

+ NVIDIA A800
+ CUDA 11.8

### 3) Environment
We have provided the environment.yml, it is recommended to create a Conda environment and install the necessary dependencies by following these steps:
```bash
conda env create -f environment.yaml -n Ab
conda activate Ab
```

### 4) [Optional] PyRosetta

PyRosetta is required to relax the generated structures and compute binding energy. Please follow the instruction [**here**](https://www.pyrosetta.org/downloads) to install.

## 2.  How to train and use AbEgDiffuser

### 1) Dataset preparation and pre-trained models
1. We use SAbDab for training and RAbD for testing. 
+ Protein structures in the `SAbDab` dataset can be downloaded [**here**](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/). Extract `all_structures.zip` into the `data` folder.
+ The `data` folder contains a snapshot of the dataset index (`sabdab_summary_all.tsv`). You may replace the index with the latest version [**here**](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all/).
+ The file `data/RAbD_test.idx` contains the RAbD test data information. 

2. Download the **ESM2** model weights from [here](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt) and the **contact regressor** weights from [here](https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt), and save these files in the `./trained_models/esm` directory.

### 2) Training the AbEgDiffuser model
If you use 4 GPUs for training, run the following command:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py ./configs/train/codesign_single.yml
```
If only one GPU is used for training, run the following command:
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train.py ./configs/train/codesign_single.yml
```

### 3) Co-design of CDRs
To perform co-design of CDRs using the RAbD test dataset, run the following command:
```bash
python design_all_test.py 60 --config ./configs/test/codesign_single.yml --seed 2025 --batch_size 16
```

### 4) Optimization
To optimize CDRs in the RAbD test dataset, run the following command:
```bash
python design_all_test.py 60 --config ./configs/test/abopt_singlecdr.yml --seed 2025 --batch_size 16
```

### 5) Design CDRs given antibody-antigen complex
Taking `7DK2_AB_C` antibody-antigen complex as an example, to generate CDRs of given antibdody-antigen complexes, run the following command:
```bash
python design_pdb.py ./data/examples/7DK2_AB_C.pdb --config ./configs/test/codesign_single.yml
```

### 6) Relaxing the designed proteins
To relax the designed proteins using PyRosetta, run the following command:
```bash
python -m AbEgDiffuser.tools.relax.run --root './results/codesign_single/'
```

### 7) Evaluation
To compute the evaluation metrics: AAR, RMSD, and IMP for all samples, run the following command:
```bash
python -m AbEgDiffuser.tools.eval.run --root './results/codesign_single/'
```

## 3. Model availability
The trained AbEgDiffuser model is available [**here** (zenodo)](https://doi.org/10.5281/zenodo.15512361). To use it, please download the model checkpoint and save it into the `./trained_models` folder. 

## 4. Citation
Please cite our paper if you use the code.

## 5. Contact
If you have any questions, please contact us via email: 
+ [Xiumin Shi](sxm@bit.edu.cn)
