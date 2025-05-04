#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda --version

conda create -n traincheck-torch251 python=3.10 -y
conda activate traincheck-torch251

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121  # modify cu121 to cu118 if not supporting 121
conda install cudatoolkit -y

pip install git+https://github.com/OrderLab/TrainCheck.git

pip install -r requirements.txt
