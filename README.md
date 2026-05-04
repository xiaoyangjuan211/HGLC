# HGLC

Quick Start Environment setup for DGLNet 

## Environment

Python 3.8 + Torch 2.0.0 + CUDA 11.8  
conda create --name HGLC python=3.8  
conda activate HGLC

## Hardware:

single RTX 5060 GPU

## Dataset

Since JAFFE and RAF-DB and FERPLUS are restricted by their respective licenses, please apply for and download the datasets from their official websites.

**Please place the dataset as follows:**

Dataset/JAFFE/
Dataset/RAF-DB/
Dataset/FERPLUS/

## Install dependencies

cd HGLC  
pip install -r requirements.txt

### JAFFE Dataset

bash Train/TrainHGLC_JAFFE.sh

### RAF-DB Dataset

bash Train/TrainHGLC_RAFDB.sh

### FERPLUS Dataset

bash Train/TrainHGLC_FERPLUS.sh