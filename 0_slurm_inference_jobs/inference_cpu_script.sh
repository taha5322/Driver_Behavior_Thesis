#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --account=def-bauer
#SBATCH --mem-per-cpu=1024M

# Activate virtual environment
source /home/tsiddi5/myvirtualenv/bin/activate

yolo task=detect mode=predict model=$MODEL_PATH conf=0.25 source=$DATASET_PATH save=True project=$SAVE_PATH name=model_inference save_txt=True
