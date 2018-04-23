#!/bin/bash
#
#SBATCH --mem=40000
#SBATCH --job-name=sketch-model-run
#SBATCH --partition=m40-long
#SBATCH --output=log-%A.out
#SBATCH --error=log-%A.err
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anubhavsingh@umass.edu

# Log the jobid.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

# Run command
python main.py --train --data_dir "/home/anubhavsingh/690IV/Project/TrainingData/Airplane/" --train_dir "/home/anubhavsingh/690IV/Project/CheckpointTesting/Airplane/"