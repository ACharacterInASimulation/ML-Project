#!/bin/bash
#SBATCH --job-name=ml-project
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=20:00:00

module load cuda/12.1
source /apps/local/anaconda3/bin/activate mlproject

cd /home/badrinath.chandana/git/ACharacterInASimulation/ML-Project/ML-Project/

python3 train.py 

echo "Finished training at $(date)"
