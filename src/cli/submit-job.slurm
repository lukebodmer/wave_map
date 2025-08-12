#!/bin/bash
#SBATCH --job-name=inverse-problems
#SBATCH --output=outputs/wave_sim_%A_%a.out
#SBATCH --error=outputs/wave_sim_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=50G
#SBATCH --partition=compute
#SBATCH --array=0-2

# Load modules
module load python/3.9 gcc/9.3.0

# Activate virtual environment
source venv/bin/activate

# Get the config files from the latin hypercube sampling
CONFIG_FILE="inputs/parameter_configs/config_$(printf '%03d' $SLURM_ARRAY_TASK_ID).toml"

# Run simulation
python run_simulation.py "$CONFIG_FILE"
