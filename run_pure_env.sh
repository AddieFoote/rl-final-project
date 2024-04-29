#!/bin/bash

#SBATCH --job-name=pure_env
#SBATCH --output=babyai_test_%j.out
#SBATCH --error=babyai_test_%j.err
#SBATCH --partition=gpu-a100-small
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH -A IRI24006
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=addiefoote@utexas.edu



# Load necessary modules (if required)
eval "$(conda shell.bash hook)"
conda activate rl-final

# Run your command
python $HOME/rl-final-project/testBabyaiEnv.py --env custom-dynamic --obs fully-observable --num-conv-layers 5 --num_env=32 --num-timesteps 10000000 --size 5 
python $HOME/rl-final-project/testBabyaiEnv.py --env custom-dynamic --obs fully-observable --num-conv-layers 5 --num_env=32 --num-timesteps 10000000 --size 6 
python $HOME/rl-final-project/testBabyaiEnv.py --env custom-dynamic --obs fully-observable --num-conv-layers 5 --num_env=32 --num-timesteps 10000000 --size 7 
python $HOME/rl-final-project/testBabyaiEnv.py --env custom-dynamic --obs fully-observable --num-conv-layers 5 --num_env=32 --num-timesteps 10000000 --size 9