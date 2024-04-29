#!/bin/bash

#SBATCH --job-name=cnn_size
#SBATCH --output=babyai_test_%j.out
#SBATCH --error=babyai_test_%j.err
#SBATCH --partition=gpu-a100-dev
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH -A IRI24006
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=codyrushing@utexas.edu


# Load necessary modules (if required)
eval "$(conda shell.bash hook)"
conda activate rl_final

# Run your command
python testBabyaiEnv.py --env custom-dynamic --obs fully-observable --num-conv-layers 3 --num_env=16 --num-timesteps 8000000 --size 6 --reward-shaping True
#python testBabyaiEnv.py --env custom-dynamic --obs fully-observable --num-conv-layers 5 --num_env=32 --num-timesteps 3000000 --size 6 --reward-shaping True | WE ARE DOING THIS ONE RN
#python testBabyaiEnv.py --env custom-dynamic --obs fully-observable --num-conv-layers 8 --num_env=16 --num-timesteps 8000000 --size 6 --reward-shaping True
