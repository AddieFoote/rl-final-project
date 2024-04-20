#!/bin/bash

algorithms=('PPO' 'A2C' 'DDPG' 'DQN' 'HER' 'SAC' 'TD3')
envs=('one-hot' 'img')
policies=('CnnPolicy' 'MlpPolicy')


counter=0
for algorithm in "${algorithms[@]}"; do
    for env in "${envs[@]}"; do
        for policy in "${policies[@]}"; do
            tmux new -t "$counter-$algorithm-$env-$policy" -d
            tmux send-keys -t "$counter-$algorithm-$env-$policy" "conda activate rl-final" Enter
            tmux send-keys -t "$counter-$algorithm-$env-$policy" "python /Users/addie/Documents/GitHub/rl-final-project/testBabyaiEnv.py --env '$env' --algorithm '$algorithm' --policy '$policy'" Enter
            (( counter ++ ))
            sleep 1
        done
    done
done