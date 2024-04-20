#!/bin/bash

algorithms=('PPO' 'A2C' 'DQN')
envs=('one-hot' 'img' 'fully-observable')
# envs=('fully-observable')
policies=('CnnPolicy' 'MlpPolicy')


counter=100
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