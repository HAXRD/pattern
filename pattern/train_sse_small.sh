#!/bin/sh

exp="0-BMs"
seed_max=1
bm="./terrain-0.mat"
world_len=1000.
granularity=25.
n_ABS=2
n_GU=10
R_2D_NLoS=100
R_2D_LoS=200
episode_len=40.
num_emulator_warmup=100000
emulator_replay_size=1000000
policy_replay_size=100000
num_planning_random=32
num_planning_random_warmup=5
num_planning_with_policy=5
top_k=8
planning_batch_size=64
num_base_emulator_episodes=100
num_base_emulator_epochs=2
num_base_emulator_batch_size=64
num_env_episodes=100000
least_emulator_buffer_size=256
least_policy_buffer_size=256
num_train_policy=1
policy_batch_size=16
num_train_emulator=1
emulator_batch_size=16
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train_sse.py --experiment_name ${exp} --seed ${seed} --BMs_fname ${bm} --world_len ${world_len} --granularity ${granularity} --episode_length ${episode_len} --n_ABS ${n_ABS} --n_GU ${n_GU} --R_2D_NLoS ${R_2D_NLoS} --R_2D_LoS ${R_2D_LoS} --use_emulator_pt --num_emulator_warmup ${num_emulator_warmup} --emulator_replay_size ${emulator_replay_size} --policy_replay_size ${policy_replay_size} --num_planning_random_warmup ${num_planning_random_warmup} --num_planning_random ${num_planning_random} --num_planning_with_policy ${num_planning_with_policy} --planning_top_k ${top_k} --planning_batch_size ${planning_batch_size} --num_base_emulator_episodes ${num_base_emulator_episodes} --num_base_emulator_epochs ${num_base_emulator_epochs} --num_base_emulator_batch_size ${num_base_emulator_batch_size} --num_env_episodes ${num_env_episodes} --least_emulator_buffer_size ${least_emulator_buffer_size} --least_policy_buffer_size ${least_policy_buffer_size} --num_train_policy ${num_train_policy} --policy_batch_size ${policy_batch_size} --num_train_emulator ${num_train_emulator} --emulator_batch_size ${emulator_batch_size}
done
