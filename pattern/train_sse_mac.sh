#!/bin/sh

exp="0-BMs-normal"
seed_max=1
bm="/Users/HAXRD/Desktop/1003/pattern/pattern/terrain-0.mat"
base_bm="/Users/HAXRD/Desktop/1003/pattern/pattern/terrain-0.mat"
world_len=1000.
granularity=31.25
n_ABS=10
n_GU=80
R_2D_NLoS=100
R_2D_LoS=200
episode_len=40.
num_base_env_episodes=1000
emulator_replay_size=10000
policy_replay_size=1000
num_planning_random=8192
num_planning_random_warmup=5000
num_planning_with_policy=512
top_k=64
planning_batch_size=128
num_base_emulator_episodes=100000
num_base_emulator_epochs=40
num_base_emulator_batch_size=128
num_env_episodes=100000
least_emulator_buffer_size=256
least_policy_buffer_size=256
num_train_policy=3
policy_batch_size=128
num_train_emulator=3
emulator_batch_size=128
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train_sse.py --experiment_name ${exp} --seed ${seed} --base_BMs_fname ${base_bm} --BMs_fname ${bm} --world_len ${world_len} --granularity ${granularity} --episode_length ${episode_len} --n_ABS ${n_ABS} --n_GU ${n_GU} --R_2D_NLoS ${R_2D_NLoS} --R_2D_LoS ${R_2D_LoS} --emulator_replay_size ${emulator_replay_size} --policy_replay_size ${policy_replay_size} --num_planning_random_warmup ${num_planning_random_warmup} --num_planning_random ${num_planning_random} --num_planning_with_policy ${num_planning_with_policy} --planning_top_k ${top_k} --planning_batch_size ${planning_batch_size} --num_base_emulator_episodes ${num_base_emulator_episodes} --num_base_emulator_epochs ${num_base_emulator_epochs} --num_base_emulator_batch_size ${num_base_emulator_batch_size} --num_env_episodes ${num_env_episodes} --least_emulator_buffer_size ${least_emulator_buffer_size} --least_policy_buffer_size ${least_policy_buffer_size} --num_train_policy ${num_train_policy} --policy_batch_size ${policy_batch_size} --num_train_emulator ${num_train_emulator} --emulator_batch_size ${emulator_batch_size} --num_base_env_episodes ${num_base_env_episodes} --use_emulator_pt --use_eval
done
# --use_emulator_pt
