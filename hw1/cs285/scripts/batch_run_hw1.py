import os

env = "Ant"
base_command = None

if env == "Ant":
    base_command = "python cs285/scripts/run_hw1.py \
        --expert_policy_file cs285/policies/experts/Ant.pkl \
        --env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
        --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
        --video_log_freq -1 --eval_batch_size 10000"


lr_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

for i,lr in enumerate(lr_list):
    command = base_command + f" -lr {lr}" + \
        f" >data/{env}-{i}.txt"
    os.system(command)
