#!/bin/bash
#SBATCH -J gpu-job            # 任务名字是 gpu-job
#SBATCH --cpus-per-task=4     # 申请 8 个 cpu 核心，这里防止内存不够必须增加 CPU 核心数
#SBATCH -p gpu                # 任务提交到 gpu 分区，必须填写，只有此分区的机器才有 GPU 卡
#SBATCH --gres=gpu:1          # 申请 1 块 GPU 卡，必须填写，SLURM 默认是不分配 GPU 的
#SBATCH -t 3-00:00:00         # 最长运行时间是 1 天

python train.py            # 运行 python 程序
