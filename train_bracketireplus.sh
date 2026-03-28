#!/bin/zsh
echo "Start to train the model...."

dataroot="/data/Hongkai/datasets/Syn_Plus" 

device='gpu'        
name="origin"       

ckpt_dir="./checkpoint"
if [ ! -d "$ckpt_dir" ]; then
    mkdir -p "$ckpt_dir"
    echo "Created checkpoint directory: $ckpt_dir"
fi

build_dir="$ckpt_dir/$name"
if [ ! -d "$build_dir" ]; then
    mkdir -p "$build_dir"
    echo "Created experiment directory: $build_dir"
fi

LOG="$build_dir/`date +%Y-%m-%d-%H-%M-%S`.txt"
echo "Log file: $LOG"

echo "================================="
echo "Training Configuration:"
echo "Dataset: $dataroot"
echo "Experiment name: $name"
echo "Device: $device"
echo "Batch size: 36"
echo "Patch size: 96"
echo "================================="

# 执行训练
python train.py \
    --dataset_name bracketireplus \
    --model cat \
    --name $name \
    --lr_policy step \
    --patch_size 96 \
    --niter 400 \
    --save_imgs True \
    --lr 1e-4 \
    --dataroot $dataroot \
    --batch_size 36 \
    --print_freq 500 \
    --calc_metrics True \
    --weight_decay 0.01 \
    --gpu_ids $device \
    -j 8 \
    --lr_decay_iters 27 \
    --block Convnext \
    --load_optimizers False \
    --load_path "" \
    --load_iter [0] \
    2>&1 | tee "$LOG"