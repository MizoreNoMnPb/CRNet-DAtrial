#-*- encoding: UTF-8 -*-
import time
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import numpy as np
import sys
import os
from PIL import Image
import torchvision.transforms as transforms

if __name__ == '__main__':
    opt = TestOptions().parse()
    dataset_test = create_dataset(opt.dataset_name, 'test', opt)
    dataset_size_test = len(dataset_test)
    print('The number of test images = %d' % dataset_size_test)

    # 使用改进后的模型
    opt.model = 'crnet_improved'
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    # 创建输出目录
    output_dir = os.path.join(opt.results_dir, opt.name, 'test_latest')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建注意力图输出目录
    attention_dir = os.path.join(output_dir, 'attention_maps')
    if not os.path.exists(attention_dir):
        os.makedirs(attention_dir)

    # 测试
    total_psnr = 0
    total_ssim = 0

    for i, data in enumerate(dataset_test):
        model.set_input(data)
        model.test()

        # 获取结果
        visuals = model.get_current_visuals()
        data_in = visuals['data_in']
        data_out = visuals['data_out']
        data_gt = visuals['data_gt']
        attention_map = visuals['attention_map']

        # 计算PSNR和SSIM
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim

        # 转换为numpy数组
        data_out_np = data_out.cpu().numpy().squeeze().transpose(1, 2, 0)
        data_gt_np = data_gt.cpu().numpy().squeeze().transpose(1, 2, 0)

        # 归一化到0-1
        data_out_np = np.clip(data_out_np, 0, 1)
        data_gt_np = np.clip(data_gt_np, 0, 1)

        # 计算PSNR和SSIM
        current_psnr = psnr(data_gt_np, data_out_np)
        current_ssim = ssim(data_gt_np, data_out_np, channel_axis=2)

        total_psnr += current_psnr
        total_ssim += current_ssim

        # 保存结果
        img_path = model.get_image_paths()
        img_name = os.path.basename(img_path[0])

        # 保存输入、输出和注意力图
        def save_image(tensor, path):
            tensor = tensor.cpu().squeeze()
            if tensor.dim() == 3:
                tensor = tensor.transpose(0, 1).transpose(1, 2)
            tensor = np.clip(tensor.numpy(), 0, 1) * 255
            tensor = tensor.astype(np.uint8)
            Image.fromarray(tensor).save(path)

        save_image(data_in, os.path.join(output_dir, 'input_' + img_name))
        save_image(data_out, os.path.join(output_dir, 'output_' + img_name))
        save_image(data_gt, os.path.join(output_dir, 'gt_' + img_name))

        # 保存注意力图
        attention_map = attention_map.cpu().squeeze()
        attention_map = (attention_map.numpy() * 255).astype(np.uint8)
        Image.fromarray(attention_map, mode='L').save(os.path.join(attention_dir, 'attention_' + img_name))

        print('Test [%d/%d] %s PSNR: %.4f SSIM: %.4f' % (i+1, dataset_size_test, img_name, current_psnr, current_ssim))

    # 计算平均PSNR和SSIM
    avg_psnr = total_psnr / dataset_size_test
    avg_ssim = total_ssim / dataset_size_test

    print('Average PSNR: %.4f' % avg_psnr)
    print('Average SSIM: %.4f' % avg_ssim)

    # 保存测试结果
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write('Average PSNR: %.4f\n' % avg_psnr)
        f.write('Average SSIM: %.4f\n' % avg_ssim)

    print('Test completed! Results saved to %s' % output_dir)
