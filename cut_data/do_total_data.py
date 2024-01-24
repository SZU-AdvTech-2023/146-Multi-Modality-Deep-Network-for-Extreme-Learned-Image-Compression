import argparse
import os
import sys
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
# python程序中使用 import XXX 时，python解析器会在当前目录、已安装和第三方模块中搜索 xxx，如果都搜索不到就会报错。将当前文件加到sys.path中可解决问题
import random
import numpy as np
import csv

import shutil
import torch
import numpy as np
import cv2
from tqdm import tqdm
import decimal
from decimal import Decimal
# from utils import reflesh_dir, cut_images, get_feature, fea2img, get_JND_wusun_patches, get_JND_patches
from utils import reflesh_dir, cut_images, remove_dir

context = decimal.getcontext()  # 获取decimal现在的上下文
context.rounding = decimal.ROUND_HALF_UP  # 修改rounding策略
parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default="G1", help='version G1-G5')
parser.add_argument('--size', type=int, default=256, help='split pic size')
args = parser.parse_args()
num_of_picture = 0
psnr_total = 0
lst = []
size = args.size
stride = size
# csv_path = './RESULT/result_teacher.csv'

mode = args.version
print("mode:", mode)
print("do:" + mode + "size:", size)

# 训练数据文件夹
train_dataset_file = "train_datasets"

row_path = "../" + train_dataset_file + "/" + mode + "/"
all_ori_dir = "../" + train_dataset_file + "/Ori/"
all_jnd_dir = "../" + train_dataset_file + "/Dis/"
all_seg_dir = "../" + train_dataset_file + "/Seg"
all_depth_dir = "../" + train_dataset_file + "/Depth"
all_salient_dir = "../" + train_dataset_file + "/Salient"

val_ori_dir = "../" + train_dataset_file + "/" + mode + "/" + "Val_data/src_ori"
val_jnd_dir = "../" + train_dataset_file + "/" + mode + "/" + "Val_data/src_jnd"
val_seg_dir = "../" + train_dataset_file + "/" + mode + "/" + "Val_data/src_seg"
val_depth_dir = "../" + train_dataset_file + "/" + mode + "/" + "Val_data/src_depth"
val_salient_dir = "../" + train_dataset_file + "/" + mode + "/" + "Val_data/src_salient"

test_ori_dir = "../" + train_dataset_file + "/" + mode + "/" + "Test_data/src_ori"
test_jnd_dir = "../" + train_dataset_file + "/" + mode + "/" + "Test_data/src_jnd"
test_seg_dir = "../" + train_dataset_file + "/" + mode + "/" + "Test_data/src_seg"
test_depth_dir = "../" + train_dataset_file + "/" + mode + "/" + "Test_data/src_depth"
test_salient_dir = "../" + train_dataset_file + "/" + mode + "/" + "Test_data/src_salient"

train_ori_dir = "../" + train_dataset_file + "/" + mode + "/" + "Train_data/src_ori"
train_jnd_dir = "../" + train_dataset_file + "/" + mode + "/" + "Train_data/src_jnd"
train_seg_dir = "../" + train_dataset_file + "/" + mode + "/" + "Train_data/src_seg"
train_depth_dir = "../" + train_dataset_file + "/" + mode + "/" + "Train_data/src_depth"
train_salient_dir = "../" + train_dataset_file + "/" + mode + "/" + "Train_data/src_salient"

val_ori_patches_dir = "../" + train_dataset_file + "/" + mode + "/" + "Val_data/patches_ori"
val_jnd_patches_dir = "../" + train_dataset_file + "/" + mode + "/" + "Val_data/patches_jnd"
val_seg_patches_dir = "../" + train_dataset_file + "/" + mode + "/" + "Val_data/patches_seg"
val_depth_patches_dir = "../" + train_dataset_file + "/" + mode + "/" + "Val_data/patches_depth"
val_salient_patches_dir = "../" + train_dataset_file + "/" + mode + "/" + "Val_data/patches_salient"

test_ori_patches_dir = "../" + train_dataset_file + "/" + mode + "/" + "Test_data/patches_ori"
test_jnd_patches_dir = "../" + train_dataset_file + "/" + mode + "/" + "Test_data/patches_jnd"
test_seg_patches_dir = "../" + train_dataset_file + "/" + mode + "/" + "Test_data/patches_seg"
test_depth_patches_dir = "../" + train_dataset_file + "/" + mode + "/" + "Test_data/patches_depth"
test_salient_patches_dir = "../" + train_dataset_file + "/" + mode + "/" + "Test_data/patches_salient"

train_ori_patches_dir = "../" + train_dataset_file + "/" + mode + "/" + "Train_data/patches_ori"
train_jnd_patches_dir = "../" + train_dataset_file + "/" + mode + "/" + "Train_data/patches_jnd"
train_seg_patches_dir = "../" + train_dataset_file + "/" + mode + "/" + "Train_data/patches_seg"
train_depth_patches_dir = "../" + train_dataset_file + "/" + mode + "/" + "Train_data/patches_depth"
train_salient_patches_dir = "../" + train_dataset_file + "/" + mode + "/" + "Train_data/patches_salient"

# 训练数据、验证数据、测试数据的拆分以及训练数据、验证数据的分块(不含测试数据的分块)
train_val_finished = True

if train_val_finished is False:
    reflesh_dir(val_ori_dir)
    reflesh_dir(val_jnd_dir)
    reflesh_dir(val_seg_dir)
    reflesh_dir(val_depth_dir)
    reflesh_dir(val_salient_dir)

    reflesh_dir(test_ori_dir)
    reflesh_dir(test_jnd_dir)
    reflesh_dir(test_seg_dir)
    reflesh_dir(test_depth_dir)
    reflesh_dir(test_salient_dir)

    reflesh_dir(train_ori_dir)
    reflesh_dir(train_jnd_dir)
    reflesh_dir(train_seg_dir)
    reflesh_dir(train_depth_dir)
    reflesh_dir(train_salient_dir)

    reflesh_dir(val_ori_patches_dir)
    reflesh_dir(val_jnd_patches_dir)
    reflesh_dir(val_seg_patches_dir)
    reflesh_dir(val_depth_patches_dir)
    reflesh_dir(val_salient_patches_dir)

    reflesh_dir(train_ori_patches_dir)
    reflesh_dir(train_jnd_patches_dir)
    reflesh_dir(train_seg_patches_dir)
    reflesh_dir(train_depth_patches_dir)
    reflesh_dir(train_salient_patches_dir)

    # 切分训练集:验证集:测试集的比例
    if mode == "G1":
        # 训练集:验证集:测试集=8:1:1
        train_val_test_ratio = '8:1:1'
    elif mode == "G2":
        train_val_test_ratio = '8:1:1'
    elif mode == "G3":
        train_val_test_ratio = '8:1:1'
    elif mode == "G4":
        train_val_test_ratio = '8:1:1'
    else:
        train_val_test_ratio = '8:1:1'

    # 划分数据train val test
    train_ratio, val_ratio, test_ratio = list(map(int, train_val_test_ratio.split(":")))
    files = os.listdir(all_ori_dir)
    # 只取50张图像
    files = files[:50]
    # print(len(files))
    # sys.exit()
    train_val_files, test_files = train_test_split(files, test_size=test_ratio / (train_ratio + val_ratio + test_ratio))
    train_files, val_files = train_test_split(train_val_files, test_size=val_ratio / (train_ratio + val_ratio))

    num_of_train = len(train_files)
    num_of_val = len(val_files)
    num_of_test = len(test_files)

    # train files
    loop = tqdm(range(len(train_files)), total=len(train_files))
    for i, file in enumerate(train_files):
        ori_path = os.path.join(all_ori_dir, file)
        jnd_path = os.path.join(all_jnd_dir, file)
        seg_path = os.path.join(all_seg_dir, file.replace('png', 'bmp'))
        depth_path = os.path.join(all_depth_dir, file.replace('png', 'bmp'))
        salient_path = os.path.join(all_salient_dir, file.replace('png', 'bmp'))

        train_ori_path = os.path.join(train_ori_dir, file)
        train_jnd_path = os.path.join(train_jnd_dir, file)
        train_seg_path = os.path.join(train_seg_dir, file)
        train_depth_path = os.path.join(train_depth_dir, file)
        train_salient_path = os.path.join(train_salient_dir, file)

        shutil.copyfile(ori_path, train_ori_path)
        shutil.copyfile(jnd_path, train_jnd_path)
        shutil.copyfile(seg_path, train_seg_path)
        shutil.copyfile(depth_path, train_depth_path)
        shutil.copyfile(salient_path, train_salient_path)

        loop.set_description(f'Train datasets copy [{i+1}/{len(train_files)}]')
        loop.update(1)

    # val files
    loop = tqdm(range(len(val_files)), total=len(val_files))
    for i, file in enumerate(val_files):
        ori_path = os.path.join(all_ori_dir, file)
        jnd_path = os.path.join(all_jnd_dir, file)
        seg_path = os.path.join(all_seg_dir, file)
        depth_path = os.path.join(all_depth_dir, file)
        salient_path = os.path.join(all_salient_dir, file)

        val_ori_path = os.path.join(val_ori_dir, file)
        val_jnd_path = os.path.join(val_jnd_dir, file)
        val_seg_path = os.path.join(val_seg_dir, file)
        val_depth_path = os.path.join(val_depth_dir, file)
        val_salient_path = os.path.join(val_salient_dir, file)

        shutil.copyfile(ori_path, val_ori_path)
        shutil.copyfile(jnd_path, val_jnd_path)
        shutil.copyfile(seg_path, val_seg_path)
        shutil.copyfile(depth_path, val_depth_path)
        shutil.copyfile(salient_path, val_salient_path)

        loop.set_description(f'Val datasets copy [{i + 1}/{len(val_files)}]')
        loop.update(1)

    # test files
    loop = tqdm(range(len(val_files)), total=len(test_files))
    for i, file in enumerate(test_files):
        ori_path = os.path.join(all_ori_dir, file)
        jnd_path = os.path.join(all_jnd_dir, file)
        seg_path = os.path.join(all_seg_dir, file)
        depth_path = os.path.join(all_depth_dir, file)
        salient_path = os.path.join(all_salient_dir, file)

        test_ori_path = os.path.join(test_ori_dir, file)
        test_jnd_path = os.path.join(test_jnd_dir, file)
        test_seg_path = os.path.join(test_seg_dir, file)
        test_depth_path = os.path.join(test_depth_dir, file)
        test_salient_path = os.path.join(test_salient_dir, file)

        shutil.copyfile(ori_path, test_ori_path)
        shutil.copyfile(jnd_path, test_jnd_path)
        shutil.copyfile(seg_path, test_seg_path)
        shutil.copyfile(depth_path, test_depth_path)
        shutil.copyfile(salient_path, test_salient_path)

        loop.set_description(f'Test datasets copy [{i + 1}/{len(test_files)}]')
        loop.update(1)

    print("num_of_Train:", num_of_train)
    print("num_of_Val:", num_of_val)
    print("num_of_Test:", num_of_test)

    # 至此，已分好集合，接下来是划分
    # train的分块，直接分，不重叠
    print("cut for train:")
    count = 0
    count += cut_images(train_ori_dir, train_ori_patches_dir, size, stride, "image", num_of_train, drop=True)
    count += cut_images(train_jnd_dir, train_jnd_patches_dir, size, stride, "jnd", num_of_train, drop=True)
    count += cut_images(train_seg_dir, train_seg_patches_dir, size, stride, "seg", num_of_train, drop=True)
    count += cut_images(train_depth_dir, train_depth_patches_dir, size, stride, "depth", num_of_train, drop=True)
    count += cut_images(train_salient_dir, train_salient_patches_dir, size, stride, "salient", num_of_train, drop=True)

    print('number of train patches:', count)
    # print('number of no fit:', fail_count)
    print('cut train finished!')

    # cut for val 直接分，不重叠
    print("cut for val:")
    count = 0
    count += cut_images(val_ori_dir, val_ori_patches_dir, size, stride, "image", num_of_val, drop=True)
    count += cut_images(val_jnd_dir, val_jnd_patches_dir, size, stride, "jnd", num_of_val, drop=True)
    count += cut_images(val_seg_dir, val_seg_patches_dir, size, stride, "seg", num_of_val, drop=True)
    count += cut_images(val_depth_dir, val_depth_patches_dir, size, stride, "depth", num_of_val, drop=True)
    count += cut_images(val_salient_dir, val_salient_patches_dir, size, stride, "salient", num_of_val, drop=True)

    print('number of val patches:', count)
    print('cut val finished!')


# 在已经划分好测试集的基础上，对测试集进行图片分块
# #去sh cut_G
import subprocess

subprocess.call('.\\cut_G.sh', shell=True) # windows下
# subprocess.call('./cut_G.sh', shell=True)  # 服务器上
# os.system('./cut_G.sh')

#
# #做test的label
# print("做" + mode + "的label,并且复制到final")
nums_of_test_patches = 0
for _, _, files in os.walk(test_ori_patches_dir):
    for file in files:
        if file.endswith(".bmp") or file.endswith(".png"):
            nums_of_test_patches += 1

test_label_patches_num = 0
with tqdm(total=nums_of_test_patches) as pbar:
    for _, _, files in os.walk(test_ori_patches_dir):
        for file in files:
            # print(len(files))
            # if file.endswith(".bmp"):
            image_file = os.path.join(test_ori_patches_dir, file)
            label_file = os.path.join(test_jnd_patches_dir, file)
            seg_file = os.path.join(test_seg_patches_dir, file)
            depth_file = os.path.join(test_depth_patches_dir, file)
            salient_file = os.path.join(test_salient_patches_dir, file)

            # image = cv2.imread(image_file)
            # seg = cv2.imread(seg_file)
            # depth = cv2.imread(depth_file)
            if os.path.exists(label_file):
                # label = cv2.imread(label_file)
                test_label_patches_num += 1
                # shutil.copyfile(label_file, os.path.join(final_test_patches_jnd, file))

            # shutil.copyfile(image_file, os.path.join(final_test_patches_image, file))
            # shutil.copyfile(seg_file, os.path.join(final_test_patches_seg, file))
            # shutil.copyfile(depth_file, os.path.join(final_test_patches_depth, file))
            # shutil.copyfile(salient_file, os.path.join(final_test_patches_salient, file))
            # pbar.set_description(desc="test patches直接保存%s " % (file))
            pbar.update(1)

print('number of test label patches:', test_label_patches_num)
# print('number of no fit:', fail_count)
print('cut test finished!')
print('begin clean...')

# 删除中间数据
# remove_dir(val_ori_patches_dir)
# remove_dir(val_jnd_patches_dir)
# remove_dir(val_seg_patches_dir)
# remove_dir(val_depth_patches_dir)
# remove_dir(val_salient_patches_dir)


# remove_dir(test_ori_patches_dir)
# remove_dir(test_jnd_patches_dir)
# remove_dir(test_seg_patches_dir)
# remove_dir(test_depth_patches_dir)
# remove_dir(test_salient_patches_dir)


# remove_dir(train_ori_patches_dir)
# remove_dir(train_jnd_patches_dir)
# remove_dir(train_seg_patches_dir)
# remove_dir(train_depth_patches_dir)
# remove_dir(train_salient_patches_dir)
