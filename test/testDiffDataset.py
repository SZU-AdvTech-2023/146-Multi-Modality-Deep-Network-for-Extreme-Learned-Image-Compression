import sys
import os

from torchvision.models import ResNet50_Weights

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from networks.MMDN import MMDN, Discriminator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from utils import get_parent_dir
from tqdm import tqdm
import argparse
import numpy as np
import torch
from torch import nn
import time
import datetime
from torch.utils.data import DataLoader
from torchvision import transforms, models
from networks.MLLMJND import MLLMJND
import cv2
import shutil
from dataset_loader import dataset_test_loader
import decimal
from load_checkpoint import Fusion_load
from torchtext.vocab import GloVe

context = decimal.getcontext()  # 获取decimal现在的上下文
context.rounding = decimal.ROUND_HALF_UP  # 修改rounding策略

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../test_datasets/', help='root dir for data')
parser.add_argument('--res_path', type=str, default='../Noise_results/', help='root dir for data')
parser.add_argument('--model', type=str, default="MMDN", help='model name')
parser.add_argument('--version', type=str, default="G1", help='version G1-G5')
parser.add_argument('--mode', type=str, default="M1", help='mode M1-MN')
parser.add_argument('--dataset', type=str, default="csiq/", help='switch dataset')
parser.add_argument('--checkpoint', type=str, default="rgb_only_mse", help='switch dataset')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--gpu', type=int, default=0, help='this is checkpoint name that will be saved')
args = parser.parse_args()

if __name__ == "__main__":
    # 使用指定‘模型文件’，对于指定‘数据集’进行测试，得到结果图infer_patches后，再合成完整的图片noise_img
    # os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'
    torch.cuda.set_device(args.gpu)
    total_start = time.time()
    # model_name = 'CmJND_mae_mse'
    # model_name = 'rgb_only_mae_mse'
    # model_name = 'rgb_only_mse'

    # root = "../diffDataset/" + args.dataset
    root = os.path.join(args.root_path, args.dataset)
    res_path = args.res_path
    pre_path = "./checkpoint/" + args.model + "/" + args.version + "/" + args.mode + "/" + args.checkpoint + ".pth"

    word_vec_dim = 300
    hidden_dim = 64
    glove = GloVe(name='6B', dim=word_vec_dim,
                  cache=r'./preprocess/sentence_process')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    net = MMDN(device=device, glove=glove, word_vec_dim=word_vec_dim, hidden_dim=hidden_dim, mode='test')
    # Discriminator
    # dis = Discriminator(device=device, glove=glove, hidden_dim=hidden_dim, word_vec_dim=word_vec_dim)
    # 创建ResNet-50模型实例
    # dis = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).cuda()
    #     # num_features = dis.fc.in_features
    #     # dis.fc = nn.Linear(num_features, 1)
    #     # dis.fc = nn.Sequential(
    #     #     nn.Linear(num_features, 2),
    #     #     nn.Sigmoid()
    #     # )

    print("load " + pre_path + "... testing " + args.dataset.replace('/', ''))
    loader = Fusion_load(model=net, checkpoint=pre_path)
    net, flag = loader.load_checkpoint()
    if not flag:
        print("Failed to load checkpoint, because checkpoint path may be false!")
        exit(0)
    net = net.cuda()
    # net = nn.DataParallel(net)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    test_dir = root
    db_test = dataset_test_loader(data_dir=test_dir, transform=transform)
    print("The length of Test set is: {}".format(len(db_test)))
    batch_size = args.batch_size
    test_loader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=2)

    psnr_count, psnr_total = 0, 0
    ssim_count, ssim_total = 0, 0
    mse_count, mse_total = 0, 0
    test_data_dir = test_dir + 'patches_ori'
    csv_dir = test_data_dir
    test_infer_patches_save_dir = test_dir + 'infer_patches'
    if os.path.exists(test_infer_patches_save_dir):
        shutil.rmtree(test_infer_patches_save_dir)
    os.mkdir(test_infer_patches_save_dir)
    num_of_bmp = 0
    print("test_data_dir:", test_data_dir)
    for _, _, files in os.walk(test_data_dir):
        for file in files:
            if file.endswith('.bmp'):
                num_of_bmp += 1
    csv_num = 0
    for _, _, files in os.walk(csv_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_num += 1
                csv_file = os.path.join(csv_dir, file)
                to_file = os.path.join(test_infer_patches_save_dir, file)
                shutil.copyfile(csv_file, to_file)
    print("num_of_test_patches is: ", num_of_bmp)
    print("num_of_csv is: ", csv_num)
    big_count = 0
    net.eval()
    with torch.no_grad():
        with tqdm(total=num_of_bmp) as pbar:
            for sampled_batch, files in test_loader:
                # image, depth, salient, seg, text = sampled_batch['image'], sampled_batch['depth'], sampled_batch[
                #     'salient'], \
                #     sampled_batch['seg'], sampled_batch['text']
                # image, depth, salient, seg = image.cuda(), depth.cuda(), salient.cuda(), seg.cuda()

                image, text = sampled_batch['image'], sampled_batch['text']
                image = image.cuda()

                # output = net(image, salient, depth, seg, text)

                cf_1, cf_2, outputs, l_it, encoded_out_img, encoded_ori_img = net(image, text)

                # image = image.squeeze()
                # outputs = outputs.squeeze()
                # outputs = (outputs + 1.) / 2.0
                # outputs.clamp_(0., 1.)
                # image = (image + 1.) / 2.0
                # image.clamp_(0., 1.)

                # image = torch.round((image + 1) * 127.5)
                outputs = outputs + image
                # outputs = torch.clamp(outputs, -1.0, 1.0)
                outputs = torch.clamp((outputs + 1) * 127.5, 0, 255)
                outputs = torch.round(outputs).float()

                v_mse = torch.mean((outputs - image) ** 2, [1, 2, 3])
                v_psnr = torch.mean(20 * torch.log10(255 / torch.sqrt(v_mse)), 0).item()
                v_mse = torch.mean(v_mse).item()

                # len_of_outputs = len(outputs)
                for i in range(0, len(files)):
                    save_file = os.path.join(test_infer_patches_save_dir, files[i])
                    # if len(outputs.shape) != 4:
                    #     out = outputs.cpu().numpy() * 255
                    # else:
                    #     out = outputs[i].cpu().numpy() * 255
                    out = outputs[i].cpu().numpy()
                    # out = outputs[i].cpu().numpy() * 255
                    # out[out > 255] = 255
                    # out[out < 0] = 0
                    out = out.astype(np.uint8())
                    out = out.transpose(1, 2, 0)
                    cv2.imwrite(save_file, out)
                    pbar.set_description(desc="imwrite %s:" % save_file)
                    pbar.update(1)

    print("v_mse =", v_mse, ", v_psnr =", v_psnr)
    print("Finish inferring! Begin to combine patch images!")

    # 将测试得到的最终结果即一块块图片组合恢复成一开始的图片，保存到test_infer_patches_combine_save_dir
    import subprocess

    test_infer_patches_combine_save_dir = res_path + args.version + '/' + args.mode + '/' + args.checkpoint + '/Noise_' + args.model + '_' + args.dataset.replace(
        '/', '')

    print("--origin:", test_infer_patches_save_dir)
    print("--target:", test_infer_patches_combine_save_dir)

    child = subprocess.Popen(
        ['python', './cut_data/combine_patches.py', '--origin', test_infer_patches_save_dir, '--target',
         test_infer_patches_combine_save_dir])

    child.wait()
    times = time.time() - total_start
    fpstime = times / 30
    times = str(datetime.timedelta(seconds=times))
    fpstime = str(datetime.timedelta(seconds=fpstime))
    print('total pictures uses:  %s  , FPS uses:  %s' % (times, fpstime))