import torch
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
import random
import numpy as np
import torch.backends.cudnn as cudnn
import argparse
from networks.MMDN import MMDN, Discriminator
from trainer.trainerMMDN import trainerMMDN
from torchtext.vocab import GloVe

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)  # if you are using multi-GPU.
cudnn.deterministic = True
cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(1234)

parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str, default='../train_datasets/', help='root dir for data')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.002, help='segmentation networks learning rate')
parser.add_argument('--seed', type=int, default=6666, help='random seed')
parser.add_argument('--model_name', type=str, default="MMDN", help='the name of the model you designed')
parser.add_argument('--version', type=str, default="G3", help='version G1-G5, the way of splitting datasets')
parser.add_argument('--mode', type=str, default="M1", help='mode M1-MN, the style of combining model')
parser.add_argument('--checkpoint', type=str, default="e100_b16_l002", help='this is checkpoint name that will be saved')
parser.add_argument('--save_per_epoch', type=int, default=5, help='this is checkpoint name that will be saved')
parser.add_argument('--gpu', type=int, default=0, help='this is checkpoint name that will be saved')

args = parser.parse_args()

if __name__ == "__main__":
    # 指定"模型"、"数据集"进行训练
    # os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    word_vec_dim = 300
    hidden_dim = 64
    glove = GloVe(name='6B', dim=word_vec_dim,
                  cache=r'D:\Desktop\JND_WORKS\MLLMJND\MLLMJND_ALL_CODE\MLLMJND_CODE\PMA\sentence_process')
    if args.mode == 'M1':
        # net = MLLMJND(glove=glove, img_size=224, word_vec_dim=word_vec_dim, image_vec_dim=300, stb_num=6, model_dim=96,
        #               window_size=4, num_head=2, patch_size=4, spatial_reduction=2)
        # print(trainerMLLMJND(args, net))
        net = MMDN(device=device, glove=glove, word_vec_dim=word_vec_dim, hidden_dim=hidden_dim, mode='train')
        # Discriminator
        dis = Discriminator(device=device, glove=glove, hidden_dim=hidden_dim, word_vec_dim=word_vec_dim)
        print(trainerMMDN(args, net))
    elif args.mode == 'M2':
        net = MMDN(device=device, glove=glove, word_vec_dim=word_vec_dim, hidden_dim=hidden_dim, mode='train')
        # Discriminator
        dis = Discriminator(device=device, glove=glove, hidden_dim=hidden_dim, word_vec_dim=word_vec_dim)
        print(trainerMMDN(args, net))
    elif args.mode == 'M3':
        net = MMDN(device=device, glove=glove, word_vec_dim=word_vec_dim, hidden_dim=hidden_dim, mode='train')
        # Discriminator
        dis = Discriminator(device=device, glove=glove, hidden_dim=hidden_dim, word_vec_dim=word_vec_dim)
        print(trainerMMDN(args, net))
    elif args.mode == 'M4':
        net = MMDN(device=device, glove=glove, word_vec_dim=word_vec_dim, hidden_dim=hidden_dim, mode='train')
        # Discriminator
        dis = Discriminator(device=device, glove=glove, hidden_dim=hidden_dim, word_vec_dim=word_vec_dim)
        print(trainerMMDN(args, net))
    elif args.mode == 'M5':
        net = MMDN(device=device, glove=glove, word_vec_dim=word_vec_dim, hidden_dim=hidden_dim, mode='train')
        # Discriminator
        dis = Discriminator(device=device, glove=glove, hidden_dim=hidden_dim, word_vec_dim=word_vec_dim)
        print(trainerMMDN(args, net))
    elif args.mode == 'M6':
        net = MMDN(device=device, glove=glove, word_vec_dim=word_vec_dim, hidden_dim=hidden_dim, mode='train')
        # Discriminator
        dis = Discriminator(device=device, glove=glove, hidden_dim=hidden_dim, word_vec_dim=word_vec_dim)
        print(trainerMMDN(args, net))
    elif args.mode == 'M7':
        net = MMDN(device=device, glove=glove, word_vec_dim=word_vec_dim, hidden_dim=hidden_dim, mode='train')
        # Discriminator
        dis = Discriminator(device=device, glove=glove, hidden_dim=hidden_dim, word_vec_dim=word_vec_dim)
        print(trainerMMDN(args, net))
    elif args.mode == 'M8':
        net = MMDN(device=device, glove=glove, word_vec_dim=word_vec_dim, hidden_dim=hidden_dim, mode='train')
        # Discriminator
        dis = Discriminator(device=device, glove=glove, hidden_dim=hidden_dim, word_vec_dim=word_vec_dim)
        print(trainerMMDN(args, net))