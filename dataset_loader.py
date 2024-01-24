import csv

import numpy as np
import os

import cv2
from torch.utils.data import Dataset
from PIL import Image

from utils import find_the_second_char_for_string
from torchvision import transforms


def default_loader(path, channel=3):
    """
    :param path: image path
    :param channel: # image channel
    :return: image
    """
    if channel == 1:
        return cv2.imread(path, 0)
        # print(path)
        # img1 = cv2.imread(path)
        # print(img1.shape)
        # return cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        # return Image.open(path).convert('YCbCr')
    else:
        assert (channel == 3)
        return cv2.imread(path)


class dataset_train_or_val_loader(Dataset):
    """
    训练或者验证的loader
    """

    # 发现在进来之前做分解再合并会有超过255的值，可以考虑在dataset中做分解合并，先不做
    def __init__(self, data_dir, transform=None, loader=default_loader):
        super(dataset_train_or_val_loader, self).__init__()
        self.transform = transform
        self.data_dir = data_dir
        self.org_map = data_dir + '/' + 'patches_ori'
        # self.label_map = data_dir + '/' + 'patches_jnd'
        # self.depth_map = data_dir + '/' + 'patches_depth'
        # self.seg_map = data_dir + '/' + 'patches_seg'
        # self.salient_map = data_dir + '/' + 'patches_salient'

        self.csv_train_path = os.path.dirname(os.path.dirname(data_dir)) + '/Text/shenvvc_ori_llm_text.csv'
        # print(self.csv_train_path)

        self.readDict = {}
        with open(self.csv_train_path, "r") as csv_file:
            reader = csv.reader(csv_file)
            self.readDict = dict(reader)

        self.files = []
        self.num_of_file = 0
        for _, _, files in os.walk(self.org_map):
            for file in files:
                if file.endswith('.bmp') or file.endswith('.png'):
                    if os.path.exists(os.path.join(self.org_map, file)):
                        self.num_of_file += 1
                        img_file = os.path.join(self.org_map, file)
                        # label_file = os.path.join(self.label_map, file)
                        # depth_file = os.path.join(self.depth_map, file)
                        # seg_file = os.path.join(self.seg_map, file)
                        # salient_file = os.path.join(self.salient_map, file)
                        # text = self.readDict[file.split('_')[0] + '.bmp'] if file.endswith('.bmp') else \
                        #     self.readDict[file.split('_')[0] + '.png']

                        idx = find_the_second_char_for_string(file, '_', 2)
                        text = self.readDict[file[:idx] + '.bmp'] if file.endswith('.bmp') else self.readDict[
                            file[:idx] + '.png']

                        self.files.append({
                            "image": img_file,
                            # "depth": depth_file,
                            # "seg": seg_file,
                            # "label": label_file,
                            # "salient": salient_file,
                            "text": text
                        })

        self.loader = loader

    def __len__(self):
        # return len(self.files)
        return self.num_of_file

    def __getitem__(self, item):
        datafiles = self.files[item]
        image = self.loader(datafiles["image"])  # datafiles["image"]保存的是路径
        # depth = self.loader(datafiles["depth"], 1)
        # seg = self.loader(datafiles["seg"], 1)
        # salient = self.loader(datafiles["salient"], 1)
        # depth = self.loader(datafiles["depth"])
        # seg = self.loader(datafiles["seg"])
        # salient = self.loader(datafiles["salient"])
        # label = self.loader(datafiles["label"])
        text = datafiles["text"]

        # label_res = (abs((image.astype(float) - label.astype(float))))
        # label = (image.astype(np.float) + label_res)
        # label = (image.astype(float) - label_res)
        # label[label < 0] = 0.
        # label[label > 255] = 255.
        # label = label.astype(np.uint8)

        if self.transform:
            image = self.transform(image)
            # depth = self.transform(depth)
            # label = self.transform(label)
            # seg = self.transform(seg)
            # salient = self.transform(salient)
        sample = {'image': image,
                  # 'label': label,
                  # 'depth': depth,
                  # 'salient': salient,
                  # 'seg': seg,
                  'text': text
                  }
        # sample = {'image': (image, datafiles["image"]), 'label': (label, datafiles["image"]),
        #           'depth': (depth, datafiles["image"]), 'salient': (salient, datafiles["image"]),
        #           'seg': (seg, datafiles["image"])}
        # 返回图像数据，以及image的文件名
        return sample, datafiles['image'].replace('\\', '/').split("/")[-1]
        # return sample, datafiles["image"]  # 返回图像数据，和原图路径


class dataset_test_loader(Dataset):
    """
        测试集loader(包括src的)
    """

    # 发现在进来之前做分解再合并会有超过255的值，可以考虑在dataset中做分解合并，先不做
    def __init__(self, data_dir, transform=None, loader=default_loader):
        super(dataset_test_loader, self).__init__()
        self.transform = transform
        self.data_dir = data_dir
        self.org_map = data_dir + '/' + 'patches_ori'
        # self.depth_map = data_dir + '/' + 'patches_depth'
        # self.seg_map = data_dir + '/' + 'patches_seg'
        # self.salient_map = data_dir + '/' + 'patches_salient'

        dataset = os.path.basename(os.path.normpath(data_dir))
        if 'Train' in dataset:
            self.csv_test_path = os.path.dirname(os.path.dirname(data_dir)) + '/Text/shenvvc_ori_llm_text.csv'
        else:
            self.csv_test_path = data_dir + dataset + '_text/' + dataset + '_ori_llm_text.csv'

        self.readDict = {}
        with open(self.csv_test_path, "r") as csv_file:
            reader = csv.reader(csv_file)
            self.readDict = dict(reader)

        self.files = []
        self.num_of_file = 0
        for _, _, files in os.walk(self.org_map):
            for file in files:
                if file.endswith('.bmp') or file.endswith('.png'):
                    self.num_of_file += 1
                    img_file = os.path.join(self.org_map, file)
                    # depth_file = os.path.join(self.depth_map, file)
                    # seg_file = os.path.join(self.seg_map, file)
                    # salient_file = os.path.join(self.salient_map, file)
                    text = self.readDict[file.rsplit('_', 1)[0] + '.bmp'] if file.endswith('.bmp') else self.readDict[
                        file.rsplit('_', 1)[0] + '.png']

                    self.files.append({
                        "image": img_file,
                        # "depth": depth_file,
                        # "seg": seg_file,
                        # "salient": salient_file,
                        "text": text
                    })

        self.loader = loader

    def __len__(self):
        # return len(self.files)
        return self.num_of_file

    def __getitem__(self, item):
        datafiles = self.files[item]
        image = self.loader(datafiles["image"])  # datafiles["image"]保存的是路径
        # depth = self.loader(datafiles["depth"])
        # seg = self.loader(datafiles["seg"])
        # salient = self.loader(datafiles["salient"])
        text = datafiles["text"]

        if self.transform:
            image = self.transform(image)
            # depth = self.transform(depth)
            # seg = self.transform(seg)
            # salient = self.transform(salient)

        sample = {'image': image,
                  # 'depth': depth,
                  # 'salient': salient,
                  # 'seg': seg,
                  'text': text
                  }
        # sample = {'image': (image, datafiles["image"]), 'label': (label, datafiles["image"]),
        #           'depth': (depth, datafiles["image"]), 'salient': (salient, datafiles["image"]),
        #           'seg': (seg, datafiles["image"])}
        # 返回图像数据，以及image的文件名
        return sample, datafiles['image'].replace('\\', '/').split("/")[-1]
        # return sample, datafiles["image"]  # 返回图像数据，和原图路径
