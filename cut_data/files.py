import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
import shutil
import cv2
import numpy as np
import csv
from tqdm import tqdm


def load_img_float(path):
    """
    return [0, 1]
    """
    img = cv2.imread(path)
    # print(path)
    if img.shape[2] >= 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img / 255


def save_img_float(path, img):
    """
    input [0, 1]
    """
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def load_img_npy(path):
    """
    input [0, 1] and *.npy
    return [0, 1]
    """
    img = np.load(path)
    return img


def save_img_npy(path, img):
    """
    input [0, 1] and *.npy
    """
    np.save(path, img)


def debug_save_img_float(img):
    """
    path='./debug_tmp.bmp'
    """
    save_img_float('./debug_tmp.bmp', img)


def get_img_mask(img, th=0.05):
    """
    对颜色向量求和, 然后像素和小于th的判定为背景
    """
    np_img = np.array(img)
    np_sum = np_img.sum(axis=2)
    return np_sum < th


def set_img_by_mask(img, mask):
    img[mask] = 0
    return img


def mask_to_img(mask):
    img = np.array(mask).astype('float64')
    return img


def img_to_mask(img):
    mask = img != 0
    return mask


def check_folder(dir):
    """rasie FileNotFoundError"""
    if not os.path.isdir(dir):
        raise FileNotFoundError(dir + ' is not exist!')


def check_file(file):
    """rasie FileNotFoundError"""
    if not os.path.exists(file):
        raise FileNotFoundError(file + ' is not exist!')


def init_folder(dir):
    if not os.path.isdir(dir):
        try:
            os.mkdir(dir)
            print('Folder init: ' + dir)
        except FileNotFoundError:
            os.makedirs(dir)
            print('Folders init: ' + dir)


def refresh_folder(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    init_folder(dir)


def copy_files(dir_ori, dir_tar, list_files=None):
    if list_files is None:
        list_files = os.listdir(dir_ori)
    num_files = len(list_files)
    with tqdm(total=num_files, desc='Copy files') as bar:
        for file in list_files:
            shutil.copyfile(os.path.join(dir_ori, file),
                            os.path.join(dir_tar, file))
            bar.update(1)


def copy_file(dir_ori, dir_tar, file):
    shutil.copyfile(os.path.join(dir_ori, file),
                    os.path.join(dir_tar, file))


def copy_and_rename_file(dir_ori, dir_tar, old_name, new_name):
    shutil.copyfile(os.path.join(dir_ori, old_name),
                    os.path.join(dir_tar, new_name))


def move_files(dir_ori, dir_tar, list_files=None):
    if list_files is None:
        list_files = os.listdir(dir_ori)
    num_files = len(list_files)
    with tqdm(total=num_files, desc='Move files') as bar:
        for file in list_files:
            shutil.move(os.path.join(dir_ori, file),
                        os.path.join(dir_tar, file))
            bar.update(1)


def check_list_len(l, max, min):
    if len(l) < min:
        raise ValueError('the list is too short! MIN: ' + min)
    if len(l) > max:
        raise ValueError('the list is too long! MAX: ' + max)


def filter_files_with_suffix(list_files, str_suffix):
    """
    suffix must be the last 4 latter
    """
    list_ret = []
    for file in list_files:
        if file[-4:] == str_suffix:
            list_ret.append(file)
    return list_ret


def get_padding_img(img, str_padding):
    """rasie ValueError"""
    padding_img = np.zeros((img.shape[0] * 3, img.shape[1] * 3, 3))
    padding_img[img.shape[0]:img.shape[0] * 2,
    img.shape[1]:img.shape[1] * 2, :] = img[:, :, :]
    if str_padding == 'zero':
        return padding_img
    if str_padding == 'circular':
        horizon_img = cv2.flip(img, 1)
        vertical_img = cv2.flip(img, 0)
        reflectall_img = cv2.flip(img, -1)
        padding_img[0:img.shape[0], 0:img.shape[1],
        :] = reflectall_img[:, :, :]
        padding_img[img.shape[0]:img.shape[0] * 2,
        0:img.shape[1], :] = horizon_img[:, :, :]
        padding_img[img.shape[0] * 2:img.shape[0] * 3,
        0:img.shape[1], :] = reflectall_img[:, :, :]

        padding_img[0:img.shape[0], img.shape[1]:img.shape[1] * 2, :] = vertical_img[:, :, :]
        padding_img[img.shape[0] * 2:img.shape[0] * 3, img.shape[1]:img.shape[1] * 2, :] = vertical_img[:, :, :]

        padding_img[0:img.shape[0], img.shape[1] *
                                    2:img.shape[1] * 3, :] = reflectall_img[:, :, :]
        padding_img[img.shape[0]:img.shape[0] * 2, img.shape[1]
                                                   * 2:img.shape[1] * 3, :] = horizon_img[:, :, :]
        padding_img[img.shape[0] * 2:img.shape[0] * 3, img.shape[1]
                                                       * 2:img.shape[1] * 3, :] = reflectall_img[:, :, :]
        return padding_img
    raise ValueError('No such padding mode: ' + str_padding)


def write_csv(path_csv, list_results):
    with open(path_csv, "a+", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list_results)


def read_csv(path_csv):
    list_lines = []
    with open(path_csv, "r") as csv_f:
        reader = csv.reader(csv_f)
        for line in reader:
            list_lines.append(line)
        return list_lines


def list_str_to_list_int(list_str):
    list_int = []
    for i in list_str:
        list_int.append(int(i))
    return list_int


def img_resize(img, h, w, cv2_interpolation):
    img = cv2.resize(img, (w, h), interpolation=cv2_interpolation)
    return img
