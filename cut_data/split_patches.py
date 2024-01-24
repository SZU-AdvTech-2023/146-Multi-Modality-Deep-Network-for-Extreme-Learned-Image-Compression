import argparse
import os

from numpy.lib.function_base import append
from tqdm import tqdm
# if (__package__ == '') or (__package__ is None):
#     import sys
#     sys.path.append(os.path.abspath(
#         os.path.dirname(os.path.dirname(__file__))))
#     from utils import files
#     from utils.statics import Box
# else:
import files
from statics import Box


def _check_in_box(box_obj, box_bound):
    # 两矩形中心距离小于半边长之和时必然相交, 可以分轴计算
    dis_x = abs((box_obj.x_max + box_obj.x_min) / 2 - (box_bound.x_max + box_bound.x_min) / 2)
    dis_y = abs((box_obj.y_max + box_obj.y_min) / 2 - (box_bound.y_max + box_bound.y_min) / 2)

    len_x = (abs(box_obj.x_max - box_obj.x_min) + abs(box_bound.x_max - box_bound.x_min)) / 2
    len_y = (abs(box_obj.y_max - box_obj.y_min) + abs(box_bound.y_max - box_bound.y_min)) / 2

    if (dis_x <= len_x and dis_y <= len_y):
        return True
    else:
        return False


def _get_box_full(ix, iy, num_size, num_stride):
    x_min = ix * num_stride
    x_max = ix * num_stride + num_size
    y_min = iy * num_stride
    y_max = iy * num_stride + num_size
    box_full = Box(x_min, x_max, y_min, y_max)
    return box_full


def _get_box_core(box_big, num_border):
    x_core_min = box_big.x_min + num_border
    x_core_max = box_big.x_max - num_border
    y_core_min = box_big.y_min + num_border
    y_core_max = box_big.y_max - num_border
    box_core = Box(x_core_min, x_core_max, y_core_min, y_core_max)
    return box_core


def _get_mapping_pos(box_core, box_ori, num_size, num_border):
    box_real_core = Box(num_border, num_size - num_border,
                        num_border, num_size - num_border)

    if box_core.x_min <= box_ori.x_min:
        box_real_core.x_min = box_ori.x_min - box_core.x_min + num_border
        box_core.x_min = box_ori.x_min
    if box_core.x_max >= box_ori.x_max:
        box_real_core.x_max = num_size - num_border - \
                              (box_core.x_max - box_ori.x_max)
        box_core.x_max = box_ori.x_max
    if box_core.y_min <= box_ori.y_min:
        box_real_core.y_min = box_ori.y_min - box_core.y_min + num_border
        box_core.y_min = box_ori.y_min
    if box_core.y_max >= box_ori.y_max:
        box_real_core.y_max = num_size - num_border - \
                              (box_core.y_max - box_ori.y_max)
        box_core.y_max = box_ori.y_max

    box_real_full = Box(box_core.x_min - box_ori.x_min, box_core.x_max - box_ori.x_min,
                        box_core.y_min - box_ori.y_min, box_core.y_max - box_ori.y_min)

    return box_real_full, box_real_core


def _get_stride(num_size, num_border, num_ratio):
    num_stride = (num_size - 2 * num_border) // num_ratio
    return int(num_stride)


def _get_box_ori(img_padding_shape):
    x_ori = img_padding_shape[0] // 3
    y_ori = img_padding_shape[1] // 3
    box_ori = Box(x_ori, x_ori * 2, y_ori, y_ori * 2)
    return box_ori


def _save_combine_info(dir_tar, file, img_patches_and_info):
    path_csv = os.path.join(dir_tar, file[:-4] + '.csv')
    path_csv = path_csv.replace('\\', '/')

    for idx, img_patch in enumerate(img_patches_and_info):
        files.save_img_float(os.path.join(
            dir_tar, file[:-4] + '_' + str(idx) + '.bmp'), img_patch[0])
        line_info = [os.path.join(
            file[:-4] + '_' + str(idx) + '.bmp'), img_patch[3], img_patch[4]]
        line_info.extend(img_patch[1].get_list())
        line_info.extend(img_patch[2].get_list())
        line_info.extend(img_patch[5].get_list())
        files.write_csv(path_csv, line_info)


def _get_img_patch(img, box):
    return img[box.x_min:box.x_max, box.y_min:box.y_max, :]


def _get_patch_info(img_patch, box_core, box_ori, num_size, num_border):
    patch_info = []
    patch_info.append(img_patch)
    box_real_full, box_real_core = _get_mapping_pos(
        box_core, box_ori, num_size, num_border)
    patch_info.append(box_real_full)
    patch_info.append(box_real_core)
    return patch_info


def _get_patches_and_info(img_padding, box_full, box_core, box_ori, num_size, num_border):
    x_ori = box_ori.x_max - box_ori.x_min
    y_ori = box_ori.y_max - box_ori.y_min
    img_patch = _get_img_patch(img_padding, box_full)
    img_patch_and_info = _get_patch_info(
        img_patch, box_core, box_ori, num_size, num_border)
    img_patch_and_info.extend([x_ori, y_ori])
    img_patch_and_info.append(box_full)
    return img_patch_and_info


def _get_step_num(img_padding_shape, num_size, num_stride):
    x_num = (img_padding_shape[0] - num_size) // num_stride
    y_num = (img_padding_shape[1] - num_size) // num_stride
    return x_num, y_num


def split_patch(img_padding, num_size, num_border, num_stride):
    img_padding_shape = img_padding.shape
    # 中间的原图的在padding上的坐标
    box_ori = _get_box_ori(img_padding_shape)
    x_num, y_num = _get_step_num(img_padding_shape, num_size, num_stride)
    img_patches_and_info = []
    for ix in range(x_num):
        for iy in range(y_num):
            box_full = _get_box_full(ix, iy, num_size, num_stride)
            # core块在padding上的坐标
            box_core = _get_box_core(box_full, num_border)
            # 如果这个核心和原图坐标存在相交
            if _check_in_box(box_core, box_ori):
                img_patch_and_info = _get_patches_and_info(
                    img_padding, box_full, box_core, box_ori, num_size, num_border)
                img_patches_and_info.append(img_patch_and_info)
    return img_patches_and_info


def split_patches(dir_ori, dir_tar, num_size, num_border, num_ratio, str_padding):
    list_files = os.listdir(dir_ori)
    # 这个操作应该可以用endwith代替
    list_bmp_files = files.filter_files_with_suffix(list_files, '.bmp')
    list_png_files = files.filter_files_with_suffix(list_files, '.png')
    list_jpg_files = files.filter_files_with_suffix(list_files, '.jpg')
    list_files = list_bmp_files + list_png_files + list_jpg_files
    num_files = len(list_files)
    # 计算stride
    num_stride = _get_stride(num_size, num_border, num_ratio)
    with tqdm(total=num_files, desc='SPatches') as bar:
        for file in list_files:
            img = files.load_img_float(os.path.join(dir_ori, file))

            img_padding = files.get_padding_img(img, str_padding)  # 对图像进行填充padding
            img_patches_and_info = split_patch(
                img_padding, num_size, num_border, num_stride)

            _save_combine_info(dir_tar, file, img_patches_and_info)
            bar.update(1)


def main(dir_ori, dir_tar, num_size, num_border, num_ratio, str_padding):
    files.check_folder(dir_ori)
    if dir_tar == '.':
        dir_tar = dir_ori + '_split'
    files.refresh_folder(dir_tar)
    split_patches(dir_ori, dir_tar, num_size,
                  num_border, num_ratio, str_padding)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Split imgs into patches, saved with position file \'FILE_NAME.csv\'')
    parser.add_argument('--origin', dest='dir_ori', type=str,
                        help='folder of origin images', default='../data202/G1/Test_data/Ori', required=False)
    parser.add_argument('--target', dest='dir_tar',
                        type=str, default='../data202/G1/Test_data/patches_image', help='', required=False)
    parser.add_argument('--size', dest='num_size',
                        type=int, help='the "patch_size"', default=224, required=False)
    parser.add_argument('--border', dest='num_border',
                        type=int, default=12, required=False)
    parser.add_argument('--overlap', dest='num_ratio', type=float,
                        help='the "patch_size"', default=1.1, required=False)
    parser.add_argument('--mode', dest='str_padding', type=str,
                        default='circular', choices=('circular', 'zero'),
                        help='"circular" padding applies the reverse reflection strategy, "zero" just padding black',
                        required=False)
    args = parser.parse_args()
    main(args.dir_ori, args.dir_tar, args.num_size,
         args.num_border, args.num_ratio, args.str_padding)
