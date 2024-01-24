import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from tqdm import tqdm
import numpy as np
# if (__package__ == '') or (__package__ is None):
#     import sys
#     sys.path.append(os.path.abspath(
#         os.path.dirname(os.path.dirname(__file__))))
#     from utils import files
#     from utils.statics import Box
# else:
import files
from statics import Box
import argparse

def _analyze_line(line):
    x_ori = line[1]
    y_ori = line[2]
    box_real_full = Box()
    box_real_full.load_from_list(line[3:7])
    box_real_core = Box()
    box_real_core.load_from_list(line[7:11])
    return x_ori, y_ori, box_real_full, box_real_core


def _get_img_patch(img, box):
    return img[box.x_min:box.x_max, box.y_min:box.y_max, :]


def _combine_img(img_patch, img_combined, times_map, line):
    _, _, box_real_full, box_real_core = _analyze_line(line)
    img_combined[box_real_full.x_min:box_real_full.x_max,
                 box_real_full.y_min:box_real_full.y_max, :] += _get_img_patch(img_patch, box_real_core)
    times_map[box_real_full.x_min:box_real_full.x_max,
              box_real_full.y_min:box_real_full.y_max] += 1
    return img_combined, times_map


def combine_patch(dir_ori, list_lines):
    img_combined = np.zeros((int(list_lines[0][1]), int(list_lines[0][2]), 3))
    times_map = np.zeros((int(list_lines[0][1]), int(list_lines[0][2]), 1))
    for line in list_lines:
        img_patch = files.load_img_float(os.path.join(dir_ori, line[0]))
        img_combined, times_map = _combine_img(
            img_patch, img_combined, times_map, line)

    img_combined = img_combined/times_map
    return img_combined


def combine_patches(dir_ori, dir_tar):
    list_files = os.listdir(dir_ori)
    list_files = files.filter_files_with_suffix(list_files, '.csv')
    num_files = len(list_files)
    with tqdm(total=num_files, desc='CPatches') as bar:
        for file in list_files:
            list_lines = files.read_csv(os.path.join(dir_ori, file))
            img_combined = combine_patch(dir_ori, list_lines)
            files.save_img_float(os.path.join(
                dir_tar, file[:-4]+'.png'), img_combined)
            bar.update(1)


def main(dir_ori, dir_tar):
    files.check_folder(dir_ori)
    if dir_tar == '.':
        dir_tar = dir_ori+'_combined'
    files.refresh_folder(dir_tar)
    combine_patches(dir_ori, dir_tar)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Combine patches to a whole image')
    parser.add_argument('--origin', dest='dir_ori', default='../data202/G1/Test_data/infer_patches_CmJND_mae_mse', type=str)
    parser.add_argument('--target', dest='dir_tar',
                        type=str, default='.', required=False)
    args = parser.parse_args()
    main(args.dir_ori, args.dir_tar)
