import numpy as np

class Box:
    """
    box_ori: 中心原始图片在padding的坐标
    box_full: 正式裁取下来的分块在padding的坐标
    box_core: 会应用到结果中的核心部分在padding中的坐标
    box_real_core: 核心部分在分块内部的局部坐标
    box_real_full: 核心部分在重建后全图的坐标
    """
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0

    def __init__(self, x_min=0, x_max=0, y_min=0, y_max=0):
        self.x_min = int(x_min)
        self.x_max = int(x_max)
        self.y_min = int(y_min)
        self.y_max = int(y_max)

    def __str__(self):
        return str([self.x_min, self.x_max, self.y_min, self.y_max])

    def load_from_list(self, list_in):
        """load from [x_min, x_max, y_min, y_max]"""
        self.x_min = int(list_in[0])
        self.x_max = int(list_in[1])
        self.y_min = int(list_in[2])
        self.y_max = int(list_in[3])

    def get_list(self):
        """return [x_min, x_max, y_min, y_max]"""
        list_out = [self.x_min, self.x_max, self.y_min, self.y_max]
        return list_out
