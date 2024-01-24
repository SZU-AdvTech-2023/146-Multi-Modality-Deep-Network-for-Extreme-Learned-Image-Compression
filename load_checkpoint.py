import torch
import numpy as np
import os


class Fusion_load(object):
    def __init__(self, model, checkpoint, optimizer=None, scheduler=None, dis=None, d_optim=None):
        """
        加载checkpoint文件的数据，包括model、optimizer、scheduler的数据，以及读取已经训练的epoch、训练消耗的总时间
        Args:
            model: 模型
            checkpoint: checkpoint文件路径
            optimizer: 指定的optimizer
            scheduler: 指定的scheduler
        """
        self.model = model
        self.checkpoint = checkpoint
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dis = dis
        self.d_optim = d_optim

        self.model_state_dict = None
        self.optimizer_state_dict = None
        self.scheduler_state_dict = None
        self.dis_state_dict = None
        self.d_optim_state_dict = None

        self.epoch = 0
        self.max_avg_psnr = 0.0
        self.total_cost_time = 0.0

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint):
            print("loading checkpoint...", self.checkpoint)
            # 主体模型
            model_dict = self.model.state_dict()
            # 判别器模型
            dis_dict = None
            if self.dis is not None:
                dis_dict = self.dis.state_dict()
            # 载入
            checkpoint_data = torch.load(self.checkpoint)
            self.epoch = checkpoint_data['epoch'] if 'epoch' in checkpoint_data else 0
            self.total_cost_time = checkpoint_data['total_cost_time'] if 'total_cost_time' in checkpoint_data else 0.0
            self.max_avg_psnr = checkpoint_data['max_avg_psnr'] if 'max_avg_psnr' in checkpoint_data else 0.0
            if 'model_state_dict' in checkpoint_data:
                self.model_state_dict = checkpoint_data['model_state_dict']
            elif 'model' in checkpoint_data:
                self.model_state_dict = checkpoint_data['model']

            self.optimizer_state_dict = checkpoint_data[
                'optimizer_state_dict'] if 'optimizer_state_dict' in checkpoint_data else None
            self.scheduler_state_dict = checkpoint_data[
                'scheduler_state_dict'] if 'scheduler_state_dict' in checkpoint_data else None

            self.dis_state_dict = checkpoint_data['dis_state_dict'] if 'dis_state_dict' in checkpoint_data else None
            # print('dis_state_dict' in checkpoint_data)
            # print(checkpoint_data['dis_state_dict'])
            # print("self.dis_state_dict =", self.dis_state_dict)
            self.d_optim_state_dict = checkpoint_data[
                'd_optim_state_dict'] if 'd_optim_state_dict' in checkpoint_data else None

            # ========================= 主体模型
            # if self.checkpoint.endswith(".npz"):
            #     modelCheckpoint = np.load(self.checkpoint)
            #     pretrained_dict = modelCheckpoint['state_dict']
            # if self.checkpoint.endswith(".pth"):
            # pretrained_dict = torch.load(self.checkpoint)
            # 过滤操作 这里存在一个大坑 不知什么原因有时候model的keys是带有module.的，有时候是不带的 有时候其中一个带一个不带 gpu上自带‘module’
            # print("mode_dict.keys() =", model_dict.keys())
            # print()
            # print("checkpoint_data.keys() =", self.model_state_dict.keys())
            # if list(model_dict.keys())[0][0:7] == list(checkpoint_data.keys())[0][0:7]:
            #     new_dict = {k: v for k, v in checkpoint_data.items() if k in model_dict.keys()}
            # else:
            #     new_dict = {k[7:]: v for k, v in checkpoint_data.items() if k[7:] in model_dict.keys()}

            if self.model_state_dict is not None:
                # if list(dis_dict.keys())[0][0:7] == list(checkpoint_data.keys())[0][0:7]:
                #     new_dict = {k: v for k, v in checkpoint_data.items() if k in dis_dict.keys()}
                # else:
                #     new_dict = {k[7:]: v for k, v in checkpoint_data.items() if k[7:] in dis_dict.keys()}

                # new_dict = {k: self.model_state_dict[k] for k in model_dict.keys() if k in self.model_state_dict}
                new_dict = {k: v for k, v in self.model_state_dict.items() if k in model_dict.keys()}
                model_dict.update(new_dict)
                # 打印出来，更新了多少的参数
                print('Model Total : {}, update: {}'.format(len(model_dict), len(new_dict)))
                self.model.load_state_dict(model_dict, True)
                print("model loaded!", self.checkpoint, "...")

            # ========================== 判别器模型
            if dis_dict is not None:
                # if list(dis_dict.keys())[0][0:7] == list(checkpoint_data.keys())[0][0:7]:
                #     new_dict = {k: v for k, v in checkpoint_data.items() if k in dis_dict.keys()}
                # else:
                #     new_dict = {k[7:]: v for k, v in checkpoint_data.items() if k[7:] in dis_dict.keys()}

                # new_dict = {k: self.dis_state_dict[k] for k in dis_dict.keys() if k in self.dis_state_dict}
                new_dict = {k: v for k, v in self.dis_state_dict.items() if k in dis_dict.keys()}
                dis_dict.update(new_dict)
                # 打印出来，更新了多少的参数
                print('Discriminator Total : {}, update: {}'.format(len(dis_dict), len(new_dict)))
                self.dis.load_state_dict(dis_dict, True)
                print("Discriminator loaded!", self.checkpoint, "...")

            # 如果不需要更新优化器那么设置为false
            if self.optimizer is not None:
                self.optimizer.load_state_dict(self.optimizer_state_dict)
                print('optimizer loaded!')
            if self.scheduler is not None:
                self.scheduler.load_state_dict(self.scheduler_state_dict)
                print('scheduler loaded!')

            if self.d_optim is not None:
                self.d_optim.load_state_dict(self.d_optim_state_dict)
                print('Discriminator optimizer loaded!')

        else:
            print('No checkpoint is included!')

        if self.dis is not None:
            return self.model, self.dis, os.path.exists(self.checkpoint)
        return self.model, os.path.exists(self.checkpoint)
