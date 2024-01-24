import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from torchtext.vocab import GloVe
from torchvision.models import ResNet50_Weights

from preprocess.sentence_process.sentence_process import sentence_to_word_vec
import zlib
import numpy as np
import difflib
from compressAI.compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressAI.compressai.models import CompressionModel
from torchsummary import summary
from networks.Entropy import TestModel
from torchvision import models


class PreprocessText(nn.Module):
    def __init__(self, device, glove):
        super().__init__()
        self.device = device
        self.glove = glove

    def forward(self, text):
        llm_word_vec_list = []
        max_word_vec_len = 0

        for t in text:
            llm_word_vec = sentence_to_word_vec(self.glove, t)

            # tokenized_text = tokenizer(t, padding='max_length', max_length=self.word_vec_dim, truncation=True,
            #                            return_tensors="pt")
            # llm_word_vec = tokenized_text['input_ids'].squeeze()

            max_word_vec_len = max(max_word_vec_len, llm_word_vec.shape[0])
            llm_word_vec_list.append(llm_word_vec)

        # print("llm_word_vec_list.shape:", len(llm_word_vec_list), len(llm_word_vec_list[0]),
        #       len(llm_word_vec_list[0][0]))

        # 扩展成同一个维度
        new_llm_word_vec_list = []
        for llm_word_vec in llm_word_vec_list:
            new_llm_word_vec = F.pad(llm_word_vec, (0, 0, 0, max_word_vec_len - llm_word_vec.shape[0]), mode='constant',
                                     value=0)
            new_llm_word_vec_list.append(new_llm_word_vec)

        f_tv = torch.stack(new_llm_word_vec_list).to(self.device)
        return f_tv


# class Analysis_net_17(nn.Module):
#     def __init__(self, input_channel, out_channel_N=192):
#         super(Analysis_net_17, self).__init__()
#         self.conv1 = nn.Conv2d(input_channel, out_channel_N, 9, stride=4, padding=4)
#         torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
#         torch.nn.init.constant_(self.conv1.bias.data, 0.01)
#         self.gdn1 = GDN(out_channel_N)
#         self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
#         torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
#         torch.nn.init.constant_(self.conv2.bias.data, 0.01)
#         self.gdn2 = GDN(out_channel_N)
#         self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, bias=False)
#         torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
#         # torch.nn.init.constant_(self.conv3.bias.data, 0.01)
#         # self.gdn3 = GDN(out_channel_N)
#         # self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
#         # torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
#         # torch.nn.init.constant_(self.conv4.bias.data, 0.01)
#
#     def forward(self, x):
#         x = self.gdn1(self.conv1(x))
#         x = self.gdn2(self.conv2(x))
#         x = self.conv3(x)
#         return x


# class Synthesis_net_17(nn.Module):
#     '''
#     Decode synthesis
#     '''
#
#     def __init__(self, out_channel, out_channel_N=192):
#         super(Synthesis_net_17, self).__init__()
#         self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
#         torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1)))
#         torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
#         self.igdn1 = GDN(out_channel_N, inverse=True)
#         self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
#         torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
#         torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
#         self.igdn2 = GDN(out_channel_N, inverse=True)
#         self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel, 9, stride=4, padding=4, output_padding=3)
#         torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
#         torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
#
#     def forward(self, x):
#         x = self.igdn1(self.deconv1(x))
#         x = self.igdn2(self.deconv2(x))
#         x = self.deconv3(x)
#         return x


class ITA(nn.Module):
    def __init__(self, input_channel, linear_input_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, kernel_size=1, stride=1, padding=0),
            nn.Dropout(),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout(),
            nn.LeakyReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, kernel_size=5, stride=1, padding=2),
            nn.Dropout(),
            nn.LeakyReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(linear_input_channel, input_channel * 3),
            nn.Dropout(),
            nn.LeakyReLU(),
        )

    def forward(self, v1, t):
        # print("t.shape =", t.shape)
        r1 = self.conv1(v1)
        r2 = self.conv2(v1)
        r3 = self.conv3(v1)
        r4 = torch.cat((r1, r2, r3), dim=1)
        r5 = r4 + torch.cat([v1.repeat(1, 3, 1, 1)], dim=1)  # [b, 3c, h, w] [5, 384, 64, 64]
        b, c, h, w = r5.shape
        r6 = r5.reshape(b, c, h * w).transpose(-1, -2)  # [b, h * w / 4, 12c] [5, 32 * 32, 12c]
        # print("r6.shape =", r6.shape)

        t1 = self.linear(t)  # [b, seq_len, 12c] [5, 6, 12c]
        t2 = t1.transpose(-1, -2)  # [b, 12c, seq_len] [5, 12c, 6]
        # print("t2.shape =", t2.shape)

        out1 = torch.bmm(r6, t2)  # [b, h * w, seq_len] [5, 32 * 32, 6]
        out2 = torch.bmm(out1, t1)  # [b, h * w, 3c] [5, 32 * 32, 12c]
        out2 = out2.transpose(-1, -2).reshape(b, c, h, w)  # [5, 12c, 32, 32]
        return out2


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out


class ResModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride=1):
        super(ResModule, self).__init__()
        self.blocks = nn.Sequential(
            ResBlock(in_channels, out_channels, stride),
            *[ResBlock(out_channels, out_channels) for _ in range(num_blocks - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class FE(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.c_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.Dropout(),
            nn.LeakyReLU(),
        )

        self.ita1 = ITA(input_channel=128, linear_input_channel=128)
        self.conv1 = nn.Sequential(
            nn.Conv2d(128 * 3, 128, kernel_size=4, stride=2, padding=1),
            nn.Dropout(),
            nn.LeakyReLU(),
        )

        self.ita2 = ITA(input_channel=128, linear_input_channel=128)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128 * 3, 256, kernel_size=4, stride=2, padding=1),
            nn.Dropout(),
            nn.LeakyReLU(),
        )

        self.res_module = ResModule(256, 256, 5)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 220, kernel_size=3, stride=1, padding=1),
            nn.Dropout(),
            nn.LeakyReLU(),
        )

    def forward(self, img, text):
        r1 = self.c_block1(img)
        r2 = self.conv1(self.ita1(r1, text))
        r3 = self.conv2(self.ita2(r2, text))
        r4 = self.conv3(self.res_module(r3))
        return r4


class EntropyModel(nn.Module):
    """
    一个输入, 输出压缩特征和模型结果
    """

    def __init__(self, hidden_dim, mode='train'):
        super().__init__()
        self.mode = mode

        # F_he Encoder
        self.c_block1 = nn.Sequential(
            nn.Conv2d(220, 320, kernel_size=3, stride=1, padding=1),
            nn.Dropout(),
            nn.LeakyReLU(),

            nn.Conv2d(320, 320, kernel_size=5, stride=2, padding=2),
            nn.Dropout(),
            nn.LeakyReLU(),

            nn.Conv2d(320, 320, kernel_size=5, stride=2, padding=2),
            nn.Dropout(),
            nn.LeakyReLU(),
        )
        # self.ae2 = Analysis_net_17(input_channel=320)

        # F_hg Decoder
        # self.ad2 = Synthesis_net_17(out_channel=320)
        self.c_block2_1 = nn.Sequential(
            nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1),
            nn.Dropout(),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1),
            nn.Dropout(),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.ita = ITA(input_channel=320, linear_input_channel=128)
        self.c_block2_2 = nn.Sequential(
            nn.Conv2d(320 * 3, 220, kernel_size=3, stride=1, padding=1),
            nn.Dropout(),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
        )

        # new code
        # self.num_slices = 2
        # self.max_support_slices = 2

        self.entropy = None

    def forward(self, x, text):
        # r = self.c_block1(x)
        # print("r =", r.shape)
        # t = torch.randn(5, 320, 2, 2).cuda()
        # print("t =", t.shape)
        # compressed_feature = self.ae2(t)
        # print("compressed_feature =", compressed_feature.shape)
        # recon_features = self.ad2(compressed_feature)
        # print("recon_features =", recon_features.shape)
        # recon_features = self.c_block2(recon_features)

        r = self.c_block1(x)
        # print("r.shape =", r.shape)
        # q = quantization_function(r, 8)
        # print(torch.max(r))
        # print(torch.min(r))

        # 传入一个tensor, 将其转成字节流, 无损压缩, 然后再恢复成tensor
        # compressed_features, recon_features = compress_and_recover(r)
        # compressed_features, compressed_bytes = compress(r)
        # recon_features = decompress(r, compressed_bytes)

        # new code
        # y = r
        # y_shape = y.shape[2:]
        # latent_scales = None
        # latent_means = None
        #
        # y_slices = y.chunk(self.num_slices, 1)
        # y_hat_slices = []
        # y_likelihood = []
        # mu_list = []
        # scale_list = []
        # for slice_index, y_slice in enumerate(y_slices):
        #     support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
        #     mean_support = torch.cat([latent_means] + support_slices, dim=1)
        #     mean_support = self.atten_mean[slice_index](mean_support)
        #     mu = self.cc_mean_transforms[slice_index](mean_support)
        #     mu = mu[:, :, :y_shape[0], :y_shape[1]]
        #     mu_list.append(mu)
        #     scale_support = torch.cat([latent_scales] + support_slices, dim=1)
        #     scale_support = self.atten_scale[slice_index](scale_support)
        #     scale = self.cc_scale_transforms[slice_index](scale_support)
        #     scale = scale[:, :, :y_shape[0], :y_shape[1]]
        #     scale_list.append(scale)
        #     _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
        #     y_likelihood.append(y_slice_likelihood)
        #     y_hat_slice = ste_round(y_slice - mu) + mu
        #     # if self.training:
        #     #     lrp_support = torch.cat([mean_support + torch.randn(mean_support.size()).cuda().mul(scale_support), y_hat_slice], dim=1)
        #     # else:
        #     lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
        #     lrp = self.lrp_transforms[slice_index](lrp_support)
        #     lrp = 0.5 * torch.tanh(lrp)
        #     y_hat_slice += lrp
        #
        #     y_hat_slices.append(y_hat_slice)
        #
        # y_hat = torch.cat(y_hat_slices, dim=1)
        # print("f_hg_ad_out =", y_hat.shape)
        # new code

        # new code2
        r = F.interpolate(r, scale_factor=16, mode='bilinear', align_corners=False)
        b, c, h, w = r.shape
        # print("b, c, h, w =", b, c, h, w)
        self.entropy = TestModel((b, c, h, w), (1, c, h, w), is_high=True).to(r.device)
        if self.mode == 'train':
            train_bpp, train_mse, recon_features, compressed_features = self.entropy(r, self.mode)
        else:
            val_bpp, val_mse, val_psnr, recon_features, compressed_features = self.entropy(r, self.mode)
        recon_features = F.interpolate(recon_features, scale_factor=1 / 16, mode='bilinear', align_corners=False)
        # new code2

        # print("compressed_features =", compressed_features.shape)  # [5980]
        recon_features = self.c_block2_1(recon_features)
        # print("c_block2_1 recon_features =", recon_features.shape)  # [5, 320, 4, 4]
        recon_features = self.ita(recon_features, text)
        recon_features = self.c_block2_2(recon_features)
        # print("recon_features =", recon_features.shape)  # [5, 220, 4, 4]
        return compressed_features, recon_features


def ste_round(x):
    return torch.round(x) - x.detach() + x


class IRC(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.linear = None
        self.softmax = nn.Softmax(dim=-1)
        self.ita = ITA(input_channel=input_channel, linear_input_channel=input_channel)
        self.adjust_linear = None

    def forward(self, v2, text):
        self.linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(text.shape[1], text.shape[1]),
            nn.Dropout(),
            nn.LeakyReLU(),
        ).to(v2.device)
        b, c, h, w = v2.shape
        r1 = v2.reshape(b, c, h * w)
        self.adjust_linear = nn.Linear(text.shape[-1], r1.shape[1]).to(v2.device)
        text = self.adjust_linear(text)
        # print("text.shape=", text.shape, ",r1.shape=", r1.shape)
        r2 = torch.bmm(text, r1)
        # print("r2.shape =", r2.shape)
        r3 = r2.transpose(-1, -2)
        r4 = torch.bmm(r2, r3)
        # print("r4.shape =", r4.shape)
        a = self.softmax(self.linear(r4))
        # print("a.shape =", a.shape)
        r5 = torch.bmm(a, text)
        r5 += text

        # print("v2.shape=", v2.shape, ",r5.shape=", r5.shape)
        v2_hat = self.ita(v2, r5)  # [5, 384, 64, 64]
        # print("v2_hat.shape", v2_hat.shape)
        return v2_hat


class FG(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(220, 256, kernel_size=3, stride=1, padding=1),
            nn.Dropout(),
            nn.LeakyReLU(),

            ResModule(256, 256, 5),
            nn.Dropout(),
            nn.LeakyReLU(),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.Dropout(),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
        )

        self.irc1 = IRC(input_channel=128)

        self.block2 = nn.Sequential(
            nn.Conv2d(128 * 3, 128, kernel_size=3, stride=1, padding=1),
            nn.Dropout(),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
        )

        self.irc2 = IRC(input_channel=128)

        self.block3 = nn.Sequential(
            nn.Conv2d(128 * 3, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
        )

        self.irc3 = IRC(input_channel=64)

        self.block4 = nn.Sequential(
            nn.Conv2d(64 * 3, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(),
            nn.LeakyReLU(),

            ResBlock(64, 64, 1),
            nn.Dropout(),
            nn.LeakyReLU(),

            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Dropout(),
            nn.LeakyReLU(),
        )

    def forward(self, v, t):
        # print("t === ", t.shape)
        r1 = self.block1(v)
        # print("r1 =", r1.shape)
        r2 = self.irc1(r1, t)
        # print("r2 =", r2.shape)
        r3 = self.block2(r2)
        # print("r3 =", r3.shape)
        r4 = self.irc2(r3, t)
        # print("r4 =", r4.shape)
        r5 = self.block3(r4)
        # print("r5 =", r5.shape)
        r6 = self.irc3(r5, t)
        # print("r6 =", r6.shape)
        r7 = self.block4(r6)
        return r7


class MMDN(nn.Module):
    def __init__(self, device, glove, word_vec_dim, hidden_dim, model_dim=192, patch_size=8, num_heads=8, num_layers=6,
                 mode='train'):
        super().__init__()
        self.device = device
        self.glove = glove
        self.hidden_dim = hidden_dim
        self.preprocess = PreprocessText(device=device, glove=glove)
        self.text_encoder = nn.LSTM(input_size=word_vec_dim, hidden_size=hidden_dim, batch_first=True,
                                    bidirectional=True)

        self.fe = FE(hidden_dim=hidden_dim)
        # self.ae = Analysis_net_17(input_channel=220)
        # self.ad = Synthesis_net_17(out_channel=220)

        self.entropy_model = EntropyModel(hidden_dim=hidden_dim, mode=mode)

        self.fg = FG(hidden_dim=hidden_dim)

        self.image_encoder = TransformerEncoder(patch_size=patch_size, hidden_dim=model_dim, num_heads=num_heads,
                                                num_layers=num_layers)
        self.it = IT()

        self.entropy = None

        initial_weights = torch.rand(5, requires_grad=True)
        self.weights = nn.Parameter(initial_weights)

    def compute_weighted_loss(self, losses):
        # 对权重使用softmax来保证它们的和为1
        weights = torch.nn.functional.softmax(self.weights, dim=0)
        weighted_losses = weights * losses
        return weighted_losses.sum()

    def forward(self, img, text):
        preprocessed_text = self.preprocess(text)  # [bs, seq_len, word_vec_dim]
        # print(preprocessed_text.shape) # [bs, seq_len, word_vec_dim]
        t, (h, c) = self.text_encoder(preprocessed_text)  # [bs, seq_len, hidden_dim * 2]
        # print("text =", t.shape)

        fe_out = self.fe(img, t)
        # print("fe =", fe_out.shape)  [4, 220, 16, 16]

        cf_2, rf_2 = self.entropy_model(fe_out, t)
        # print("cf_2 =", cf_2.shape)  # [4, 320, 1, 1]
        # print("rf_2 =", rf_2.shape)  # [4, 220, 32, 32]

        # new code
        # y = fe_out
        # y_shape = y.shape[2:]
        # latent_scales = rf_2
        # latent_means = rf_2
        #
        # y_slices = y.chunk(self.num_slices, 1)
        # y_hat_slices = []
        # y_likelihood = []
        # mu_list = []
        # scale_list = []
        # for slice_index, y_slice in enumerate(y_slices):
        #     support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
        #     mean_support = torch.cat([latent_means] + support_slices, dim=1)
        #     mean_support = self.atten_mean[slice_index](mean_support)
        #     mu = self.cc_mean_transforms[slice_index](mean_support)
        #     mu = mu[:, :, :y_shape[0], :y_shape[1]]
        #     mu_list.append(mu)
        #     scale_support = torch.cat([latent_scales] + support_slices, dim=1)
        #     scale_support = self.atten_scale[slice_index](scale_support)
        #     scale = self.cc_scale_transforms[slice_index](scale_support)
        #     scale = scale[:, :, :y_shape[0], :y_shape[1]]
        #     scale_list.append(scale)
        #     _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
        #     y_likelihood.append(y_slice_likelihood)
        #     y_hat_slice = ste_round(y_slice - mu) + mu
        #     # if self.training:
        #     #     lrp_support = torch.cat([mean_support + torch.randn(mean_support.size()).cuda().mul(scale_support), y_hat_slice], dim=1)
        #     # else:
        #     lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
        #     lrp = self.lrp_transforms[slice_index](lrp_support)
        #     lrp = 0.5 * torch.tanh(lrp)
        #     y_hat_slice += lrp
        #
        #     y_hat_slices.append(y_hat_slice)
        #
        # y_hat = torch.cat(y_hat_slices, dim=1)
        # print("f_hg_ad_out =", y_hat.shape)
        # new code

        fe_out = F.interpolate(fe_out, scale_factor=2, mode='bilinear', align_corners=False)
        # fe_out = F.interpolate(fe_out, scale_factor=2, mode='nearest')
        # rf_1 = torch.clamp(rf_2 + fe_out, -1.0, 1.0)
        fe_out_out = rf_2 + fe_out

        # cf_1, rf_1 = compress_and_recover(rf_1)

        # new code2
        b, c, h, w = fe_out_out.shape
        fe_out_out = F.interpolate(fe_out_out, scale_factor=2, mode='bilinear', align_corners=False)
        self.entropy = TestModel((b, c, h, w), (1, c, h, w), is_high=True).to(fe_out_out.device)
        train_bpp, train_mse, rf_1, cf_1 = self.entropy(fe_out_out, mode='train')
        rf_1 = F.interpolate(rf_1, scale_factor=0.5, mode='bilinear', align_corners=False)
        # new code2

        # print(cf_1.shape, cf_2.shape)  # [4, 220, 1, 1] [4, 320, 1, 1]
        # print(rf_1.shape, rf_2.shape)  # [4, 220, 32, 32] [4, 220, 32, 32]
        # print(torch.max(cf_1))
        # print(torch.min(cf_1))
        # print(torch.mean(cf_1))
        # print(torch.max(cf_2))
        # print(torch.min(cf_2))
        # print(torch.mean(cf_2))

        # Loss 1: cf_1 and cf_2
        # 均值填充
        # print(len(cf_1), len(cf_2))
        b, c, h, w = cf_1.shape
        cf_1 = cf_1.reshape(b * c * h * w)
        b, c, h, w = cf_2.shape
        cf_2 = cf_2.reshape(b * c * h * w)
        cf_1, cf_2 = mean_padding(cf_1, cf_2)
        # cf_1 = torch.tensor(cf_1, requires_grad=True)
        # cf_2 = torch.tensor(cf_2, requires_grad=True)
        cf_1 = cf_1.clone().detach().float().requires_grad_(True)
        cf_2 = cf_2.clone().detach().float().requires_grad_(True)
        # print(len(cf_1), len(cf_2))
        # mse_loss = torch.nn.MSELoss()
        # loss = 0.00001 * mse_loss(cf_1, cf_2)
        # print(loss)

        # 截断
        # print(len(cf_1), len(cf_2))
        # cf_1, cf_2 = mean_crop(cf_1, cf_2)
        # print(len(cf_1), len(cf_2))
        # mse_loss = torch.nn.MSELoss()
        # loss = 0.00001 * mse_loss(cf_1, cf_2)
        # print(loss)

        out1 = rf_1 + rf_2
        # print(out1.shape)  # [5, 220, 32, 32]

        out = self.fg(out1, t)
        out = torch.clamp(out, -0.05, 0.05)

        encoded_out_img = self.image_encoder(out)
        encoded_ori_img = self.image_encoder(img)

        # out = torch.clamp(out, -1.0, 1.0)

        l_it = self.it(encoded_out_img, t)
        # for test
        # dis = Discriminator(hidden_dim=self.hidden_dim).cuda()
        # real_output = dis(out, t)
        # print("real_output", real_output.shape)
        return cf_1, cf_2, out, l_it, encoded_out_img, encoded_ori_img


class ImageEmbedding(nn.Module):
    def __init__(self, hidden_dim, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        conv_output = self.conv(x)
        bs, oc, oh, ow = conv_output.shape
        patch_embedding = conv_output.reshape((bs, oc, oh * ow)).transpose(-1, -2)
        return patch_embedding


class TransformerEncoder(nn.Module):
    def __init__(self, patch_size, hidden_dim, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()

        self.patch_size = patch_size
        self.embedding = ImageEmbedding(hidden_dim=hidden_dim, patch_size=patch_size)
        # self.position_encoding = PositionalEncoding(hidden_dim)
        self.layers = nn.ModuleList([TransformerEncoderLayer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x = self.embedding(x)
        # x = self.position_encoding(x)

        b, c, h, w = x.shape
        nh, nw = h // self.patch_size, w // self.patch_size
        nc = c * self.patch_size * self.patch_size
        # reshaped_tensor = x.reshape(b, nc, nh, nw).reshape(b, nc, nh * nw).transpose(-1, -2)
        # x = reshaped_tensor
        x = self.embedding(x)

        # print("进入Transformer之前 x =", x.shape)
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        # print("出来Transformer时 x =", x.shape)
        x = x.transpose(-1, -2).reshape(b, nc, nh, nw).reshape(b, c, h, w)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x

        x = self.norm1(x)
        x, _ = self.self_attention(x, x, x)
        x = x + residual

        residual = x

        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + residual

        return x


# 扩充
def mean_padding(t1, t2):
    device_used = t1.device
    if len(t1) < len(t2):
        mean_value = t1.float().mean()
        to_pad = len(t2) - len(t1)
        t1 = F.pad(t1, (0, to_pad), value=mean_value.item())
    elif len(t1) > len(t2):
        mean_value = t2.float().mean()
        to_pad = len(t1) - len(t2)
        t2 = F.pad(t2, (0, to_pad), value=mean_value.item())
    return t1.to(device_used), t2.to(device_used)


# 截断
def mean_crop(t1, t2):
    device_used = t1.device
    if len(t1) < len(t2):
        t2 = t2[:len(t1)]
    elif len(t1) > len(t2):
        t1 = t1[:len(t2)]
    return t1.to(device_used), t2.to(device_used)


def calculate_bytes_loss(data1, data2):
    # 假设有两个bytes类型的数据data1和data2

    # 计算两个数据的相似度比较
    # similarity = difflib.SequenceMatcher(None, data1, data2).ratio()

    # 或者计算两个数据的差异
    # differences = difflib.ndiff(data1, data2)
    # delta = ''.join(differences)

    # 也可以使用其他相似度度量方法，例如哈希值的差异
    hash_difference = hash(data1) - hash(data2)
    return hash_difference


def quantization_function(image_tensor, levels):
    device_used = image_tensor.device
    img_numpy = image_tensor.cpu().detach().numpy()
    # 将图像的像素值映射到指定的levels范围内
    quantized_image = np.digitize(img_numpy, np.arange(0, 256, 256 / levels)) - 1
    quantized_image = torch.from_numpy(quantized_image).to(device_used)
    return quantized_image


def compress(r):
    device_used = r.device
    # print("r =", r.shape)  # [5, 320, 3, 3]
    r_array = r.cpu().detach().numpy()
    # print("r_array =", r_array.shape)
    r_bytes = r_array.tobytes()
    # print("r_bytes =", len(r_bytes))  # 57600
    compressed_bytes = zlib.compress(r_bytes)
    # print("compressed_bytes =", len(compressed_bytes))  # 53489
    compressed_array = np.frombuffer(compressed_bytes, dtype=np.uint8)
    compressed_array = np.copy(compressed_array).astype(np.float32)
    compressed_features = torch.from_numpy(compressed_array).to(device_used)
    return compressed_features, compressed_bytes


def decompress(r, compressed_bytes):
    b, c, h, w = r.shape
    device_used = r.device
    compressed_array = np.frombuffer(compressed_bytes, dtype=np.uint8)
    compressed_array = np.copy(compressed_array).astype(np.float32)
    compressed_features = torch.from_numpy(compressed_array).to(device_used)
    decompressed_bytes = zlib.decompress(compressed_bytes)
    # print("decompressed_bytes =", len(decompressed_bytes))  # 57600
    decompressed_array = np.frombuffer(decompressed_bytes, dtype=np.float32)
    decompressed_array = np.copy(decompressed_array)
    recon_features = torch.from_numpy(decompressed_array).view(b, c, h, w).to(device_used)
    return recon_features


def compress_and_recover(r):
    b, c, h, w = r.shape
    device_used = r.device
    # print("r =", r.shape)  # [5, 320, 3, 3]
    r_array = r.cpu().detach().numpy()
    # print("r_array =", r_array.shape)
    r_bytes = r_array.tobytes()
    # print("r_bytes =", len(r_bytes))  # 57600
    compressed_bytes = zlib.compress(r_bytes)
    # print("compressed_bytes =", len(compressed_bytes))  # 53489
    compressed_array = np.frombuffer(compressed_bytes, dtype=np.uint8)
    compressed_array = np.copy(compressed_array).astype(np.uint8)
    compressed_features = torch.from_numpy(compressed_array).to(device_used)
    decompressed_bytes = zlib.decompress(compressed_bytes)
    # print("decompressed_bytes =", len(decompressed_bytes))  # 57600
    decompressed_array = np.frombuffer(decompressed_bytes, dtype=r_array.dtype)
    decompressed_array = np.copy(decompressed_array)
    recon_features = torch.from_numpy(decompressed_array).view(b, c, h, w).to(device_used)
    return compressed_features, recon_features


# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.adjust = None
#         self.main = None
#
#     def forward(self, x, t):
#         b, c, h, w = x.shape
#         b, seq_len, h2 = t.shape
#         self.adjust = nn.Linear(h * w, seq_len).to(x.device)
#         x = x.reshape(b, c, h * w)
#         r = self.adjust(x)
#         out = torch.bmm(r, t)  # [b, c, h2]
#         out = out.view(-1, c * h2)
#         self.main = nn.Sequential(
#             nn.Linear(c * h2, 256),
#             nn.Dropout(),
#             nn.LeakyReLU(),
#
#             nn.Linear(256, 128),
#             nn.Dropout(),
#             nn.LeakyReLU(),
#
#             nn.Linear(128, 64),
#             nn.Dropout(),
#             nn.LeakyReLU(),
#
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         ).to(out.device)
#         return self.main(out)


class Discriminator(nn.Module):
    def __init__(self, device, glove, hidden_dim, word_vec_dim):
        super().__init__()

        self.preprocess = PreprocessText(device=device, glove=glove)
        self.text_encoder = nn.LSTM(input_size=word_vec_dim, hidden_size=hidden_dim, batch_first=True,
                                    bidirectional=True)

        self.tensor_adjust = nn.Linear(3, hidden_dim * 2)

        self.conv_adjust = None
        self.conv_list = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(),
            nn.LeakyReLU(),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(),
            nn.LeakyReLU(),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(),
            nn.LeakyReLU(),

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(),
            nn.LeakyReLU(),

            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(),
            nn.LeakyReLU(),

            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(),
            nn.LeakyReLU(),
        )

        # self.main = nn.Sequential(
        #     nn.Linear(48, 16),
        #     # nn.Dropout(),
        #     nn.LeakyReLU(),
        #
        #     # nn.Linear(1024, 512),
        #     # nn.Dropout(),
        #     # nn.LeakyReLU(),
        #     #
        #     # nn.Linear(512, 256),
        #     # nn.Dropout(),
        #     # nn.LeakyReLU(),
        #
        #     nn.Linear(16, 2),
        #     nn.Sigmoid()
        # )

        self.main = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).cuda()
        num_features = self.main.fc.in_features
        self.main.fc = nn.Sequential(
            nn.Linear(num_features, 2),
            nn.Sigmoid()
        )

    def forward(self, x, t):
        t = self.preprocess(t)  # [bs, seq_len, word_vec_dim]
        # print(preprocessed_text.shape) # [bs, seq_len, word_vec_dim]
        t, (h, c) = self.text_encoder(t)  # [bs, seq_len, hidden_dim * 2]
        # print("dis t=", t.shape)
        b, c, h, w = x.shape
        b, seq_len, h2 = t.shape

        x = x.reshape(b, c, h * w).transpose(-1, -2)  # [b, h * w, c]
        x = self.tensor_adjust(x)  # [b, h * w, h2]
        t = t.reshape(b, h2, seq_len)  # [b, h2, seq_len]
        # print("dis x=", x.shape)  # [16, 65536, 128] [b, h * w, h2]
        # print("dis t=", t.shape)  # [16, 128, 143] [b, h2, seq_len]
        out = torch.bmm(x, t)  # [b, h * w, seq_len]
        out = out.transpose(-1, -2).reshape(b, seq_len, h, w)  # [b, seq_len, h, w]
        self.conv_adjust = nn.Conv2d(seq_len, 128, kernel_size=3, stride=1, padding=1).to(x.device)

        out = self.conv_adjust(out)
        # print("out ===", out.shape)
        out = self.conv_list(out)
        # print("out ===", out.shape)
        out = out.reshape(-1, 48)
        # print("out ===", out.shape)

        return self.main(out)


class IT(nn.Module):

    def __init__(self):
        super().__init__()
        self.adjust = None

    def forward(self, x, t):
        b, c, h, w = x.shape
        b, seq_len, h2 = t.shape
        num = math.gcd(c * h * w, seq_len * h2)

        x1 = x.reshape(b, num, c * h * w // num)
        t1 = t.reshape(b, num, seq_len * h2 // num)

        # x [b, num, c * h * w // num] => [b, num, seq_len * h2 // num]
        # t [b, num, seq_len * h2 // num] => [b, seq_len * h2 // num, num]
        self.adjust = nn.Linear(c * h * w // num, seq_len * h2 // num).to(x1.device)
        x2 = self.adjust(x1)
        t2 = t1.transpose(-1, -2)
        out1 = torch.bmm(x2, t2)
        out2 = torch.bmm(t2, x2)

        # 对两个张量分别进行对数计算
        log_out1 = torch.log(out1)
        log_out2 = torch.log(out2)

        # 对结果进行求和
        sum_log = torch.sum(log_out1) + torch.sum(log_out2)
        # sum_log = torch.sum(sum_log)
        # sum_log = torch.tensor(sum_log, requires_grad=True)

        # 最后取负号
        out = -sum_log
        return out


if __name__ == '__main__':
    # prompt = """ Please provide a description of this image, which requires: (1) The description should include
    # at least one brightness word to reflect the brightness adaptation effect of the human eye. (2) The
    # description should include descriptors with different levels of darkness. (3) The description should
    # include object and attribute descriptors. """
    # origin_path_tuple = ('path1', 'path2', 'path3', 'path4', 'path5')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    image = torch.zeros((5, 3, 256, 256)).to(device)
    # image = torch.zeros((5, 3, 128, 128)).to(device)
    # image = torch.zeros((5, 3, 64, 64)).to(device)
    text_list = ['this is an apple', 'a banana', 'may be an orange pipe', 'watermelon memory',
                 'it can be a strawberry juice']
    word_vec_dim = 300
    glove = GloVe(name='6B', dim=word_vec_dim,
                  cache=r'D:\Desktop\JND_WORKS\MLLMJND\MLLMJND_ALL_CODE\MLLMJND_CODE\PMA\sentence_process')
    mmdn = MMDN(device=device, glove=glove, word_vec_dim=word_vec_dim, hidden_dim=64, model_dim=48, patch_size=4,
                num_heads=4, num_layers=3, mode='train')
    mmdn = mmdn.to(device)
    cf_1, cf_2, output, l_it, encoded_out_img, encoded_ori_img = mmdn(image, text_list)

    print("output.shape =", output.shape)

    # 打印模型的参数量统计(文本输入大小那里有问题)
    # summary(mllmJND, input_size=[(3, 224, 224), (3, 224, 224), (3, 224, 224), (3, 224, 224), (1, 1, 300)])

    print("\n Gen Summary parameters:")
    total_params = sum(p.numel() for p in mmdn.parameters())
    total_params += sum(p.numel() for p in mmdn.buffers())
    print(f'{total_params:,} total parameters.')
    print(f'{total_params / (1024 * 1024):.2f}M total parameters.')
    total_trainable_params = sum(p.numel() for p in mmdn.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(f'{total_trainable_params / (1024 * 1024):.2f}M training parameters.')

    # 计算损失
    # Rate Loss
    # mse_loss = torch.nn.MSELoss()
    # rate_loss = 0.00001 * mse_loss(cf_1, cf_2)
    # print("rate_loss =", rate_loss)

    # GAN Loss
    # Discriminator
    dis = Discriminator(device=device, glove=glove, hidden_dim=64, word_vec_dim=word_vec_dim).cuda()
    real_output = dis(image, text_list)

    print("output.shape =", real_output.shape)

    print("\n Dis Summary parameters:")
    total_params = sum(p.numel() for p in dis.parameters())
    total_params += sum(p.numel() for p in dis.buffers())
    print(f'{total_params:,} total parameters.')
    print(f'{total_params / (1024 * 1024):.2f}M total parameters.')
    total_trainable_params = sum(p.numel() for p in dis.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(f'{total_trainable_params / (1024 * 1024):.2f}M training parameters.')
