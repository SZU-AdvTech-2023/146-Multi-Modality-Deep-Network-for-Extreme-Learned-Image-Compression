import torch
from torch import nn
import numpy as np
from torch.autograd import Function
from networks.gdn import GDN, IGDN
import torch.nn.functional as F
import cv2
from torchsummary import summary

class analysisTransformModel(nn.Module):
    def __init__(self, in_dim, num_filters, conv_trainable=True):
        super(analysisTransformModel, self).__init__()
        self.transform = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_dim, num_filters[0], 5, 2, 0),
            GDN(num_filters[0]),

            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(num_filters[0], num_filters[1], 5, 2, 0),
            GDN(num_filters[1]),

            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(num_filters[1], num_filters[2], 5, 2, 0),
            GDN(num_filters[2]),

            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(num_filters[2], num_filters[3], 5, 2, 0)
        )

    def forward(self, inputs):
        x = self.transform(inputs)
        return x


class synthesisTransformModel(nn.Module):
    def __init__(self, in_dim, num_filters, conv_trainable=True):
        super(synthesisTransformModel, self).__init__()
        self.transform = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(in_dim, num_filters[0], 5, 2, 3, output_padding=1),
            IGDN(num_filters[0], inverse=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(num_filters[0], num_filters[1], 5, 2, 3, output_padding=1),
            IGDN(num_filters[1], inverse=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(num_filters[1], num_filters[2], 5, 2, 3, output_padding=1),
            IGDN(num_filters[2], inverse=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(num_filters[2], num_filters[3], 5, 2, 3, output_padding=1),
            IGDN(num_filters[3], inverse=True)
        )

    def forward(self, inputs):
        x = self.transform(inputs)
        return x


class h_analysisTransformModel(nn.Module):
    def __init__(self, in_dim, num_filters, strides_list, conv_trainable=True):
        super(h_analysisTransformModel, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_dim, num_filters[0], 3, strides_list[0], 1),
            nn.ReLU(),
            nn.Conv2d(num_filters[0], num_filters[1], 5, strides_list[1], 2),
            nn.ReLU(),
            nn.Conv2d(num_filters[1], num_filters[2], 5, strides_list[2], 2)
        )

    def forward(self, inputs):
        x = torch.abs(inputs)
        x = self.transform(x)
        return x


class h_synthesisTransformModel(nn.Module):
    def __init__(self, in_dim, num_filters, strides_list, conv_trainable=True):
        super(h_synthesisTransformModel, self).__init__()
        self.transform = nn.Sequential(
            nn.ConvTranspose2d(in_dim, num_filters[0], 5, strides_list[0], 2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters[0], num_filters[1], 5, strides_list[1], 2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters[1], num_filters[2], 3, strides_list[2], 1),
        )

    def forward(self, inputs):
        x = self.transform(inputs)
        return x


class GaussianModel(nn.Module):
    def __init__(self):
        super(GaussianModel, self).__init__()

        self.m_normal_dist = torch.distributions.normal.Normal(0., 1.)

    def _cumulative(self, inputs, stds, mu):
        half = 0.5
        eps = 1e-6
        upper = (inputs - mu + half) / (stds)
        lower = (inputs - mu - half) / (stds)
        cdf_upper = self.m_normal_dist.cdf(upper)
        cdf_lower = self.m_normal_dist.cdf(lower)
        res = cdf_upper - cdf_lower
        return res

    def forward(self, inputs, hyper_sigma, hyper_mu):
        likelihood = self._cumulative(inputs, hyper_sigma, hyper_mu)
        likelihood_bound = 1e-8
        likelihood = torch.clamp(likelihood, min=likelihood_bound)
        return likelihood


class PredictionModel_Context(nn.Module):
    def __init__(self, in_dim, dim=192, trainable=True, outdim=None):
        super(PredictionModel_Context, self).__init__()
        if outdim is None:
            outdim = dim
        self.transform = nn.Sequential(
            nn.Conv2d(in_dim, dim, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, dim, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Linear(dim * 2 * 2, outdim)
        self.flatten = nn.Flatten()

    def forward(self, y_rounded, h_tilde, y_sampler, h_sampler):
        b, c, h, w = y_rounded.size()
        y_sampled = y_sampler(y_rounded)
        h_sampled = h_sampler(h_tilde)
        merged = torch.cat([y_sampled, h_sampled], 1)
        y_context = self.transform(merged)
        y_context = self.flatten(y_context)
        y_context = self.fc(y_context)
        hyper_mu = y_context[:, :c]
        hyper_mu = hyper_mu.view(b, h, w, c).permute(0, 3, 1, 2)
        hyper_sigma = y_context[:, c:]
        hyper_sigma = torch.exp(hyper_sigma)
        hyper_sigma = hyper_sigma.contiguous().view(b, h, w, c).permute(0, 3, 1, 2)

        return hyper_mu, hyper_sigma


class conv_generator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(conv_generator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.transform = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, out_dim * 3)
        )

    def forward(self, x):
        b, _, _, _ = x.shape
        x = x.view(b, -1)

        weights = self.transform(x)
        weights = weights.view(b, 3, self.out_dim, 1, 1)

        return weights


class Syntax_Model(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Syntax_Model, self).__init__()
        self.down0 = nn.Conv2d(in_dim, 32, 3, 2, 1)
        self.down1 = nn.Conv2d(32, 64, 3, 2, 1)

        self.conv = nn.Conv2d(in_dim + 32 + 64, out_dim, 1, 1, 0)
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, syntax):
        out1 = self.pooling(syntax)

        ds1 = self.down0(syntax)
        ds1 = F.relu(ds1)
        out2 = self.pooling(ds1)

        ds2 = self.down1(ds1)
        ds2 = F.relu(ds2)
        out3 = self.pooling(ds2)

        out = torch.cat((out1, out2, out3), 1)
        out = self.conv(out)
        return out


class PredictionModel_Syntax(nn.Module):
    def __init__(self, in_dim, dim=192, trainable=True, outdim=None):
        super(PredictionModel_Syntax, self).__init__()
        if outdim is None:
            outdim = dim

        self.down0 = nn.Conv2d(in_dim, dim, 3, 2, 1)
        self.down1 = nn.Conv2d(dim, dim, 3, 2, 1)
        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(dim * 2 + in_dim, outdim)
        self.flatten = nn.Flatten()

    def forward(self, y_rounded, h_tilde, h_sampler=None):
        b, c, h, w = y_rounded.size()

        ds0 = self.down0(h_tilde)
        ds0 = F.relu(ds0)

        ds1 = self.down1(ds0)
        ds1 = F.relu(ds1)

        ds0 = self.pooling(ds0)
        ds1 = self.pooling(ds1)
        ori = self.pooling(h_tilde)
        y_context = torch.cat((ori, ds0, ds1), 1)

        y_context = self.flatten(y_context)
        y_context = self.fc(y_context)
        hyper_mu = y_context[:, :c]
        hyper_mu = hyper_mu.view(b, h, w, c).permute(0, 3, 1, 2)
        hyper_sigma = y_context[:, c:]
        hyper_sigma = torch.exp(hyper_sigma)
        hyper_sigma = hyper_sigma.contiguous().view(b, h, w, c).permute(0, 3, 1, 2)

        return hyper_mu, hyper_sigma


class BlockSample(nn.Module):
    def __init__(self, in_shape, masked=True):
        super(BlockSample, self).__init__()
        self.masked = masked
        dim = in_shape[1]
        flt = np.zeros((dim * 16, dim, 7, 7), dtype=np.float32)
        for i in range(0, 4):
            for j in range(0, 4):
                if self.masked:
                    if i == 3:
                        if j == 2 or j == 3:
                            break
                for k in range(0, dim):
                    s = k * 16 + i * 4 + j
                    flt[s, k, i, j + 1] = 1
        flt_tensor = torch.Tensor(flt).float().cuda()
        self.register_buffer('sample_filter', flt_tensor)

    def forward(self, inputs):
        t = F.conv2d(inputs, self.sample_filter, padding=3)
        b, c, h, w = inputs.size()
        # print(b, c, h, w)
        t = t.contiguous().view(b, c, 4, 4, h, w).permute(0, 4, 5, 1, 2, 3)
        t = t.contiguous().view(b * h * w, c, 4, 4)
        return t


class BypassRound(Function):
    @staticmethod
    def forward(ctx, inputs):
        return torch.round(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


bypass_round = BypassRound.apply


class TestModel(nn.Module):
    def __init__(self, train_size, test_size, is_high):
        super().__init__()
        self.train_size = train_size
        self.test_size = test_size
        self.is_high = is_high

        b, c, h, w = train_size
        tb, tc, th, tw = test_size

        if self.is_high:
            # N = 384
            # M = 32
            N = 576
            M = c
        else:
            N = 192
            M = 16
        self.M = M
        self.N = N

        self.a_model = analysisTransformModel(c, [N, N, N, N])
        self.s_model = synthesisTransformModel(N - M, [N, N, N, M])

        self.syntax_model = Syntax_Model(M, M)
        self.conv_weights_gen = conv_generator(in_dim=M, out_dim=M)

        self.ha_model = h_analysisTransformModel(N, [N, N, N], [1, 2, 2])
        self.hs_model = h_synthesisTransformModel(N, [N, N, N], [2, 2, 1])

        self.v_z2_sigma = nn.Parameter(torch.ones((1, N, 1, 1), dtype=torch.float32, requires_grad=True))
        self.register_parameter('z2_sigma', self.v_z2_sigma)

        self.entropy_bottleneck_z2 = GaussianModel()
        self.entropy_bottleneck_z3 = GaussianModel()
        self.entropy_bottleneck_z3_syntax = GaussianModel()

        self.prediction_model = PredictionModel_Context(in_dim=2 * N - M, dim=N, outdim=(N - M) * 2)
        self.prediction_model_syntax = PredictionModel_Syntax(in_dim=N, dim=M, outdim=M * 2)

        self.y_sampler = BlockSample((b, N - M, h // 8, w // 8))
        self.h_sampler = BlockSample((b, N, h // 8, w // 8), False)
        self.test_y_sampler = BlockSample((b, N - M, th // 8, tw // 8))
        self.test_h_sampler = BlockSample((b, N, th // 8, tw // 8), False)

    def batch_conv(self, weights, inputs):
        b, ch, _, _ = inputs.shape
        _, ch_out, _, k, _ = weights.shape

        weights = weights.reshape(b * ch_out, ch, k, k)
        inputs = torch.cat(torch.split(inputs, 1, dim=0), dim=1)
        out = F.conv2d(inputs, weights, stride=1, padding=0, groups=b)
        out = torch.cat(torch.split(out, ch_out, dim=1), dim=0)

        return out

    def forward(self, inputs, mode='train'):
        b, c, h, w = self.train_size
        tb, tc, th, tw = self.test_size

        z3 = self.a_model(inputs)
        z2 = self.ha_model(z3)
        # print("z3 =", z3.shape)

        noise = torch.rand_like(z2) - 0.5
        z2_noisy = z2 + noise
        z2_rounded = bypass_round(z2)

        h2 = self.hs_model(z2_rounded)
        z2_sigma = self.z2_sigma.cuda()
        z2_mu = torch.zeros_like(z2_sigma)

        z3_syntax = z3[:, :self.M, :, :]
        z3_syntax = self.syntax_model(z3_syntax)
        z3_content = z3[:, self.M:, :, :]
        # print("z3_content =", z3_content.shape)

        # Content
        noise = torch.rand_like(z3_content) - 0.5
        z3_content_noisy = z3_content + noise
        z3_content_rounded = bypass_round(z3_content)
        # print("z3_content_rounded =", z3_content_rounded.shape)

        # Syntax
        noise = torch.rand_like(z3_syntax) - 0.5
        z3_syntax_noisy = z3_syntax + noise
        z3_syntax_rounded = bypass_round(z3_syntax)

        if mode == 'train':
            z2_likelihoods = self.entropy_bottleneck_z2(z2_noisy, z2_sigma, z2_mu)
            # print("z2_likelihoods =", z2_likelihoods.shape)

            # Content
            z3_content_mu, z3_content_sigma = self.prediction_model(z3_content_rounded, h2, self.y_sampler,
                                                                    self.h_sampler)
            # print("z3_content_mu =", z3_content_mu.shape)
            # print("z3_content_sigma =", z3_content_sigma.shape)
            z3_content_likelihoods = self.entropy_bottleneck_z3(z3_content_noisy, z3_content_sigma, z3_content_mu)

            # Syntax
            z3_syntax_sigma, z3_syntax_mu = self.prediction_model_syntax(z3_syntax_rounded, h2)
            z3_syntax_likelihoods = self.entropy_bottleneck_z3_syntax(z3_syntax_noisy, z3_syntax_sigma, z3_syntax_mu)
            # print("z3_syntax_likelihoods =", z3_syntax_likelihoods.shape)

        else:
            z2_likelihoods = self.entropy_bottleneck_z2(z2_rounded, z2_sigma, z2_mu)

            # Content
            z3_content_mu, z3_content_sigma = self.prediction_model(z3_content_rounded, h2, self.test_y_sampler,
                                                                    self.test_h_sampler)
            z3_content_likelihoods = self.entropy_bottleneck_z3(z3_content_rounded, z3_content_sigma, z3_content_mu)

            # Syntax
            z3_syntax_sigma, z3_syntax_mu = self.prediction_model_syntax(z3_syntax_rounded, h2)
            z3_syntax_likelihoods = self.entropy_bottleneck_z3_syntax(z3_syntax_rounded, z3_syntax_sigma, z3_syntax_mu)
        # print("z3_content_rounded =", z3_content_rounded.shape)
        x_tilde = self.s_model(z3_content_rounded)
        conv_weights = self.conv_weights_gen(z3_syntax_rounded)

        x_tilde_bf = self.batch_conv(conv_weights, x_tilde)

        # if self.post_processing:
        #     x_tilde = self.HAN(x_tilde_bf)
        #     conv_weights = self.conv_weights_gen_HAN(z3_syntax_rounded)
        #     x_tilde = self.batch_conv(conv_weights, x_tilde)
        #     x_tilde = self.add_mean(x_tilde)
        # else:
        #     x_tilde = x_tilde_bf

        num_pixels = inputs.size()[0] * h * w

        # print("x_tilde =", x_tilde.shape)
        # print("inputs =", inputs.shape)

        if mode == 'train':

            bpp_list = [torch.sum(torch.log(l), [0, 1, 2, 3]) / (-np.log(2) * num_pixels) for l in
                        [z2_likelihoods, z3_content_likelihoods, z3_syntax_likelihoods]]
            # bpp_list = [torch.sum(torch.log(l), [0, 1, 2, 3]) / (-np.log(2) * num_pixels) for l in
            #             [z2_likelihoods, z3_content_likelihoods]]

            train_bpp = bpp_list[0] + bpp_list[1] + bpp_list[2]

            train_mse = torch.mean((inputs[:, :, :h, :w] - x_tilde[:, :, :h, :w]) ** 2, [0, 1, 2, 3])
            train_mse *= 255 ** 2

            return train_bpp, train_mse, x_tilde, z3_syntax_likelihoods


        elif mode == 'test':
            test_num_pixels = inputs.size()[0] * th * tw

            bpp_list = [torch.sum(torch.log(l), [0, 1, 2, 3]) / (-np.log(2) * test_num_pixels) for l in
                        [z2_likelihoods, z3_content_likelihoods, z3_syntax_likelihoods]]
            # bpp_list = [torch.sum(torch.log(l), [0, 1, 2, 3]) / (-np.log(2) * test_num_pixels) for l in
            #             [z2_likelihoods, z3_content_likelihoods]]

            eval_bpp = bpp_list[0] + bpp_list[1] + bpp_list[2]

            # Bring both images back to 0..255 range.
            gt = torch.round((inputs + 1) * 127.5)
            x_hat = torch.clamp((x_tilde + 1) * 127.5, 0, 255)
            x_hat = torch.round(x_hat).float()

            v_mse = torch.mean((x_hat - gt) ** 2, [1, 2, 3])
            v_psnr = torch.mean(20 * torch.log10(255 / torch.sqrt(v_mse)), 0)

            return eval_bpp, v_mse, v_psnr, x_tilde, z3_syntax_likelihoods


if __name__ == '__main__':
    net = TestModel((5, 220, 64, 64), (1, 220, 64, 64), is_high=True).cuda()
    img = torch.randn(5, 220, 4, 4).cuda()
    img = F.interpolate(img, scale_factor=16, mode='bilinear', align_corners=False)
    # net = TestModel((5, 220, 16, 16), (1, 220, 16, 16), is_high=True).cuda()
    # img = torch.randn(5, 220, 16, 16).cuda()
    train_bpp, train_mse, x_tilde, z3_syntax_likelihoods = net(img, mode='train')
    print("train_bpp =", train_bpp)
    print("train_mse =", train_mse)
    print("x_tilde =", x_tilde.shape)
    print("z3_syntax_likelihoods =", z3_syntax_likelihoods.shape)
    x_tilde = F.interpolate(x_tilde, scale_factor=1/16, mode='bilinear', align_corners=False)
    print("x_tilde =", x_tilde.shape)
    # eval_bpp, v_mse, v_psnr, x_tilde, z3_syntax_likelihoods = net(img, mode='test')

    # img = cv2.imread('data_0.bmp')
    # img2 = cv2.imread('data_1.bmp')
    # img = img[-256:, -256:, :]
    # img2 = img2[:256, :256, :]

    # cv2.imshow('img1', img)
    # cv2.waitKey(0)
    # cv2.imshow('img2', img2)
    # cv2.waitKey(0)
    # cv2.imwrite('split_img1.bmp', img)
    # cv2.imwrite('split_img2.bmp', img2)

    # img_tensor = torch.from_numpy(img).permute(2, 0, 1).type(torch.float32)
    # img_tensor2 = torch.from_numpy(img2).permute(2, 0, 1).type(torch.float32)
    # img_stack = torch.stack((img_tensor, img_tensor2))
    # # print(img_stack.shape)
    # net = TestModel((2, 3, 256, 256), (1, 3, 256, 256), is_high=True).cuda()
    # img_stack = img_stack.cuda()

    # summary(net, input_size=[(3, 256, 256)])

    # train_bpp, train_mse, x_tilde = net(img_stack, mode='train')
    # eval_bpp, v_mse, v_psnr, x_tilde = net(img_stack, mode='test')
    # for i in range(len(x_tilde)):
    #     image = x_tilde[i].permute(1, 2, 0).cpu().detach().numpy()
    #     print("image.shape =", image.shape)
    #     cv2.imwrite('res_' + str(i+1) + '.bmp', image)
