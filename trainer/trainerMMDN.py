import cv2
import torch
import os
import sys

from torchvision.models import AlexNet_Weights, ResNet50_Weights

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
import random
import numpy as np
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import datetime
import logging
from tqdm import tqdm
from skimage.metrics import mean_squared_error as mse_sk
from skimage.metrics import structural_similarity as ssim_sk
# from paddle_msssim import ssim, ms_ssim
from pytorch_msssim import ssim, ms_ssim
from utils import get_feature, fea2img, psnr1, dnorm, psnr255, switch_optimizer_tensor_to_device
from dataset_loader import dataset_train_or_val_loader
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from torch import nn

from load_checkpoint import Fusion_load
from networks.MMDN import Discriminator, PreprocessText
import torchvision.models as models

# import torchvision.transforms as transforms
# 异常检测开启
torch.autograd.set_detect_anomaly(True)


def trainerMMDN(args, model):
    torch.manual_seed(args.seed)
    version = args.version  # 划分方法
    mode = args.mode  # 模型组合方式
    save_per_epoch = args.save_per_epoch  # 保存模型文件的频率
    # save_name = "only-mseloss"
    # save_name = "xx_full_loss"
    # save_name = "xy_full_loss"
    # save_name = "yx_full_loss"
    # save_name = "CmJND_full_loss"
    # save_name = "yx_mae_mse"
    # save_name = "xx_mae_mse"
    # save_name = "xy_mae_mse"
    # save_name = "CmJND_loss_to_save"
    # save_name = "CmJND_mae_mse"
    # save_name = "rrr_mae_mse"
    # save_name = "srr_mae_mse"
    # save_name = "rrs_mae_mse"
    # save_name = "rdr_mae_mse"
    # save_name = "sesese_mae_mse"
    # save_name = "dedede_mae_mse"
    # save_name = "sasasa_mae_mse"
    # save_name = "rgb_only_mse"
    cp = args.checkpoint
    model_name = args.model_name  # MLLMJND

    # save_mode_path = "../checkpoint/" + save_name + '.pth'
    # log_path = "../checkpoint/" + save_name + '.txt'

    save_mode_file = "./checkpoint/" + model_name + "/" + version + '/' + mode + '/'
    save_mode_path = save_mode_file + cp + '.pth'
    log_path = save_mode_file + cp + '.txt'
    # 检查文件夹是否存在
    if not os.path.exists(save_mode_file):
        # 如果文件夹不存在，则创建它
        os.makedirs(save_mode_file)

    # input_prompt = """ Please provide a description of this image, which requires: (1) The description should include
    #     at least one brightness word to reflect the brightness adaptation effect of the human eye. (2) The
    #     description should include descriptors with different levels of darkness. (3) The description should
    #     include object and attribute descriptors. """

    base_lr = args.base_lr
    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()
    # print(dis.parameters())
    # print(model.parameters())

    # 创建ResNet-50模型实例
    dis = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).cuda()
    num_features = dis.fc.in_features
    dis.fc = nn.Sequential(
        nn.Linear(num_features, 2),
        nn.Sigmoid()
    )
    # nn.init.xavier_uniform_(dis.fc.weight)
    # nn.init.zeros_(dis.fc.bias)
    # dis.sigmoid = nn.Sigmoid()
    # for param in dis.parameters():
    #     param.requires_grad = False
    d_optim = optim.Adam(dis.parameters(), lr=0.01)
    bce_loss = torch.nn.BCELoss()
    bce_logit_loss = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=base_lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=3, verbose=True)
    writer = SummaryWriter(logdir='./train/' + model_name + '/' + version + '/' + mode + '/' + cp + '/logs')

    # 加载之前训练的文件
    print("loading the best checkpoint_data <path>'" + save_mode_path + "' ...")
    loader = Fusion_load(model=model, checkpoint=save_mode_path, optimizer=optimizer, scheduler=scheduler, dis=dis,
                         d_optim=d_optim)
    model, dis, flag = loader.load_checkpoint()
    # loader = Fusion_load(model=model, checkpoint=save_mode_path, optimizer=optimizer, scheduler=scheduler)
    # model, flag = loader.load_checkpoint()
    # model = nn.DataParallel(model).cuda()

    # 预训练AlexNet , pretrained=True
    alexnet = models.alexnet(weights=AlexNet_Weights.DEFAULT).cuda()

    model = model.cuda()
    dis = dis.cuda()
    switch_optimizer_tensor_to_device(optimizer)
    switch_optimizer_tensor_to_device(d_optim)

    max_epoch = args.max_epochs

    # 如果没有加载checkpoint
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    if not flag:
        logging.info("***************************************")
        logging.info("log save to " + log_path)
        logging.info("***************************************")
        logging.info(cp)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    root = args.root_path
    data_dir = root + version + "/Train_data"
    # data_dir = "../data202/" + version + "/Val_data"
    print(" --------- Train data dir: ", data_dir)
    db_train = dataset_train_or_val_loader(data_dir=data_dir, transform=transform)
    if not flag:
        logging.info("The length of train set is: %d" % (len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, pin_memory=False,
                              worker_init_fn=worker_init_fn, num_workers=0)
    if not flag:
        logging.info("==========================================")
        logging.info('batch size is: %d ' % args.batch_size)
        logging.info("==================================begin!!!!!!============================================")

    # total_start = time.time()
    total_cost_time = loader.total_cost_time
    # best_model_state_dict = None
    # best_optimizer_state_dict = None
    # best_scheduler_state_dict = None
    # best_dis_state_dict = None
    # best_d_optim_state_dict = None
    best_model_state_dict = model.state_dict()
    best_optimizer_state_dict = optimizer.state_dict()
    best_scheduler_state_dict = scheduler.state_dict()
    best_dis_state_dict = dis.state_dict()
    best_d_optim_state_dict = d_optim.state_dict()
    max_avg_psnr = loader.max_avg_psnr
    print("max_avg_psnr loaded is", max_avg_psnr)
    # avg_psnr_list = []
    # epoch_avg_loss_list = []
    # with tqdm(total= max_epoch) as pbar:
    print("epoch from", loader.epoch + 1, "to", max_epoch)
    for epoch_num in range(loader.epoch + 1, max_epoch + 1):
        model.cuda()
        model.train()
        loop = tqdm(range(len(train_loader)), total=len(train_loader))
        epoch_start = time.time()
        loss_list = []
        dis_loss_list = []
        gen_loss_list = []
        r_loss_list = []
        rcst_loss_list = []
        p_loss_list = []
        sc_loss_list = []
        v_mse_list = []
        v_psnr_list = []
        epoch_lr = 0
        for param_group in optimizer.param_groups:
            epoch_lr = param_group['lr']
        for idx, (sampled_batch, _) in enumerate(train_loader):
            image_batch, text_batch = sampled_batch['image'], sampled_batch['text']
            image_batch = image_batch.cuda()
            # print("image_batch:", image_batch.shape)
            # print("label_batch:", label_batch.shape)
            # print("depth_batch:", depth_batch.shape)
            # print("salient_batch:", salient_batch.shape)
            # print("seg_batch:", seg_batch.shape)
            # print("text_batch:", len(text_batch))

            # preprocess = PreprocessText(device=image_batch.device, glove=model.glove)
            # preprocessed_text = preprocess(text_batch)
            #
            # t, (h, c) = text_encoder(preprocessed_text)

            # sys.exit()
            # outputs, fr_out, cm_out = model(image_batch, salient_batch, depth_batch, seg_batch)
            # outputs, fr_out, cm_out = model(image_batch, image_batch, image_batch, image_batch) #rrr
            # outputs, fr_out, cm_out = model(image_batch, salient_batch, image_batch, image_batch) #srr
            # outputs, fr_out, cm_out = model(image_batch, image_batch, depth_batch, image_batch) #rdr
            # outputs, fr_out, cm_out = model(image_batch, image_batch, image_batch, seg_batch) #rrs
            # outputs, fr_out, cm_out = model(image_batch, seg_batch, seg_batch, seg_batch) #sesese
            # outputs, fr_out, cm_out = model(image_batch, depth_batch, depth_batch, depth_batch) #dedede

            # rgb_only
            # 看看cm_out和fr_out图(不一样的)
            # print(fr_out.shape, cm_out.shape)
            # cv2.imshow("fr_out", fr_out[0][0].detach().cpu().numpy())
            # cv2.imshow( "cm_out", cm_out[0][0].detach().cpu().numpy())
            # cv2.waitKey(0)

            # loss = mse_loss(outputs, label_batch)
            # loss = 0.0001 * mae_loss(fr_out, cm_out) + mse_loss(outputs, label_batch)
            # loss = mse_loss(outputs, label_batch)
            # loss = torch.mean(torch.abs(0.4 - torch.lt(label_batch, outputs).float()) * torch.pow(outputs - label_batch, 2))
            # loss = 0.0001 * mae_loss(fr_out, cm_out) + torch.mean(
            #     torch.abs(0.4 - torch.lt(label_batch, outputs).float()) * torch.pow(outputs - label_batch, 2))
            # logging.info('epoch %d : batch_loss: %f' % (epoch_num, loss_.item()))

            # 计算损失
            loss = 0.0
            dis_loss = 0.0
            gen_loss = 0.0
            r_loss = 0.0
            rcst_loss = 0.0
            p_loss = 0.0
            sc_loss = 0.0

            # # 一、GAN Loss
            # # 1) Discriminator
            # d_optim.zero_grad()
            # # Discriminator 1.Real Image Loss
            # real_output = dis(image_batch, text_batch)
            # d_real_loss = bce_loss(real_output, torch.ones_like(real_output))
            # d_real_loss.backward()
            # loss += d_real_loss.item()
            # dis_loss += d_real_loss.item()
            # # Discriminator 2.Fake Image Loss
            # cf_1, cf_2, outputs, l_it, encoded_out_img, encoded_ori_img = model(image_batch, text_batch)
            # fake_output = dis(outputs.detach(), text_batch)
            # d_fake_loss = bce_loss(fake_output, torch.zeros_like(fake_output))
            # d_fake_loss.backward()
            # loss += d_fake_loss.item()
            # dis_loss += d_fake_loss.item()
            # d_optim.step()

            # # 三、Reconstruction Loss
            # reconstruction_loss = mse_loss(image_batch, outputs)
            # # reconstruction_loss.backward()  #
            # loss += reconstruction_loss.item()
            # rcst_loss += reconstruction_loss.item()

            # # 四、Perceptual Loss
            # with torch.no_grad():
            #     x = alexnet(image_batch)
            #     x_hat = alexnet(outputs)
            # perceptual_loss = mse_loss(x, x_hat)
            # # perceptual_loss.backward()  #
            # loss += perceptual_loss.item()
            # p_loss += perceptual_loss.item()

            # # 2) Generator
            # optimizer.zero_grad()
            # gen_output = dis(outputs, text_batch)
            # g_loss = bce_loss(gen_output, torch.ones_like(gen_output))
            # g_loss.backward()
            # loss += g_loss.item()
            # gen_loss += g_loss.item()

            # # 二、Rate Loss
            # rate_loss = 0.00001 * mse_loss(cf_1, cf_2)
            # # print("rate_loss =", rate_loss)
            # rate_loss.backward()
            # loss += rate_loss.item()
            # r_loss += rate_loss.item()

            # # 五、Multimodal Semantic-Consistent Loss
            # msc_loss = l_it + mse_loss(encoded_out_img, encoded_ori_img)
            # # msc_loss.backward()  #
            # # loss += msc_loss.item()
            # sc_loss += msc_loss.item()

            # 反向传播时检测是否有异常值，定位code
            # with torch.autograd.detect_anomaly():
            #     t_loss = reconstruction_loss + perceptual_loss + g_loss + rate_loss + msc_loss + d_fake_loss + d_real_loss
            #     t_loss.backward()
            #
            # loss += t_loss.item()

            # d_optim.step()

            # new code
            # 1) Discriminator
            # Discriminator 1.Real Image Loss
            # with torch.no_grad():
            d_optim.zero_grad()
            real_output = dis(image_batch)
            # print(real_output.shape)
            # print(torch.ones_like(real_output).shape)
            d_real_loss = bce_logit_loss(real_output, torch.ones_like(real_output))
            # d_real_loss.backward()
            # loss += d_real_loss.item()
            # dis_loss += d_real_loss.item()
            # Discriminator 2.Fake Image Loss
            cf_1, cf_2, outputs, l_it, encoded_out_img, encoded_ori_img = model(image_batch, text_batch)
            # outputs = torch.clamp((outputs + 1) * 127.5, 0, 255)
            # outputs = torch.round(outputs).float()
            outputs = outputs + image_batch
            # outputs = (outputs + 1.) / 2.0
            # outputs.clamp_(0., 1.)
            # outputs = outputs * 255.0
            # outputs = torch.round(outputs)
            # outputs = torch.clamp(outputs, 0, 255)
            # with torch.no_grad():
            fake_output = dis(outputs.detach())
            d_fake_loss = bce_logit_loss(fake_output, torch.zeros_like(fake_output))
            # d_fake_loss.backward()
            # loss += d_fake_loss.item()
            # dis_loss += d_fake_loss.item()
            d_optim.step()

            # 三、Reconstruction Loss
            reconstruction_loss = mse_loss(image_batch, outputs)
            # reconstruction_loss.backward()  #
            # loss += reconstruction_loss.item()
            # rcst_loss += reconstruction_loss.item()

            # 四、Perceptual Loss
            with torch.no_grad():
                x = alexnet(image_batch)
                x_hat = alexnet(outputs)
            perceptual_loss = mse_loss(x, x_hat)
            # perceptual_loss.backward()  #
            # loss += perceptual_loss.item()
            # p_loss += perceptual_loss.item()

            # 2) Generator
            optimizer.zero_grad()
            # with torch.no_grad():
            gen_output = dis(outputs)
            g_loss = bce_logit_loss(gen_output, torch.ones_like(gen_output))
            # g_loss.backward()
            # loss += g_loss.item()
            # gen_loss += g_loss.item()

            # 二、Rate Loss
            rate_loss = mse_loss(cf_1, cf_2)
            # print("rate_loss =", rate_loss)
            # rate_loss.backward()
            # loss += rate_loss.item()
            # r_loss += rate_loss.item()

            # 五、Multimodal Semantic-Consistent Loss
            msc_loss = l_it + mse_loss(encoded_out_img, encoded_ori_img)
            # msc_loss.backward()  #
            # loss += msc_loss.item()
            # sc_loss += msc_loss.item()

            losses = torch.stack([d_real_loss, d_fake_loss, perceptual_loss, g_loss, rate_loss])
            weighted_loss = model.compute_weighted_loss(losses)
            weighted_loss.backward()
            loss += weighted_loss.item()

            # 自身
            # gt = torch.round((image_batch + 1) * 127.5)
            gt = image_batch
            x_hat = outputs
            v_mse = torch.mean((x_hat - gt) ** 2, [1, 2, 3])
            v_psnr = torch.mean(20 * torch.log10(255 / torch.sqrt(v_mse)), 0).item()
            v_mse = torch.mean(v_mse).item()
            v_mse_list.append(v_mse)
            v_psnr_list.append(v_psnr)

            optimizer.step()

            loss_list.append(loss)
            # dis_loss_list.append(dis_loss)
            # gen_loss_list.append(gen_loss)
            # r_loss_list.append(r_loss)
            # rcst_loss_list.append(rcst_loss)
            # p_loss_list.append(p_loss)
            # sc_loss_list.append(sc_loss)
            # MSE_loss_list.append(MSE_loss.item() * 10000)

            # 更新信息
            # loop.set_description(f'Item [{epoch_idx + 1}/{10}] Epoch [{epochs + 1}/{100}]')
            loop.set_description(f'Train Epoch [{epoch_num}/{max_epoch}] Item [{idx + 1}/{len(train_loader)}]')
            # loop.set_postfix(loss=loss, dis_loss=dis_loss, gen_loss=gen_loss, r_loss=r_loss, rcst_loss=rcst_loss,
            #                  p_loss=p_loss, sc_loss=sc_loss, v_mse=v_mse, v_psnr=v_psnr)
            loop.set_postfix(loss=loss, v_mse=v_mse, v_psnr=v_psnr)
            # loop.set_postfix(loss=loss)
            loop.update(1)
        logging.info('===============epoch_lr: %e ===============' % epoch_lr)
        epoch_avg_loss = np.mean(loss_list).item()
        # epoch_avg_dis_loss = np.mean(dis_loss_list).item()
        # epoch_avg_gen_loss = np.mean(gen_loss_list).item()
        # epoch_avg_r_loss = np.mean(r_loss_list).item()
        # epoch_avg_rcst_loss = np.mean(rcst_loss_list).item()
        # epoch_avg_p_loss = np.mean(p_loss_list).item()
        # epoch_avg_sc_loss = np.mean(sc_loss_list).item()
        epoch_avg_v_mse = np.mean(v_mse_list).item()
        epoch_avg_v_psnr = np.mean(v_psnr_list).item()
        # epoch_avg_loss_list.append(epoch_avg_loss)
        # min_loss = np.min(epoch_avg_loss_list)
        # if epoch_avg_loss <= min_loss:
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info(
        #         "*** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ")
        #     logging.info("save model.state_dict() to " + save_mode_path + '...')
        #     logging.info(
        #         "*** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ")

        scheduler.step(epoch_avg_loss)
        logging.info('epoch %d : epoch_avg_loss: %f' % (epoch_num, epoch_avg_loss))
        # writer.add_scalar('info/total_loss', loss, epoch_num)
        # writer.add_scalar('info/epoch_avg_loss', epoch_avg_loss, epoch_num)
        writer.add_scalar('Total Loss', epoch_avg_loss, epoch_num)
        # writer.add_scalar('Discriminator Loss', epoch_avg_dis_loss, epoch_num)
        # writer.add_scalar('Generator Loss', epoch_avg_gen_loss, epoch_num)
        # writer.add_scalar('Rate Loss', epoch_avg_r_loss, epoch_num)
        # writer.add_scalar('Reconstruction Loss', epoch_avg_rcst_loss, epoch_num)
        # writer.add_scalar('Perceptual Loss', epoch_avg_p_loss, epoch_num)
        # writer.add_scalar('Multimodal Semantic-Consistent Loss', epoch_avg_sc_loss, epoch_num)
        writer.add_scalar('V MSE', epoch_avg_v_mse, epoch_num)
        writer.add_scalar('V PSNR', epoch_avg_v_psnr, epoch_num)

        val_dir = root + version + "/Val_data"
        # val_dir = root + version + "/Test_data"
        db_val = dataset_train_or_val_loader(data_dir=val_dir, transform=transform)
        logging.info('The length of Val set is: %d' % (len(db_val)))
        val_loader = DataLoader(db_val, batch_size=2, shuffle=False, num_workers=0)
        logging.info('Val: epoch %d: %d test iterations per epoch' % (epoch_num, len(val_loader)))
        mse_out_gt_loss_list, mse_out_org_loss_list = [], []
        # PSNR
        psnr_out_org_list = []
        # MSE
        mse_out_org_list = []
        # SSIM
        # ssim_out_gt_list, ssim_out_org_list = [], []
        # MS-SSIM
        # ms_ssim_out_gt_list, ms_ssim_out_org_list = [], []

        outputs_mean_list, label_mean_list, image_mean_list = [], [], []
        loop = tqdm(range(len(val_loader)), total=len(val_loader))

        avg_psnr_list = []  # limit

        model.eval()
        with torch.no_grad():
            for idx, (sampled_batch, _) in enumerate(val_loader):
                image, text = sampled_batch['image'], sampled_batch['text']
                image = image.cuda()

                # preprocess = PreprocessText(device=image.device, glove=model.glove)
                # preprocessed_text = preprocess(text)

                cf_1, cf_2, outputs, l_it, encoded_out_img, encoded_ori_img = model(image, text)  # rgb_only

                # image = image.squeeze()
                # outputs = outputs.squeeze()
                # outputs = (outputs + 1.) / 2.0
                # outputs.clamp_(0., 1.)
                # image = (image + 1.) / 2.0
                # image.clamp_(0., 1.)
                # print("image =", image.shape)
                # print("outputs =", outputs.shape)

                outputs = outputs + image
                # image = torch.round((image + 1) * 127.5)
                outputs = torch.clamp((outputs + 1) * 127.5, 0, 255)
                outputs = torch.round(outputs).float()

                v_mse = torch.mean((outputs - image) ** 2, [1, 2, 3])
                v_psnr = torch.mean(20 * torch.log10(255 / torch.sqrt(v_mse)), 0).item()

                len_of_outputs = len(outputs)

                for i in range(0, len_of_outputs):
                    # out = outputs[i].cpu().numpy() * 255
                    # img = image[i].cpu().numpy() * 255
                    out = outputs[i].cpu().numpy()
                    img = image[i].cpu().numpy()
                    # print("np.max(out) =", np.max(out), "np.min(out) =", np.min(out))
                    # print("np.max(img) =", np.max(img), "np.min(img) =", np.min(img))
                    # out[out > 255] = 255
                    # out[out < 0] = 0
                    out = out.astype(np.uint8())
                    img = img.astype(np.uint8())
                    # jnd = jnd.astype(np.uint8())

                    # print(out.shape)
                    # print(jnd.shape)

                    # out = outputs[i].cpu().numpy()
                    # img = image[i].cpu().numpy()
                    # PSNR
                    psnr_out_org = psnr_sk(out, img)
                    psnr_out_org_list.append(psnr_out_org)

                    # SSIM
                    # out = torch.tensor(out, dtype=torch.float32).unsqueeze(0)
                    # jnd = torch.tensor(jnd, dtype=torch.float32).unsqueeze(0)
                    # img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
                    # ssim_out_gt = ssim(out, jnd)
                    # ssim_out_org = ssim(out, img)
                    # ssim_out_gt_list.append(ssim_out_gt)
                    # ssim_out_org_list.append(ssim_out_org)

                    # MS-SSIM
                    # ms_ssim_out_gt = ms_ssim(out, jnd)
                    # ms_ssim_out_org = ms_ssim(out, img)
                    # ms_ssim_out_gt_list.append(ms_ssim_out_gt)
                    # ms_ssim_out_org_list.append(ms_ssim_out_org)

                # 计算平均值
                # PSNR
                total_psnr_out_org = np.mean(psnr_out_org_list, dtype=np.float64)

                # MSE
                # total_mse_out_org = np.mean(mse_out_org_list, dtype=np.float64)

                # SSIM
                # total_ssim_out_gt = np.mean(ssim_out_gt_list, dtype=np.float64)
                # total_ssim_out_org = np.mean(ssim_out_org_list, dtype=np.float64)

                # MS-SSIM
                # total_ms_ssim_out_gt = np.mean(ms_ssim_out_gt_list, dtype=np.float64)
                # total_ms_ssim_out_org = np.mean(ms_ssim_out_org_list, dtype=np.float64)

                # 更新信息
                # loop.set_description(f'Item [{epoch_idx + 1}/{10}] Epoch [{epochs + 1}/{100}]')
                loop.set_description(
                    f'Valid Epoch [{epoch_num}/{max_epoch}] Item [{idx + 1}/{len(val_loader)}]')
                loop.set_postfix(psnr=total_psnr_out_org, v_psnr=v_psnr)
                loop.update(1)

            # one validation requires recording complete data once
            # mse_loss_validation = np.mean(mse_out_gt_loss_list)
            psnr_validation = np.mean(psnr_out_org_list, dtype=np.float64)
            # mse_validation = np.mean(mse_out_gt_list, dtype=np.float64)
            # ssim_validation = np.mean(ssim_out_org_list, dtype=np.float64)
            # ms_ssim_validation = np.mean(ms_ssim_out_gt_list, dtype=np.float64)

            # save to file 'checkpoint/XX.txt'
            # MSE_LOSS
            # logging.info('total_mse_out_gt_loss： %f' % mse_loss_validation)
            # logging.info('total_mse_out_org_loss： %f' % (np.mean(mse_out_org_loss_list)))

            # PSNR
            logging.info('* total_psnr_out_org： %f' % psnr_validation)

            # MSE
            # logging.info('* total_mse_out_gt： %f' % mse_validation)
            # logging.info('total_mse_out_org： %f' % np.mean(mse_out_org_list, dtype=np.float64))

            # SSIM
            # logging.info('* total_ssim_out_gt： %f' % ssim_validation)
            # logging.info('total_ssim_out_org： %f' % np.mean(ssim_out_org_list, dtype=np.float64))

            # MS-SSIM
            # logging.info('* total_ms_ssim_out_gt： %f' % ms_ssim_validation)
            # logging.info('total_ms_ssim_out_org： %f' % np.mean(ms_ssim_out_org_list, dtype=np.float64))

            # Mean
            # logging.info('total_image_mean： %f' % (np.mean(image_mean_list)))
            # logging.info('total_outputs_mean： %f' % (np.mean(outputs_mean_list)))
            # logging.info('total_label_mean： %f' % (np.mean(label_mean_list)))

            # save to tensorboardX
            writer.add_scalar('PSNR', psnr_validation, epoch_num)
            # writer.add_scalar('SSIM', ssim_validation, epoch_num)
            # writer.add_scalar('MS_SSIM', ms_ssim_validation, epoch_num)

            epoch_times = time.time() - epoch_start
            total_cost_time += epoch_times

            avg_psnr = np.mean(psnr_out_org_list)
            print("avg_psnr = ", avg_psnr)

            # 比较平均psnr，大于max_avg_psnr时保存pth文件，目前只保留最好的
            # print("avg_psnr >= max_avg_psnr ? ", avg_psnr >= max_avg_psnr)
            if avg_psnr >= max_avg_psnr:
                max_avg_psnr = avg_psnr
                best_model_state_dict = model.state_dict()
                best_optimizer_state_dict = optimizer.state_dict()
                best_scheduler_state_dict = scheduler.state_dict()
                best_dis_state_dict = dis.state_dict()
                best_d_optim_state_dict = d_optim.state_dict()

                if epoch_num % save_per_epoch == 0:
                    checkpoint_data = {
                        'max_avg_psnr': max_avg_psnr,
                        'epoch': epoch_num,
                        'total_cost_time': total_cost_time,
                        'model_state_dict': best_model_state_dict,
                        'optimizer_state_dict': best_optimizer_state_dict,
                        'scheduler_state_dict': best_scheduler_state_dict,
                        'dis_state_dict': best_dis_state_dict,
                        'd_optim_state_dict': best_d_optim_state_dict
                    }
                    # 保存当前最优模型
                    torch.save(checkpoint_data, save_mode_path)
                    # 保存epoch模型
                    save_epoch_path = save_mode_path.replace(".pth", "_") + str(epoch_num) + '.pth'
                    torch.save(checkpoint_data, save_epoch_path)
                    logging.info(
                        "*** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ")
                    logging.info("save the best checkpoint_data to " + save_mode_path + '...')
                    logging.info("epoch: %d" % epoch_num)
                    logging.info("max_avg_psnr: %f" % max_avg_psnr)
            else:
                if epoch_num % save_per_epoch == 0:
                    checkpoint_data = {
                        'max_avg_psnr': max_avg_psnr,
                        'epoch': epoch_num,
                        'total_cost_time': total_cost_time,
                        'model_state_dict': best_model_state_dict,
                        'optimizer_state_dict': best_optimizer_state_dict,
                        'scheduler_state_dict': best_scheduler_state_dict,
                        'dis_state_dict': best_dis_state_dict,
                        'd_optim_state_dict': best_d_optim_state_dict
                    }
                    # 保存当前最优模型
                    torch.save(checkpoint_data, save_mode_path)
                    # 保存epoch模型
                    save_epoch_path = save_mode_path.replace(".pth", "_") + str(epoch_num) + '.pth'
                    torch.save(checkpoint_data, save_epoch_path)
                    logging.info("epoch: %d" % epoch_num)

        epoch_times = str(datetime.timedelta(seconds=epoch_times))
        # 写进文档
        logging.info('current one epoch training takes:  %s' % epoch_times)
        logging.info('total training so far takes:  %s' % str(datetime.timedelta(seconds=total_cost_time)))
        logging.info(
            "*** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ")
        logging.info(
            "                                                                                                ")
        logging.info(
            "                                                                                                ")
        logging.info(
            "                                                                                                ")
        logging.info(
            "                                                                                                ")

    writer.close()
    # total_cost_time += time.time() - total_start
    total_cost_time = str(datetime.timedelta(seconds=total_cost_time))
    logging.info('[END] The total training takes: %s' % total_cost_time)
    return "Training Finished!"
