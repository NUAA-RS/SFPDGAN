# 训练代码 train.py
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
import os
from PIL import Image
import numpy as np
import visdom
from generator_gam_deep import EnhancedGenerator, DeepUnfoldedGammaLayer
import torch.distributions as dist
from torch.linalg import eigvalsh
import torch.nn.functional as F

# Visdom初始化
viz = visdom.Visdom()
assert viz.check_connection(), "请先启动Visdom服务器：python -m visdom.server"

# 超参数
batch_size = 4
lr = 0.0001
epochs = 101
lambda_l1 = 100
image_size = 256
channels = 3


class GammaDistributionLoss(nn.Module):
    def __init__(self, lambda_gamma=10.0, eps=1e-6):
        super().__init__()
        self.lambda_gamma = lambda_gamma
        self.eps = eps

    def estimate_gamma_params(self, x):
        """可导的Gamma参数估计"""
        mu = x.mean()
        var = x.var(unbiased=False)  # 使用有偏估计保持稳定性

        k = (mu ** 2) / (var + self.eps)
        theta = var / (mu + self.eps)
        return k, theta

    def forward(self, real_VV, real_VH, fake_VV, fake_VH):
        # 计算差值绝对值
        real_diff = torch.abs(real_VV - real_VH).mean(dim=3)  # 转为单通道
        fake_diff = torch.abs(fake_VV - fake_VH).mean(dim=3)

        # 估计Gamma参数
        k_real, theta_real = self.estimate_gamma_params(real_diff)
        k_fake, theta_fake = self.estimate_gamma_params(fake_diff)

        # 构建分布
        real_gamma = dist.Gamma(k_real, 1 / (theta_real + self.eps))  # PyTorch使用rate参数
        fake_gamma = dist.Gamma(k_fake, 1 / (theta_fake + self.eps))

        # 计算KL散度
        kl_loss = dist.kl_divergence(fake_gamma, real_gamma).mean()

        return self.lambda_gamma * kl_loss
# class GammaLoss(nn.Module):
#     """深度展开的Gamma约束损失"""
#
#     def __init__(self, mode='kl'):
#         super().__init__()
#         self.gamma_layer = DeepUnfoldedGammaLayer()
#         self.mode = mode
#
#         if mode == 'l1':
#             self.loss = nn.L1Loss()
#         elif mode == 'kl':
#             # KL散度计算函数
#             self.kl_loss = lambda k_f, t_f, k_r, t_r: \
#                 (k_f - k_r) * torch.digamma(k_f + 1e-6) + \
#                 (torch.lgamma(k_r + 1e-6) - torch.lgamma(k_f + 1e-6)) + \
#                 k_r * (torch.log(t_f + 1e-6) - torch.log(t_r + 1e-6)) + \
#                 k_f * ((t_r + 1e-6) / (t_f + 1e-6) - 1)
#         else:
#             raise ValueError("Unsupported loss mode")
#
#     def forward(self, fake, real):
#         # 提取参数
#         fake_params = self.gamma_layer(fake.mean(1, keepdim=True))
#         real_params = self.gamma_layer(real.mean(1, keepdim=True))
#
#         k_fake, theta_fake = fake_params.chunk(2, dim=1)
#         k_real, theta_real = real_params.chunk(2, dim=1)
#
#         # 计算损失
#         if self.mode == 'l1':
#             return 0.5 * (self.loss(k_fake, k_real) + self.loss(theta_fake, theta_real))
#         elif self.mode == 'kl':
#             kl = self.kl_loss(k_fake, theta_fake, k_real.detach(), theta_real.detach())
#             return torch.mean(kl)

class PairwiseDifferenceLoss(nn.Module):
    """差异一致性损失函数"""
    def __init__(self, mode='l1'):
        super().__init__()
        if mode == 'l1':
            self.loss = nn.L1Loss()
        elif mode == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise ValueError("Unsupported loss mode")

    def forward(self, real_VV, real_VH, fake_VV, fake_VH):
        """
        计算真实图像对与生成图像对的差异一致性
        参数：
            real_VV: 真实VV彩色标签 [B,C,H,W]
            real_VH: 真实VH彩色标签 [B,C,H,W]
            fake_VV: 生成VV彩色图像 [B,C,H,W]
            fake_VH: 生成VH彩色图像 [B,C,H,W]
        返回：
            差异一致性损失值
        """
        # 计算真实图像对的差异
        real_diff = real_VV - real_VH
        # 计算生成图像对的差异
        fake_diff = fake_VV - fake_VH
        # 计算差异一致性损失
        return self.loss(fake_diff, real_diff.detach())

# 自定义数据集
class DualSARDataset(Dataset):
    def __init__(self, base_dir):
        self.vh_dir = os.path.join(base_dir, 'VH')
        self.vv_dir = os.path.join(base_dir, 'VV')
        self.vh_color_dir = os.path.join(base_dir, 'VH_color')
        self.vv_color_dir = os.path.join(base_dir, 'VV_color')

        self.file_list = [f for f in os.listdir(self.vh_dir) if f.endswith('.png')]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 单通道归一化
        ])

        self.color_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 三通道归一化
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]

        vh = Image.open(os.path.join(self.vh_dir, filename)).convert('L')
        vv = Image.open(os.path.join(self.vv_dir, filename)).convert('L')
        vh_color = Image.open(os.path.join(self.vh_color_dir, filename)).convert('RGB')
        vv_color = Image.open(os.path.join(self.vv_color_dir, filename)).convert('RGB')

        return {
            'VH': self.transform(vh),
            'VV': self.transform(vv),
            'VH_color': self.color_transform(vh_color),
            'VV_color': self.color_transform(vv_color)
        }


# 生成器（U-Net）
class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(Generator, self).__init__()

        # self.scattering_extractor = ScatteringFeatureExtractor(in_channels)
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.down = nn.Sequential(
            *block(1, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.ReLU()
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.down(x)
        return self.up(x)


# 判别器
class Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 3, 1, 1)
        )

    def forward(self, x):
        return self.model(x)


# 初始化模型
G1 = EnhancedGenerator()
G2 = EnhancedGenerator()
D1 = Discriminator()
D2 = Discriminator()

optimizer_G1 = optim.Adam(G1.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D1 = optim.Adam(D1.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_G2 = optim.Adam(G2.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D2 = optim.Adam(D2.parameters(), lr=lr, betas=(0.5, 0.999))
# 损失函数
adversarial_loss = nn.MSELoss()
l1_loss = nn.L1Loss()
lambda_gamma = 10  # Gamma损失权重
gamma_loss = GammaDistributionLoss(lambda_gamma=10)

# lambda_gamma = 0.01
lambda_diff = 50  # 差异损失权重
pairwise_diff_loss = PairwiseDifferenceLoss(mode='l1')
# 优化器
# optimizer_G = optim.Adam(
#     list(G1.parameters()) + list(G2.parameters()),
#     lr=lr, betas=(0.5, 0.999)
# )
# optimizer_D = optim.Adam(
#     list(D1.parameters()) + list(D2.parameters()),
#     lr=lr, betas=(0.5, 0.999)
# )

# 数据集
dataset = DualSARDataset('dataset_color/train')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
loss_history1 = {'G1': [], 'D1': []}
loss_history2 = {'G2': [], 'D2': []}

# 训练
for epoch in range(epochs):
    epoch_g1_loss = 0.0
    epoch_d1_loss = 0.0
    epoch_g2_loss = 0.0
    epoch_d2_loss = 0.0
    total_batches = len(dataloader)

    for i, batch in enumerate(dataloader):

        real_VH = batch['VH']
        real_VV = batch['VV']
        real_VH_color = batch['VH_color']
        real_VV_color = batch['VV_color']

        valid = torch.ones((real_VH.size(0), 1, 16, 16)).requires_grad_(False)
        fake = torch.zeros((real_VH.size(0), 1, 16, 16)).requires_grad_(False)

        # 训练生成器
        optimizer_G1.zero_grad()

        fake_VH_color = G1(real_VH)
        fake_VV_color = G2(real_VV)

        loss_G1_adv = adversarial_loss(D1(torch.cat([real_VH, fake_VH_color], 1)), valid)
        loss_G1_l1 = l1_loss(fake_VH_color, real_VH_color) * lambda_l1
        total_loss_G1 = (
                loss_G1_adv  +  # 对抗损失
                loss_G1_l1 +
                # gamma_loss(real_VV_color, real_VH_color,  # Gamma分布损失
                #            fake_VV_color, fake_VH_color) +
                # 像素级L1损失
                # loss_G1_gamma + loss_G2_gamma + # 散射特征损失
                lambda_diff * pairwise_diff_loss(  # 新增差异一致性损失
            real_VV_color, real_VH_color,
            fake_VV_color, fake_VH_color
        )
        )
        total_loss_G1.backward()
        optimizer_G1.step()

        optimizer_D1.zero_grad()
        # 真实样本损失
        loss_real_D1 = adversarial_loss(D1(torch.cat([real_VH, real_VH_color], 1)), valid)
        # 生成样本损失
        loss_fake_D1 = adversarial_loss(D1(torch.cat([real_VH, fake_VH_color.detach()], 1)), fake)
        total_loss_D1 = (loss_real_D1 + loss_fake_D1 )
        total_loss_D1.backward()
        optimizer_D1.step()



        optimizer_G2.zero_grad()

        fake_VH_color = G1(real_VH)
        fake_VV_color = G2(real_VV)
        # 对抗损失
        loss_G2_adv = adversarial_loss(D2(torch.cat([real_VV, fake_VV_color], 1)), valid)
        # L1损失
        loss_G2_l1 = l1_loss(fake_VV_color, real_VV_color) * lambda_l1

        # lambda_gamma = min(epoch / 10.0, 1.0) * 0.1
        # loss_G1_gamma = gamma_loss(fake_VH_color, real_VH_color)  * lambda_gamma
        # loss_G2_gamma = gamma_loss(fake_VV_color, real_VV_color)  * lambda_gamma

        # gamma损失
        # loss_G1_gamma = l1_loss(fake_VH_color, real_VH_color) * lambda_gamma
        # loss_G2_gamma = l1_loss(fake_VV_color, real_VV_color) * lambda_gamma
        total_loss_G2 = (
                loss_G2_adv +  # 对抗损失
                loss_G2_l1 +
                # gamma_loss(real_VV_color, real_VH_color,  # Gamma分布损失
                #            fake_VV_color, fake_VH_color) +
                # # 像素级L1损失
                # loss_G1_gamma + loss_G2_gamma + # 散射特征损失
                lambda_diff * pairwise_diff_loss(  # 新增差异一致性损失
            real_VV_color, real_VH_color,
            fake_VV_color, fake_VH_color
        )
        )
        total_loss_G2.backward()
        optimizer_G2.step()

        # 训练判别器
        optimizer_D2.zero_grad()

        # 真实样本损失
        loss_real_D2 = adversarial_loss(D2(torch.cat([real_VV, real_VV_color], 1)), valid)

        # 生成样本损失
        loss_fake_D2 = adversarial_loss(D2(torch.cat([real_VV, fake_VV_color.detach()], 1)), fake)

        total_loss_D2 = ( loss_real_D2 + loss_fake_D2)
        total_loss_D2.backward()
        optimizer_D2.step()





        # 可视化
        batches_done = epoch * len(dataloader) + i
        if batches_done % 10 == 0:
            viz.line(
                X=[batches_done],
                Y=[[total_loss_G1.item(), total_loss_D1.item()]],
                win='loss',
                update='append',
                opts=dict(title='训练损失', legend=['生成器1损失', '判别器1损失'])
            )
            viz.line(
                X=[batches_done],
                Y=[[total_loss_G2.item(), total_loss_D2.item()]],
                win='loss',
                update='append',
                opts=dict(title='训练损失', legend=['生成器2损失', '判别器2损失'])
            )

            # 显示生成图像
            viz.images(fake_VH_color.data[:4] * 0.5 + 0.5, nrow=4, win='VH生成',
                       opts=dict(title='生成VH彩色图像'))
            viz.images(real_VH_color.data[:4] * 0.5 + 0.5, nrow=4, win='VH真实',
                       opts=dict(title='真实VH彩色图像'))
            viz.images(fake_VV_color.data[:4] * 0.5 + 0.5, nrow=4, win='VV生成',
                       opts=dict(title='生成VV彩色图像'))
            viz.images(real_VV_color.data[:4] * 0.5 + 0.5, nrow=4, win='VV真实',
                       opts=dict(title='真实VV彩色图像'))

        epoch_g1_loss += total_loss_G1.item()
        epoch_g2_loss += total_loss_G2.item()
        epoch_d1_loss += total_loss_D1.item()
        epoch_d2_loss += total_loss_D2.item()
        # 在训练循环的visualization部分添加


        # if batches_done % 100 == 0:
        #     # 计算差异图
        #     real_diff_map = (real_VV_color - real_VH_color).abs().mean(dim=1, keepdim=True)
        #     fake_diff_map = (fake_VV_color - fake_VH_color).abs().mean(dim=1, keepdim=True)
        #
        #     # 可视化对比
        #     viz.images(torch.cat([real_diff_map[:4], fake_diff_map[:4]]) * 0.5 + 0.5,
        #                nrow=4, win='diff_comparison',
        #                opts=dict(title='真实差异 vs 生成差异'))

        # 实时控制台打印（每50个batch打印一次）
        if i % 50 == 0:
            current_loss_g1 = total_loss_G1.item()
            current_loss_d1 = total_loss_D1.item()
            current_loss_g2 = total_loss_G2.item()
            current_loss_d2 = total_loss_D2.item()
            print(f"[Epoch {epoch + 1}/{epochs}] Batch {i}/{total_batches} | "
                  f"G1 Loss: {current_loss_g1:.4f} | D1 Loss: {current_loss_d1:.4f}")
            print(f"[Epoch {epoch + 1}/{epochs}] Batch {i}/{total_batches} | "
                  f"G2 Loss: {current_loss_g2:.4f} | D2 Loss: {current_loss_d2:.4f}")
        # Visdom动态更新（每10个batch更新一次）
        # if batches_done % 10 == 0:
        #     # 可视化Gamma参数分布
        #     with torch.no_grad():
        #         real_params = G1.gamma_layer(real_VH_color.mean(1, keepdim=True))
        #         fake_params = G1.gamma_layer(fake_VH_color.mean(1, keepdim=True))
        #
        #     viz.line(
        #         X=[batches_done],
        #         Y=[[real_params[:, 0].mean(), fake_params[:, 0].mean()]],
        #         win = 'gamma_k',
        #         update = 'append',
        #         opts = dict(title='形状参数k对比', legend=['真实k', '生成k'])
        #     )
        #
        #     viz.line(
        #         X=[batches_done],
        #         Y=[[real_params[:, 1].mean(), fake_params[:, 1].mean()]],
        #         win = 'gamma_theta',
        #         update = 'append',
        #         opts = dict(title='尺度参数θ对比', legend=['真实θ', '生成θ'])
        #     )
        if batches_done % 10 == 0:
            # 更新损失曲线
            viz.line(
                X=[batches_done],
                Y=[[total_loss_G1.item(), total_loss_D1.item()]],
                win='loss',
                update='append',
                opts=dict(
                    title=f'训练损失 (Epoch {epoch + 1}/{epochs})',
                    legend=['生成器1损失', '判别器1损失'],
                    showlegend=True
                )
            )
            viz.line(
                X=[batches_done],
                Y=[[total_loss_G2.item(), total_loss_D2.item()]],
                win='loss',
                update='append',
                opts=dict(
                    title=f'训练损失 (Epoch {epoch + 1}/{epochs})',
                    legend=['生成器2损失', '判别器2损失'],
                    showlegend=True
                )
            )

            # 创建文本信息窗口
            text_content = f"""
                    <h3>训练进度</h3>
                    <b>当前Epoch:</b> {epoch + 1}/{epochs}<br>
                    <b>当前Batch:</b> {i}/{total_batches}<br>
                    <b>生成器1损失:</b> {current_loss_g1:.4f}<br>
                    <b>判别器1损失:</b> {current_loss_d1:.4f}<br>
                    <b>生成器2损失:</b> {current_loss_g2:.4f}<br>
                    <b>判别器2损失:</b> {current_loss_d2:.4f}
                    """
            viz.text(text_content, win='progress', opts=dict(title='训练状态'))

        batches_done += 1

        # 计算epoch平均损失
    epoch_g1_loss /= total_batches
    epoch_d1_loss /= total_batches
    epoch_g2_loss /= total_batches
    epoch_d2_loss /= total_batches
    loss_history1['G1'].append(epoch_g1_loss)
    loss_history1['D1'].append(epoch_d1_loss)
    loss_history2['G2'].append(epoch_g2_loss)
    loss_history2['D2'].append(epoch_d2_loss)
    # 打印epoch摘要
    print(f"\nEpoch {epoch + 1} 完成 | "
          f"平均生成器1损失: {epoch_g1_loss:.4f} | "
          f"平均判别器1损失: {epoch_d1_loss:.4f} | "
          f"平均生成器2损失: {epoch_g2_loss:.4f} | "
          f"平均判别器2损失: {epoch_d2_loss:.4f}\n"
          "-----------------------------------------")

    # 更新Visdom的epoch级损失曲线
    viz.line(
        X=[epoch + 1],
        Y=[[epoch_g1_loss, epoch_d1_loss]],
        win='epoch_loss',
        update='append' if epoch > 0 else None,
        opts=dict(
            title='Epoch级损失趋势',
            legend=['生成器1', '判别器1'],
            xlabel='Epoch',
            ylabel='损失值'
        )
    )
    viz.line(
        X=[epoch + 1],
        Y=[[epoch_g2_loss, epoch_d2_loss]],
        win='epoch_loss',
        update='append' if epoch > 0 else None,
        opts=dict(
            title='Epoch级损失趋势',
            legend=['生成器2', '判别器2'],
            xlabel='Epoch',
            ylabel='损失值'
        )
    )

    # viz.line(
    #     X=[batches_done],
    #     Y=[[pairwise_diff_loss.item()]],
    #     win='diff_loss',
    #     update='append',
    #     opts=dict(title='差异一致性损失', legend=['差异损失'])
    # )

    # 保存模型
    if epoch % 10 == 0:
        torch.save(G1.state_dict(), f'checkpoints/G1_epoch{epoch}.pth')
        torch.save(G2.state_dict(), f'checkpoints/G2_epoch{epoch}.pth')

viz.line(
    X=list(range(1, epochs+1)),
    Y=[loss_history1['G1'], loss_history1['D1']],
    win='final_loss',
    opts=dict(
        title='训练损失总结',
        legend=['生成器1', '判别器1'],
        xlabel='Epoch',
        ylabel='损失值',
        showlegend=True
    )
)

viz.line(
    X=list(range(1, epochs+1)),
    Y=[loss_history2['G1'], loss_history2['D1']],
    win='final_loss',
    opts=dict(
        title='训练损失总结',
        legend=['生成器2', '判别器2'],
        xlabel='Epoch',
        ylabel='损失值',
        showlegend=True
    )
)



