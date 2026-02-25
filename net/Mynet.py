import torch
import torch.nn as nn
import torch.nn.functional as F
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.gb_cca import *
from huggingface_hub import PyTorchModelHubMixin
import math

class MCPF(nn.Module):
    def __init__(self, in_dim, reduction=1, use_se=True):
        super().__init__()
        mid_dim = in_dim // reduction

        self.q_proj = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, bias=False),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1, groups=mid_dim, bias=False)
        )
        self.k_proj = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, bias=False),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1, groups=mid_dim, bias=False)
        )
        self.v_proj = nn.Conv2d(in_dim, mid_dim, kernel_size=1, bias=False)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.out_proj = nn.Conv2d(mid_dim, in_dim, kernel_size=1, bias=False)
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_dim, in_dim // 8, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_dim // 8, in_dim, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, x1, x2):
        q = self.q_proj(x1)
        k = self.k_proj(x2)
        v = self.v_proj(x2)
        attn = torch.sigmoid(q + k)
        x = attn * v
        x = self.fuse_conv(x)
        out = self.out_proj(x)
        if self.use_se:
            scale = self.se(x1)
            out = out * scale
        return out + x1



def extract_HVI_complement_grouped(x, eps=1e-8, normalize=True):

    assert x.dim() == 4 and x.size(1) == 3, "x should be [B,3,H,W]"
    Bsz, _, H, W = x.shape
    R, G, B = x[:, 0:1], x[:, 1:2], x[:, 2:3]

    Y  = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y_lin = Y

    alpha = 20.0
    Y_log = torch.log1p(alpha * Y) / math.log1p(alpha)

    def box_blur(img, k=7):
        ker = torch.ones((1, 1, k, k), device=img.device, dtype=img.dtype) / (k * k)
        return F.conv2d(img, ker, padding=k // 2)
    Y_mean_local = box_blur(Y, k=7)
    Y_weber = (Y - Y_mean_local) / (Y_mean_local + eps)

    Y_lp = box_blur(Y, k=15)  # 近似光照
    Y_retinex = torch.log(Y + eps) - torch.log(Y_lp + eps)

    k_hvi = 0.2
    Y_sin_k = torch.sin(0.5 * math.pi * Y).clamp_min(0) ** k_hvi

    LSP = torch.cat([Y_lin, Y_log, Y_weber, Y_retinex, Y_sin_k], dim=1)


    Cb0, Cr0 = Cb - 0.5, Cr - 0.5
    std_cb = torch.sqrt((Cb0 ** 2).mean(dim=(2, 3), keepdim=True) + eps)
    std_cr = torch.sqrt((Cr0 ** 2).mean(dim=(2, 3), keepdim=True) + eps)
    Cb_n = Cb0 / std_cb
    Cr_n = Cr0 / std_cr

    rho = torch.sqrt(Cb_n ** 2 + Cr_n ** 2 + eps)

    sobel_x = torch.tensor([[[[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]]]], dtype=x.dtype, device=x.device)
    sobel_y = torch.tensor([[[[-1, -2, -1],
                              [ 0,  0,  0],
                              [ 1,  2,  1]]]], dtype=x.dtype, device=x.device)

    def grad_mag(ch):
        gx = F.conv2d(ch, sobel_x, padding=1)
        gy = F.conv2d(ch, sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + eps)

    E_cb = grad_mag(Cb_n)
    E_cr = grad_mag(Cr_n)
    E_chroma = 0.5 * (E_cb + E_cr)

    gate = Y_sin_k
    CSP = gate * (E_chroma + 0.5 * rho)

    if normalize:
        def mmn(t):
            b, c, h, w = t.shape
            tv = t.view(b, c, -1)
            t_min, _ = tv.min(dim=-1, keepdim=True)
            t_max, _ = tv.max(dim=-1, keepdim=True)
            tv = (tv - t_min) / (t_max - t_min + eps)
            return tv.view(b, c, h, w)
        LSP = mmn(LSP)
        CSP = mmn(CSP)

    return LSP, CSP

class GSEIRB(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, use_residual=True, res_scale=1):
        super().__init__()
        self.use_residual = use_residual
        self.res_scale = res_scale

        mid_channels = int(in_channels * expansion)
        if mid_channels % 2 != 0:
            mid_channels += 1

        self.pw1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.dw = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1,
                            groups=mid_channels, bias=False)
        self.act = nn.SiLU(inplace=True)

        self.pw2 = nn.Conv2d(mid_channels // 2, out_channels, kernel_size=1, bias=False)

        se_hidden = max(out_channels // 8, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, se_hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_hidden, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)


    def forward(self, x):
        identity = x

        x = self.pw1(x)
        x = self.dw(x)
        x = self.act(x)

        # GLU门控
        x_a, x_b = torch.chunk(x, 2, dim=1)
        x = x_a * torch.sigmoid(x_b)

        x = self.pw2(x)

        # SE重标定
        scale = self.se(x)
        x = x * scale

        # 残差/投影
        if self.use_residual:
            if self.proj is not None:
                identity = self.proj(identity)
            if identity.shape == x.shape:
                x = x + self.res_scale * identity

        return x

class BIPCENet(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
                 ):
        super(BIPCENet, self).__init__()

        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.HVE_block0_ab = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block1_ab = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )

        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.IE_block0_ab = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(5, ch1, 3, stride=1, padding=0, bias=False),
        )

        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)
        self.IE_block1_ab = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2_ab = NormDownsample(ch2, ch3, use_norm=norm)

        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
        )

        self.HV_1 = GB_CCA_HV(ch2, head2)
        self.HV_2 = GB_CCA_HV(ch3, head3)
        self.HV_3 = GB_CCA_HV(ch4, head4)
        self.HV_4 = GB_CCA_HV(ch4, head4)
        self.HV_5 = GB_CCA_HV(ch3, head3)
        self.HV_6 = GB_CCA_HV(ch2, head2)

        self.I_1 = GB_CCA_I(ch2, head2)
        self.I_2 = GB_CCA_I(ch3, head3)
        self.I_3 = GB_CCA_I(ch4, head4)
        self.I_4 = GB_CCA_I(ch4, head4)
        self.I_5 = GB_CCA_I(ch3, head3)
        self.I_6 = GB_CCA_I(ch2, head2)

        self.trans = RGB_HVI()
        self.fusion_module1 = MCPF(in_dim=36)
        self.fusion_module2 = MCP(in_dim=36)

        self.fem1 =   GSEIRB(ch1, ch2)
        self.fem1_y = GSEIRB(ch1, ch2)
        self.fem2 =   GSEIRB(ch2, ch3)
        self.fem2_y = GSEIRB(ch2, ch3)
        self.fem3 =   GSEIRB(ch3, ch4)
        self.fem3_y = GSEIRB(ch3, ch4)
        self.jump_nn1=nn.Conv2d(36,36,kernel_size=3,stride=1,padding=1)
        self.jump_nn2=nn.Conv2d(36,36,kernel_size=3,stride=1,padding=1)
    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        LSP, CSP = extract_HVI_complement_grouped(x)

        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        a_enc0 = self.IE_block0_ab(LSP)
        a_enc1 = self.IE_block1_ab(a_enc0)

        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        b_0 = self.HVE_block0_ab(CSPCSP)
        b_1 = self.HVE_block1_ab(b_0)

        # 先融合再编码
        i_enc1 = self.fusion_module1(i_enc1, a_enc1)
        hv_1   = self.fusion_module2(hv_1, b_1)

        # 跳连缓存
        i_jump0 = (i_enc0)
        hv_jump0 = (hv_0)


        # 亮度增强 + 交互
        a_up1 = self.fem1_y(a_enc1)
        b_up1 = self.fem1(b_1)

        i_enc2 = self.I_1(i_enc1, hv_1, a_up1)
        hv_2   = self.HV_1(hv_1, i_enc1, b_up1)

        v_jump1 = i_enc2
        hv_jump1 = hv_2

        # 下采样
        i_enc2 = self.IE_block2(i_enc2)
        hv_2   = self.HVE_block2(hv_2)

        a_up2 = self.fem2_y(a_up1)
        b_up2 = self.fem2(b_up1)
        a_up2 = F.interpolate(a_up2, scale_factor=0.5, mode='bilinear', align_corners=False)
        b_up2 = F.interpolate(b_up2, scale_factor=0.5, mode='bilinear', align_corners=False)

        i_enc3 = self.I_2(i_enc2, hv_2, a_up2)
        hv_3   = self.HV_2(hv_2, i_enc2, b_up2)

        v_jump2 = i_enc3
        hv_jump2 = hv_3

        # 继续下采样
        i_enc3 = self.IE_block3(i_enc3)
        hv_3   = self.HVE_block3(hv_3)

        a_up3 = self.fem3_y(a_up2)
        b_up3 = self.fem3(b_up2)
        a_up3= F.interpolate(a_up3, scale_factor=0.5, mode='bilinear', align_corners=False)
        b_up3 = F.interpolate(b_up3, scale_factor=0.5, mode='bilinear', align_corners=False)

        i_enc4 = self.I_3(i_enc3, hv_3, a_up3)
        hv_4   = self.HV_3(hv_3, i_enc3, b_up3)

        # 瓶颈交互
        i_dec4 = self.I_4(i_enc4, hv_4, a_up3)
        hv_4   = self.HV_4(hv_4, i_enc4, b_up3)

        # 上采样与跳连
        hv_3   = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)

        i_dec2 = self.I_5(i_dec3, hv_3, a_up2)
        hv_2   = self.HV_5(hv_3, i_dec3, b_up2)

        hv_2   = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec2, v_jump1)

        i_dec1 = self.I_6(i_dec2, hv_2, a_up1)
        hv_1   = self.HV6(hv_2, i_dec2, b_up1)

        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)

        hv_1   = self.HVD_block1(hv_1, hv_jump0)
        hv_0   = self.HVD_block0(hv_1)

        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi

        output_rgb = self.trans.PHVIT(output_hvi)
        return output_rgb

    def HVIT(self, x):
        hvi = self.trans.HVIT(x)
        return hvi
