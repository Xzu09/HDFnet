import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

# ==============================================================================
# 0. 尝试导入 mamba_ssm (兼容 BasicSR 环境，防报错)
# ==============================================================================
# ================= 替换原有的 try-except =================
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    # 直接抛出强硬的错误，终止进程
    raise ImportError(
        "\n[Fatal Error] mamba_ssm is not installed! \n"
        "The Mamba block requires selective_scan_fn to function properly. \n"
        "Please install it via: pip install mamba-ssm \n"
        "We refuse to fall back to a silent zero-tensor bypass."
    )


# ==============================================================================
# 1. 基础组件与无损入口/上下采样模块
# ==============================================================================
class ChannelLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class SimpleResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.act = nn.Mish(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.skip(x) + self.conv2(self.act(self.conv1(x)))


def dwt_split(x):
    x1, x2 = x[:, :, 0::2, 0::2], x[:, :, 1::2, 0::2]
    x3, x4 = x[:, :, 0::2, 1::2], x[:, :, 1::2, 1::2]
    x_LL = (x1 + x2 + x3 + x4) / 2.0
    x_HL = (-x1 - x2 + x3 + x4) / 2.0
    x_LH = (-x1 + x2 - x3 + x4) / 2.0
    x_HH = (x1 - x2 - x3 + x4) / 2.0
    return x_LL, x_HL, x_LH, x_HH


def iwt(x_LL, x_High):
    B, C, H, W = x_LL.shape
    x_HL, x_LH, x_HH = torch.chunk(x_High, 3, dim=1)
    x1 = (x_LL - x_HL - x_LH + x_HH) / 2.0
    x2 = (x_LL - x_HL + x_LH - x_HH) / 2.0
    x3 = (x_LL + x_HL - x_LH - x_HH) / 2.0
    x4 = (x_LL + x_HL + x_LH + x_HH) / 2.0
    h = torch.empty([B, C, H * 2, W * 2], device=x_LL.device, dtype=x_LL.dtype)
    h[:, :, 0::2, 0::2] = x1
    h[:, :, 1::2, 0::2] = x2
    h[:, :, 0::2, 1::2] = x3
    h[:, :, 1::2, 1::2] = x4
    return h


class PixelUnshuffleDown(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.conv = nn.Conv2d(in_feat, out_feat // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.unshuffle = nn.PixelUnshuffle(2)

    def forward(self, x): return self.unshuffle(self.conv(x))


class PixelShuffleUp(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.conv = nn.Conv2d(in_feat, out_feat * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x): return self.shuffle(self.conv(x))

# ==============================================================================
# 2. P (Physics): 物理参数提取器与物理引导调制器
# ==============================================================================
class SpatialPhysicsEstimator(nn.Module):
    def __init__(self, in_ch=3, phys_dim=64):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=4, stride=4, padding=0),
            nn.Mish(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=4, padding=0),
            nn.Mish(inplace=True),
            nn.Conv2d(32, phys_dim, kernel_size=3, stride=2, padding=1),
            nn.Mish(inplace=True)
        )
        self.phys_proj = nn.Sequential(
            nn.Conv2d(phys_dim, phys_dim, kernel_size=1),
            nn.Mish(inplace=True),
            nn.Conv2d(phys_dim, phys_dim, kernel_size=1)
        )

    def forward(self, x):
        feat = self.extractor(x)
        return self.phys_proj(feat)

#
#class PhysicsGuidedModulator(nn.Module):
#    def __init__(self, feat_dim, phys_dim=64):
#        super().__init__()
#        self.norm = nn.InstanceNorm2d(feat_dim, affine=False)
#        self.conv_gamma = nn.Conv2d(phys_dim, feat_dim, 1)
#        self.conv_beta = nn.Conv2d(phys_dim, feat_dim, 1)
#
#    def forward(self, x_LL, phys_map):
#        B, C, H, W = x_LL.shape
#        if phys_map.shape[-2:] != (H, W):
#            phys_map = F.interpolate(phys_map, size=(H, W), mode='bilinear', align_corners=False)
#
#        gamma = self.conv_gamma(phys_map)
#        gamma = torch.sigmoid(gamma) * 2.0
#        beta = torch.tanh(self.conv_beta(phys_map)) * 1.5
#        return self.norm(x_LL) * gamma + beta
class PhysicsGuidedModulator(nn.Module):
    def __init__(self, feat_dim, phys_dim=64):
        super().__init__()
        # 删除 self.norm
        self.conv_gamma = nn.Conv2d(phys_dim, feat_dim, 1)
        self.conv_beta = nn.Conv2d(phys_dim, feat_dim, 1)

    def forward(self, x_LL, phys_map):
        B, C, H, W = x_LL.shape
        if phys_map.shape[-2:] != (H, W):
            phys_map = F.interpolate(phys_map, size=(H, W), mode='bilinear', align_corners=False)

        # 推荐使用 1 + gamma 的形式，使得网络更容易学到恒等映射 (Identity)
        gamma = self.conv_gamma(phys_map)
        beta = self.conv_beta(phys_map)
        return x_LL * (1.0 + gamma) + beta

# ==============================================================================
# 3. M (Mamba) 核心组件
# ==============================================================================
class SS2D_ExpandAxialGated(nn.Module):
    def __init__(self, d_model, dir_V='fwd', dir_H='fwd', d_state=16, d_conv=3, expand=2.0, dt_rank="auto"):
        super().__init__()
        self.d_model, self.dir_V, self.dir_H = d_model, dir_V, dir_H
        self.d_inner = int(expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.split_dim = self.d_inner // 2
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, groups=self.d_inner, bias=True, kernel_size=d_conv,
                                padding=(d_conv - 1) // 2)
        self.act = nn.Mish(inplace=True)
        self.x_proj_H = nn.Parameter(torch.empty(1, self.split_dim, self.dt_rank + d_state * 2))
        self.dt_projs_H = nn.Parameter(torch.empty(1, self.split_dim, self.dt_rank))
        self.dt_bias_H = nn.Parameter(torch.empty(1, self.split_dim))
        self.Ds_H = nn.Parameter(torch.ones(1 * self.split_dim))
        self.A_logs_H = nn.Parameter(
            repeat(torch.log(repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d=self.split_dim)),
                   "d n -> r d n", r=1).flatten(0, 1))
        self.x_proj_V = nn.Parameter(torch.empty(1, self.split_dim, self.dt_rank + d_state * 2))
        self.dt_projs_V = nn.Parameter(torch.empty(1, self.split_dim, self.dt_rank))
        self.dt_bias_V = nn.Parameter(torch.empty(1, self.split_dim))
        self.Ds_V = nn.Parameter(torch.ones(1 * self.split_dim))
        self.A_logs_V = nn.Parameter(
            repeat(torch.log(repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d=self.split_dim)),
                   "d n -> r d n", r=1).flatten(0, 1))
        for param in [self.x_proj_H, self.x_proj_V]: nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        for param in [self.dt_projs_H, self.dt_projs_V]: nn.init.uniform_(param, -0.01, 0.01)
        for param in [self.dt_bias_H, self.dt_bias_V]: nn.init.constant_(param, 0.1)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def _axial_scan(self, x, x_proj_w, dt_projs_w, dt_bias, Ds, A_logs, seq_len, batch_size, direction):
        B_seq, C, L = batch_size, self.split_dim, seq_len
        if direction == 'bwd': x = torch.flip(x, dims=[-1])
        x_dbl = torch.einsum("b k d l, k d c -> b k c l", x.view(B_seq, 1, -1, L), x_proj_w)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, 16, 16], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B_seq, 1, -1, L), dt_projs_w)
        xs_in, dts_in = x.float().view(B_seq, -1, L), dts.contiguous().float().view(B_seq, -1, L)
        Bs_in, Cs_in = Bs.float().view(B_seq, 1, -1, L), Cs.float().view(B_seq, 1, -1, L)
        Ds_val, As_val = Ds.float().view(-1), -torch.exp(A_logs.float()).view(-1, 16)

        # 移除 if selective_scan_fn is not None 的判断，直接执行算子
        out_y = selective_scan_fn(
            xs_in, dts_in, As_val, Bs_in, Cs_in, Ds_val, z=None,
            delta_bias=dt_bias.float().view(-1), delta_softplus=True,
            return_last_state=False
        ).view(B_seq, 1, -1, L)

        out_y = out_y[:, 0]
        if direction == 'bwd': out_y = torch.flip(out_y, dims=[-1])
        return out_y

    def forward(self, x: torch.Tensor):
        x_channel_last = x.permute(0, 2, 3, 1).contiguous()
        x_proj, z = self.in_proj(x_channel_last).chunk(2, dim=-1)
        x_proj = self.act(self.conv2d(x_proj.permute(0, 3, 1, 2).contiguous()))
        B, _, H, W = x_proj.shape
        x_H, x_V = torch.chunk(x_proj, 2, dim=1)
        xs_H = x_H.permute(0, 2, 1, 3).contiguous().view(B * H, self.split_dim, W)
        out_H_1D = self._axial_scan(xs_H, self.x_proj_H, self.dt_projs_H, self.dt_bias_H, self.Ds_H, self.A_logs_H,
                                    seq_len=W, batch_size=B * H, direction=self.dir_H)
        out_H = out_H_1D.view(B, H, self.split_dim, W).permute(0, 2, 1, 3).contiguous()
        xs_V = x_V.permute(0, 3, 1, 2).contiguous().view(B * W, self.split_dim, H)
        out_V_1D = self._axial_scan(xs_V, self.x_proj_V, self.dt_projs_V, self.dt_bias_V, self.Ds_V, self.A_logs_V,
                                    seq_len=H, batch_size=B * W, direction=self.dir_V)
        out_V = out_V_1D.view(B, W, self.split_dim, H).permute(0, 2, 3, 1).contiguous()
        gated_H, gated_V = out_H * F.sigmoid(out_V), out_V * F.sigmoid(out_H)
        out_fused = torch.cat([gated_H, gated_V], dim=1)
        out = self.out_norm(out_fused.permute(0, 2, 3, 1).contiguous()) * F.silu(z)
        return self.out_proj(out).permute(0, 3, 1, 2).contiguous()


class GatedFFN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False)
        self.pointwise_conv = nn.Conv2d(hidden_dim // 2, hidden_dim, 1, bias=False)

    def forward(self, x):
        z1, z2 = self.depthwise_conv(x).chunk(2, dim=1)
        return self.pointwise_conv(F.mish(z1) * z2) + x


class HierarchicalLFSSM(nn.Module):
    def __init__(self, dim, dir_V='fwd', dir_H='fwd', region_size=4):
        super().__init__()
        self.region_size = region_size
        self.ln_1 = ChannelLayerNorm(dim)
        self.local_ssm = SS2D_ExpandAxialGated(d_model=dim, dir_V=dir_V, dir_H=dir_H)
        self.region_pool = nn.AvgPool2d(region_size, stride=region_size)
        self.region_ssm = SS2D_ExpandAxialGated(d_model=dim, dir_V=dir_V, dir_H=dir_H)
        self.spatial_gate = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.Sigmoid())
        self.skip_scale = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.ln_2 = ChannelLayerNorm(dim)
        self.conv_blk = GatedFFN(dim)
        self.skip_scale2 = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        normed_x = self.ln_1(x)
        feat_local = self.local_ssm(normed_x)
        pad_h = (self.region_size - H % self.region_size) % self.region_size
        pad_w = (self.region_size - W % self.region_size) % self.region_size
        normed_x_padded = F.pad(normed_x, (0, pad_w, 0, pad_h), mode='reflect') if pad_h > 0 or pad_w > 0 else normed_x
        region_x = self.region_pool(normed_x_padded)
        feat_region = self.region_ssm(region_x)
        feat_region_up = F.interpolate(feat_region, size=(H + pad_h, W + pad_w), mode='bilinear', align_corners=False)[
                         :, :, :H, :W]
        gate = self.spatial_gate(torch.cat([feat_local, feat_region_up], dim=1))
        feat_fused = gate * feat_local + (1 - gate) * feat_region_up
        x = x * self.skip_scale + feat_fused
        return x * self.skip_scale2 + self.conv_blk(self.ln_2(x))


# ==============================================================================
# 4. 小波频率引导与重构组件 (包含消融目标：动态引导模块)
# ==============================================================================
class SpatialGuidedLowFreqModulator(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(feat_dim, affine=False)
        self.spatial_context = nn.Sequential(nn.Conv2d(feat_dim, feat_dim // 2, 3, padding=1), nn.Mish(inplace=True),
                                             nn.Conv2d(feat_dim // 2, feat_dim * 2, 3, padding=1))
        self.sft_fuse = nn.Conv2d(feat_dim * 2, feat_dim * 2, 1)
        nn.init.constant_(self.sft_fuse.weight, 0)
        nn.init.constant_(self.sft_fuse.bias, 0)

    def forward(self, x_LL):
        gamma, beta = self.sft_fuse(self.spatial_context(x_LL)).chunk(2, dim=1)
        return x_LL + self.norm(x_LL) * gamma + beta


class DynamicDirectionalGuide(nn.Module):
    """
    Final Version: 方向性高频物理引导模块
    通过低频特征提取垂直、水平和对角线维度的物理梯度，以此门控高频噪声。
    """

    def __init__(self, dim=64):  # 移除了 guide_mode 参数
        super().__init__()
        # 固定使用 Directional 逻辑
        self.grad_x = nn.Conv2d(dim, dim, kernel_size=(1, 5), padding=(0, 2), groups=dim, bias=False)
        self.grad_y = nn.Conv2d(dim, dim, kernel_size=(5, 1), padding=(2, 0), groups=dim, bias=False)
        self.grad_xy = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.noise_gate = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3),
            nn.Mish(inplace=True),
            nn.Conv2d(dim * 3, dim * 3, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_ll, hf_hl, hf_lh, hf_hh):
        # 直接执行物理门控逻辑
        grad_x = self.grad_x(feat_ll)
        grad_y = self.grad_y(feat_ll)
        grad_xy = self.grad_xy(feat_ll)

        merged_grads = torch.cat([grad_x, grad_y, grad_xy], dim=1)
        gate_x, gate_y, gate_xy = torch.chunk(self.noise_gate(merged_grads), 3, dim=1)

        out_hl = hf_hl + grad_x * gate_x
        out_lh = hf_lh + grad_y * gate_y
        out_hh = hf_hh + grad_xy * gate_xy
        return out_hl, out_lh, out_hh


class DirectionalRefineBlock(nn.Module):
    def __init__(self, dim, direction='vertical'):
        super().__init__()
        if direction == 'vertical':
            k1, p1, k2, p2 = (3, 1), (1, 0), (5, 1), (2, 0)
        elif direction == 'horizontal':
            k1, p1, k2, p2 = (1, 3), (0, 1), (1, 5), (0, 2)
        else:
            k1, p1, k2, p2 = (3, 3), (1, 1), (5, 5), (2, 2)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=k1, padding=p1, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=k2, padding=p2, groups=dim)
        self.fuse = nn.Conv2d(dim * 2, dim, 1)
        self.act = nn.Mish(inplace=True)

    def forward(self, x):
        return x + self.fuse(torch.cat([self.act(self.conv1(x)), self.act(self.conv2(x))], dim=1))


class MultiScaleContextBlock(nn.Module):
    def __init__(self, dim3):
        super().__init__()
        self.conv0 = nn.Identity()
        self.conv1 = nn.Sequential(nn.Conv2d(dim3, dim3, 3, 1, 1, groups=dim3), nn.Conv2d(dim3, dim3, 1))
        self.conv2 = nn.Sequential(nn.Conv2d(dim3, dim3, 5, 1, 2, groups=dim3), nn.Conv2d(dim3, dim3, 1))
        self.conv3 = nn.Sequential(nn.Conv2d(dim3, dim3, 7, 1, 3, groups=dim3), nn.Conv2d(dim3, dim3, 1))
        self.fuse = nn.Conv2d(dim3 * 4, dim3, 1)

    def forward(self, x): return self.fuse(
        torch.cat([self.conv0(x), self.conv1(x), self.conv2(x), self.conv3(x)], dim=1))


class PhysicallySoundFrequencyRefine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.amp_modulator = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.Mish(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        )
        self.phase_adaptor = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.spatial_gate = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=True), nn.Sigmoid())
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h, w = x.shape[-2:]
        freq = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        amp = torch.abs(freq) + 1e-8
        phase = torch.angle(freq)

        amp_modulated = amp * (1 + torch.tanh(self.amp_modulator(amp)))
        phase_offset = torch.tanh(self.phase_adaptor(phase)) * 0.1
        phase_modulated = phase + phase_offset
        freq_restored = amp_modulated * torch.exp(1j * phase_modulated)

        x_freq_out = torch.fft.irfft2(freq_restored, s=(h, w), dim=(-2, -1), norm='ortho')
        return x + torch.tanh(self.res_scale) * self.out_proj(x_freq_out) * self.spatial_gate(x)


# ==============================================================================
# 5. PWMBlock
# ==============================================================================
class PWMBlock(nn.Module):
    def __init__(self, dim, dir_V='fwd', dir_H='fwd', phys_dim=64):
        super().__init__()
        self.phys_modulator = PhysicsGuidedModulator(
            feat_dim=dim,
            phys_dim=phys_dim
        )

        self.lfssm = HierarchicalLFSSM(
            dim,
            dir_V=dir_V,
            dir_H=dir_H
        )
        self.guide_LL = SpatialGuidedLowFreqModulator(dim)

        self.refine_HL = DirectionalRefineBlock(dim, direction='vertical')
        self.refine_LH = DirectionalRefineBlock(dim, direction='horizontal')
        self.refine_HH = DirectionalRefineBlock(dim, direction='diagonal')
        self.dynamic_guide = DynamicDirectionalGuide(dim=dim)

        self.norm_hf = ChannelLayerNorm(dim * 3)
        self.hf_msca = MultiScaleContextBlock(dim * 3)
        self.spatial_refiner = nn.Sequential(
            nn.Conv2d(
                dim * 3,
                dim * 3,
                3,
                1,
                1,
                groups=dim * 3
            ),
            nn.Conv2d(dim * 3, dim * 3, 1)
        )
        self.spatial_modulator = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, phys_map):
        # Wavelet decomposition
        x_LL, x_HL, x_LH, x_HH = dwt_split(x)

        # Low-frequency restoration with the input-dependent spatial map
        mod_LL = self.phys_modulator(x_LL, phys_map)
        mod_LL = self.lfssm(mod_LL)
        mod_LL = self.guide_LL(mod_LL)

        # Direction-specific high-frequency refinement
        feat_HL = self.refine_HL(x_HL)
        feat_LH = self.refine_LH(x_LH)
        feat_HH = self.refine_HH(x_HH)

        # Restored low-frequency representation guides high-frequency recovery
        mod_HL, mod_LH, mod_HH = self.dynamic_guide(
            mod_LL,
            feat_HL,
            feat_LH,
            feat_HH
        )

        # High-frequency aggregation and inverse wavelet reconstruction
        x_High_merged = torch.cat(
            [mod_HL, mod_LH, mod_HH],
            dim=1
        )
        feat_High = self.hf_msca(
            self.norm_hf(x_High_merged)
        )
        cleaned_High = feat_High + self.spatial_refiner(feat_High)

        x_restored = iwt(mod_LL, cleaned_High)
        return x + self.gamma * (
            x_restored * self.spatial_modulator(x_restored)
        )


# ==============================================================================
# 6. 主网络：PWM-Net
# ==============================================================================
class PWMNet(nn.Module):
    def __init__(self, in_ch=3, dim=64, phys_dim=64):
        super().__init__()
        dims = [dim, dim * 2, dim * 2, dim * 4]

        # The spatial conditioning map is estimated only from the input image.
        # No external prior is required.
        self.phys_estimator = SpatialPhysicsEstimator(
            in_ch=in_ch,
            phys_dim=phys_dim
        )

        self.stem = SimpleResBlock(in_ch, dim)

        self.level1 = PWMBlock(
            dims[0],
            dir_V='fwd',
            dir_H='fwd',
            phys_dim=phys_dim
        )
        self.down1 = PixelUnshuffleDown(dims[0], dims[1])

        self.level2 = PWMBlock(
            dims[1],
            dir_V='bwd',
            dir_H='bwd',
            phys_dim=phys_dim
        )
        self.down2 = PixelUnshuffleDown(dims[1], dims[2])

        self.level3 = PWMBlock(
            dims[2],
            dir_V='fwd',
            dir_H='bwd',
            phys_dim=phys_dim
        )
        self.down3 = PixelUnshuffleDown(dims[2], dims[3])

        self.bottleneck = PWMBlock(
            dims[3],
            dir_V='bwd',
            dir_H='fwd',
            phys_dim=phys_dim
        )

        self.up3 = PixelShuffleUp(dims[3], dims[2])
        self.fuse_skip3 = nn.Conv2d(dims[2] * 2, dims[2], 1)
        self.dec3 = PWMBlock(
            dims[2],
            dir_V='fwd',
            dir_H='bwd',
            phys_dim=phys_dim
        )

        self.up2 = PixelShuffleUp(dims[2], dims[1])
        self.fuse_skip2 = nn.Conv2d(dims[1] * 2, dims[1], 1)
        self.dec2 = PWMBlock(
            dims[1],
            dir_V='bwd',
            dir_H='bwd',
            phys_dim=phys_dim
        )

        self.up1 = PixelShuffleUp(dims[1], dims[0])
        self.fuse_skip1 = nn.Conv2d(dims[0] * 2, dims[0], 1)
        self.dec1 = PWMBlock(
            dims[0],
            dir_V='fwd',
            dir_H='fwd',
            phys_dim=phys_dim
        )

        self.global_freq_refine = PhysicallySoundFrequencyRefine(dims[0])
        self.out_conv = nn.Conv2d(dims[0], in_ch, 3, 1, 1)

    def safe_concat(self, x_up, x_skip):
        if x_up.shape[-2:] != x_skip.shape[-2:]:
            x_up = F.interpolate(
                x_up,
                size=x_skip.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        return torch.cat([x_up, x_skip], dim=1)

    def forward(self, x):
        # Input-dependent spatial conditioning map.
        phys_map = self.phys_estimator(x)

        feat = self.stem(x)

        feat1 = self.level1(
            feat,
            phys_map
        )

        feat2 = self.level2(
            self.down1(feat1),
            phys_map
        )

        feat3 = self.level3(
            self.down2(feat2),
            phys_map
        )

        feat_bot = self.bottleneck(
            self.down3(feat3),
            phys_map
        )

        feat_up3 = self.dec3(
            self.fuse_skip3(
                self.safe_concat(
                    self.up3(feat_bot),
                    feat3
                )
            ),
            phys_map
        )

        feat_up2 = self.dec2(
            self.fuse_skip2(
                self.safe_concat(
                    self.up2(feat_up3),
                    feat2
                )
            ),
            phys_map
        )

        feat_up1 = self.dec1(
            self.fuse_skip1(
                self.safe_concat(
                    self.up1(feat_up2),
                    feat1
                )
            ),
            phys_map
        )

        feat_up1 = self.global_freq_refine(feat_up1)
        restored_img = self.out_conv(feat_up1)

        return restored_img
