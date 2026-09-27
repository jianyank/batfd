"""单体分辨的非对称自编码器（轴线①）。

论文把所有 20 通道压成一个 256 维潜向量，再「令**一部分**潜变量与 SOC、内阻
对齐」，但**从未指明哪个潜变量对应哪个单体** —— 论文 §4.2 / 图 7 承认这失败了：
潜变量 *"fails to capture the inter-cell discrepancies, particularly the
abnormal evolution of Cell 8"*。

本实现把它变成**结构约束**：8 个单体各有一个**权重共享**的编码分支，单体 c 的
分支只吃单体 c 自己的电压 + 共享总电流，因此槽位 c 在结构上只可能来自单体 c 的
信号。再把这个槽位对齐到单体 c 的内阻 —— 单体间差异**被迫**表示出来。

结构（``x`` (B, C_in, T)，C_in=20 = 8 电压 + 8 电流 + 4 温度，T=256）::

    单体分支（8 个单体共享同一套权重，逐单体独立前向）
      [v_c, i] (B, 2, 256)
        Conv1d(2→16, k9, s2) → Conv1d(16→16, k5, s2) → Conv1d(16→8, k5, s2) → GELU
        AdaptiveAvgPool → (B, 8) → Linear(8 → cell_latent) → mu_c, logvar_c ∈ (B, 8, cell_latent)

    包级分支（吃全部 C_in 通道，承载 SOC / 工况 / 温度）
        Conv1d(C_in→32, k7, s2) → Conv1d(32→32, k5, s2) → Conv1d(32→16, k5, s2) → GELU
        AdaptiveAvgPool → (B, 16) → Linear(16 → pack_latent) → mu_p, logvar_p ∈ (B, pack_latent)

    物理槽：a_c = mu_c[..., 0]（对齐单体 c 的内阻）；soc = mu_p[..., 0]（对齐 SOC）

    解码器（8 个单体同样权重共享）
        输入 [z_c (cell_latent), p (pack_latent)]
        Linear(→ 16×8)  → (B, 16, 8)
        ConvT1d ×5（每层 k4 s2 p1，长度翻倍：8→16→32→64→128→256）
        输出 v̂_c (B, 1, 256)

    非对称：输入 20 通道，**只重建 8 路单体电压**（与论文 §2.2 / Table 2 一致）

消融开关
--------
``model.cell_resolved = false``
    单体分支替换成「包级潜向量广播到所有单体」，即论文那种未指定对应关系的
    形态。用于隔离「单体分辨结构本身」带来多少增益。
``train.lambda2 = 0`` / ``train.lambda3 = 0``
    分别关掉物理对齐与 KL 正则。
``model.use_sibling_context = true``
    解码时额外喂入兄弟单体潜变量的均值。默认关闭 —— 兄弟信息会让单个单体的
    故障被邻胞信息掩盖，反而降低检测灵敏度。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustNormalizer(nn.Module):
    """逐通道稳健标准化：``(x − median) / max(1.4826·MAD, floor)``。

    用中位数/MAD 而非均值/标准差：待检出的异常会把均值和标准差抬高，造成阈值
    虚高、真异常漏检。统计量**只在训练集上拟合**，以 buffer 随模型存盘，保证推理
    时可复现、且测试集信息不泄漏。

    ``floor`` 是相对下限（乘以 |median|），防止近零方差通道被放大成噪声主导。
    """

    def __init__(self, n_channels: int, floor: float = 1e-3) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.floor = floor
        self.register_buffer("median", torch.zeros(n_channels))
        self.register_buffer("scale", torch.ones(n_channels))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "RobustNormalizer":
        """``x`` 形状 ``(N, C, T)`` 或 ``(N, C)``，按通道统计。"""
        if x.dim() == 3:
            flat = x.permute(1, 0, 2).reshape(x.shape[1], -1)
        elif x.dim() == 2:
            flat = x.t()
        else:
            raise ValueError(f"期望 (N,C,T) 或 (N,C)，收到 {tuple(x.shape)}")
        med = flat.median(dim=1).values
        mad = (flat - med[:, None]).abs().median(dim=1).values * 1.4826
        floor = self.floor * med.abs().clamp_min(1e-12)
        self.median.copy_(med)
        self.scale.copy_(torch.maximum(mad, floor))
        self.fitted.fill_(True)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.fitted):
            raise RuntimeError("RobustNormalizer 尚未 fit；必须先只用训练集拟合")
        m = self.median.view(1, -1, 1)
        s = self.scale.view(1, -1, 1)
        return (x - m) / s

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.scale.view(1, -1, 1) + self.median.view(1, -1, 1)

    def extra_repr(self) -> str:
        return f"n_channels={self.n_channels}, floor={self.floor}, fitted={bool(self.fitted)}"


@dataclass
class CellAEOutput:
    """一次前向的全部中间量，便于做诊断与消融。"""

    v_hat: torch.Tensor        # (B, n_cells, T) 重建电压（标准化空间）
    mu_cell: torch.Tensor      # (B, n_cells, cell_latent)
    logvar_cell: torch.Tensor
    mu_pack: torch.Tensor      # (B, pack_latent)
    logvar_pack: torch.Tensor
    z_cell: torch.Tensor       # (B, n_cells, cell_latent) 采样后的潜变量
    z_pack: torch.Tensor

    @property
    def a_cell(self) -> torch.Tensor:
        """物理槽：逐单体的内阻对齐量，形状 (B, n_cells)。"""
        return self.mu_cell[..., 0]

    @property
    def soc_pack(self) -> torch.Tensor:
        """物理槽：SOC 对齐量，形状 (B,)。"""
        return self.mu_pack[..., 0]


class _ConvStack1d(nn.Module):
    """若干层 Conv1d(k, stride 2) + GELU。"""

    def __init__(self, in_ch: int, out_chs: list[int], k: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_ch
        for oc in out_chs:
            layers += [
                nn.Conv1d(prev, oc, kernel_size=k, stride=2, padding=k // 2),
                nn.GELU(),
            ]
            prev = oc
        self.net = nn.Sequential(*layers)
        self.out_ch = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CellEncoder(nn.Module):
    """单体分支：吃 [v_c, i]，输出该单体的潜变量分布。

    权重在 8 个单体间共享 —— 结构上保证单体分辨的关键：输入只有单体 c 自己的
    电压与共享电流，输出就只能是单体 c 的函数。
    """

    def __init__(self, latent: int, hidden: list[int] | None = None, in_ch: int = 2) -> None:
        super().__init__()
        hidden = hidden or [16, 16, 8]
        self.conv = _ConvStack1d(in_ch, hidden, k=5)
        self.head = nn.Linear(self.conv.out_ch, latent * 2)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).mean(dim=-1)          # (B*n_cells, hidden)
        mu, logvar = self.head(h).chunk(2, dim=-1)
        # logvar 夹在合理范围，避免早期训练数值爆炸
        return mu, logvar.clamp(-8.0, 8.0)


class PackEncoder(nn.Module):
    """包级分支：吃全部输入通道，输出 SOC / 工况 / 温度相关的潜变量。"""

    def __init__(self, in_ch: int, latent: int, hidden: list[int] | None = None) -> None:
        super().__init__()
        hidden = hidden or [32, 32, 16]
        self.conv = _ConvStack1d(in_ch, hidden, k=7)
        self.head = nn.Linear(self.conv.out_ch, latent * 2)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).mean(dim=-1)
        mu, logvar = self.head(h).chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 8.0)


class CellDecoder(nn.Module):
    """单体解码器：从 [z_c, p] 重建单体 c 的电压，权重在单体间共享。

    长度链：8 → 16 → 32 → 64 → 128 → 256（5 层 ConvTranspose1d，k=4 s=2 p=1，
    每层恰好翻倍）。
    """

    def __init__(self, latent: int, t_out: int = 256, base_len: int = 8, hidden: int = 16) -> None:
        super().__init__()
        n_up = 5
        assert base_len * (2**n_up) == t_out, (
            f"base_len={base_len} 经 {n_up} 层翻倍后为 {base_len * 2**n_up}，与 t_out={t_out} 不符"
        )
        self.base_len = base_len
        self.fc = nn.Linear(latent, hidden * base_len)
        blocks: list[nn.Module] = []
        for _ in range(n_up - 1):
            blocks += [nn.ConvTranspose1d(hidden, hidden, 4, stride=2, padding=1), nn.GELU()]
        blocks.append(nn.ConvTranspose1d(hidden, 1, 4, stride=2, padding=1))
        self.up = nn.Sequential(*blocks)

    def init_current_skip(self, hidden: int, k: int = 9) -> None:
        """电流直通支路（可选）。见 :class:`CellResolvedAE` 的 ``decoder_current_skip``。"""
        self.cur_branch = nn.Sequential(
            nn.Conv1d(1, hidden, k, padding=k // 2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, k, padding=k // 2),
            nn.GELU(),
            nn.Conv1d(hidden, 1, k, padding=k // 2),
        )

    def forward(self, z: torch.Tensor, cur: torch.Tensor | None = None) -> torch.Tensor:
        """``z`` (B, dec_latent)；``cur`` (B, T) 为可选的电流直通输入。"""
        h = self.fc(z).reshape(z.shape[0], -1, self.base_len)
        out = self.up(h).squeeze(1)                       # (B, T)
        if cur is not None and hasattr(self, "cur_branch"):
            out = out + self.cur_branch(cur.unsqueeze(1)).squeeze(1)
        return out


class CellResolvedAE(nn.Module):
    """单体分辨非对称自编码器（见模块 docstring 的结构图）。"""

    # 前向需要额外的电流通道（单体分支的共享输入）。
    # 这是个显式标记而不是靠 hasattr 猜：LFAAE 也有 cell_norm，
    # 用属性探测会误判分支（实测踩过这个坑）。
    requires_current: bool = True

    def __init__(
        self,
        n_cells: int,
        in_channels: int,
        cell_latent: int = 4,
        pack_latent: int = 8,
        t_len: int = 256,
        cell_conv: list[int] | None = None,
        pack_conv: list[int] | None = None,
        decoder_hidden: int = 16,
        cell_resolved: bool = True,
        variational: bool = True,
        use_sibling_context: bool = False,
        decoder_current_skip: bool = False,
        norm_floor: float = 1e-3,
    ) -> None:
        super().__init__()
        self.n_cells = n_cells
        self.in_channels = in_channels
        self.t_len = t_len
        self.cell_latent = cell_latent
        self.pack_latent = pack_latent
        self.decoder_hidden = decoder_hidden
        self.cell_resolved = cell_resolved
        self.variational = variational
        self.use_sibling_context = use_sibling_context
        self.decoder_current_skip = decoder_current_skip

        self.normalizer = RobustNormalizer(in_channels, floor=norm_floor)
        self.cell_norm = RobustNormalizer(n_cells, floor=norm_floor)

        self.pack_encoder = PackEncoder(in_channels, pack_latent, pack_conv)
        if cell_resolved:
            self.cell_encoder = CellEncoder(cell_latent, cell_conv, in_ch=2)  # [单体电压, 共享电流]
        else:
            # 消融：单个共享编码器，输出一个潜向量广播给所有单体
            self.shared_encoder = CellEncoder(cell_latent, cell_conv, in_ch=in_channels)

        dec_latent = cell_latent + pack_latent + (cell_latent if use_sibling_context else 0)
        self.decoder = CellDecoder(
            dec_latent, t_out=t_len, base_len=8, hidden=decoder_hidden
        )
        # 电流直通支路：电压波形主要由电流驱动，而电流被压成全局向量后，解码器
        # 要重塑出 256 步时序 —— 实测这是重建精度的主瓶颈（加大潜变量维度只带来
        # 边际改善）。电流是外生输入，编解码两端都可见；单体特有的偏差仍必须走
        # 潜变量，否则会绕开瓶颈、失去检测能力。
        if decoder_current_skip:
            self.decoder.init_current_skip(decoder_hidden)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.variational or not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def encode(
        self, x_norm: torch.Tensor, v_norm: torch.Tensor, i_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (mu_cell, logvar_cell, mu_pack, logvar_pack)。"""
        mu_p, logvar_p = self.pack_encoder(x_norm)

        b, _, t = v_norm.shape
        if self.cell_resolved:
            # 每个单体输入 [自己的电压, 共享电流]；电流是包级量，广播到所有单体。
            i_bcast = i_norm.unsqueeze(1).expand(-1, self.n_cells, -1)   # (B, n_cells, T)
            cell_in = torch.stack([v_norm, i_bcast], dim=2)              # (B, n_cells, 2, T)
            cell_in = cell_in.reshape(b * self.n_cells, 2, t)
            mu_c, logvar_c = self.cell_encoder(cell_in)
            mu_c = mu_c.reshape(b, self.n_cells, -1)
            logvar_c = logvar_c.reshape(b, self.n_cells, -1)
        else:
            mu_c, logvar_c = self.shared_encoder(x_norm)         # (B, latent)
            mu_c = mu_c.unsqueeze(1).expand(-1, self.n_cells, -1).contiguous()
            logvar_c = logvar_c.unsqueeze(1).expand(-1, self.n_cells, -1).contiguous()
        return mu_c, logvar_c, mu_p, logvar_p

    def decode(
        self, z_cell: torch.Tensor, z_pack: torch.Tensor, i_norm: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, n, _ = z_cell.shape
        p = z_pack.unsqueeze(1).expand(-1, n, -1)
        if self.use_sibling_context:
            total = z_cell.sum(dim=1, keepdim=True)
            sib = (total - z_cell) / max(n - 1, 1)
            z_in = torch.cat([z_cell, sib, p], dim=-1)
        else:
            z_in = torch.cat([z_cell, p], dim=-1)
        z_flat = z_in.reshape(b * n, -1)

        cur = None
        if self.decoder_current_skip:
            if i_norm is None:
                raise ValueError("decoder_current_skip=True 时必须把归一化后的电流传给 decode()")
            # 电流是包级量，广播到每个单体（与编码端一致）
            cur = i_norm.unsqueeze(1).expand(-1, n, -1).reshape(b * n, -1)

        v_hat = self.decoder(z_flat, cur).reshape(b, n, self.t_len)
        return v_hat

    def forward(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        i: torch.Tensor,
        *,
        return_output: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, CellAEOutput]:
        """``x`` (B, C_in, T) 原始输入；``v`` (B, n_cells, T) 单体电压；``i`` (B, T) 总电流。

        返回标准化空间里的重建电压 ``(B, n_cells, T)``。
        """
        x_norm = self.normalizer(x)
        v_norm = self.cell_norm(v)
        i_norm = self._normalize_current(i)

        mu_c, logvar_c, mu_p, logvar_p = self.encode(x_norm, v_norm, i_norm)
        z_c = self._reparameterize(mu_c, logvar_c)
        z_p = self._reparameterize(mu_p, logvar_p)
        v_hat = self.decode(z_c, z_p, i_norm)

        if not return_output:
            return v_hat
        out = CellAEOutput(
            v_hat=v_hat,
            mu_cell=mu_c,
            logvar_cell=logvar_c,
            mu_pack=mu_p,
            logvar_pack=logvar_p,
            z_cell=z_c,
            z_pack=z_p,
        )
        return v_hat, out

    def _normalize_current(self, i: torch.Tensor) -> torch.Tensor:
        """只归一化「总电流」那一列，返回 ``(B, T)``。

        ⚠ 不能写成 ``self.normalizer(i.unsqueeze(1)).squeeze(1)``：
        :class:`RobustNormalizer` 的统计量长度是输入通道数，会按通道广播，
        把 ``(B, 1, T)`` 变成 ``(B, 20, T)``（实测踩过这个坑）。
        这里显式取出电流所在通道的中心与尺度，形状与语义都不含糊。
        """
        idx = self._current_channel_index
        if idx is None:
            raise RuntimeError(
                "未设置电流所在输入通道（set_current_channel_index），"
                "无法归一化电流 —— 这会让单体分支用到错误尺度的输入"
            )
        med = self.normalizer.median[idx]
        scale = self.normalizer.scale[idx]
        return (i - med) / scale

    @torch.no_grad()
    def fit_normalizers(self, x: torch.Tensor, v: torch.Tensor, i: torch.Tensor) -> None:
        """只用训练集拟合输入与电压的稳健标准化统计量。"""
        self.normalizer.fit(x)
        self.cell_norm.fit(v)
        # 电流用的是 normalizer 的第 (n_cells) 个通道（见 build 里的列序），
        # 单独再核对一次，避免列序改动后静默错位
        i_fit = RobustNormalizer(1, floor=self.normalizer.floor).fit(i.unsqueeze(1))
        idx = self._current_channel_index
        if idx is None:
            raise RuntimeError("未记录电流所在输入通道，无法校准电流标准化统计量")
        self.normalizer.median[idx] = i_fit.median[0]
        self.normalizer.scale[idx] = i_fit.scale[0]

    @property
    def _current_channel_index(self) -> int | None:
        return getattr(self, "_cur_idx", None)

    def set_current_channel_index(self, idx: int) -> None:
        """告诉模型「总电流在输入张量的第几个通道」，供标准化校准使用。

        显式设置而非硬编码：``channels.dedup_current`` 或列序改变时会立刻报错，
        而不是静默用错通道。
        """
        self._cur_idx = int(idx)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
