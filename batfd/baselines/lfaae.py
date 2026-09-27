"""基线：论文的「潜变量信息引导的非对称自编码器」（LFAAE）。

按论文 **Table 1** 逐层实现，**不是**照抄旧 MATLAB 脚本 —— 旧脚本
（``1/A7_ModelTrain.m``）的结构在 MATLAB R2025b 里根本无法构建（瓶颈
``fullyConnectedLayer(2048)`` 比编码器输出 672 维还宽，且 ``[42 1]`` 的转置卷积
还原不出 256×20）。

Table 1 尺寸链（记法：层  输出时间×通道；已用 ``out = (in−1)·s + k − 2p`` 逐层核算）：
输入 256×20（8 电压 + 8 电流 + 4 温度）→ Conv 16×[5×2] s[1,2] 252×10
→ MaxPool [3×1] s[3,1] 84×10 → Conv 32×[1×10] s[1,1] 84×1
→ MaxPool [3×1] s[3,1] 28×1 → Conv 256×[28×1] s[1,1] 1×1（潜变量 256 维，与 §4.1 一致）
→ ConvT 32×[28×1] s[2,1] 28×1 → ConvT 32×[3×1] s[3,1] 84×1
→ ConvT 16×[1×8] s[**1,3**] 84×8（见下）→ ConvT 16×[3×1] s[3,1] 252×8
→ ConvT 1×[5×1] s[1,1] 256×8（与 Table 2「输出 256×8×1」一致）。

**一处必要的订正**：论文 Table 1 把第 3 个转置卷积的 stride 写成 ``[3,1]``，
按该值算出来是 250 而不是 84，整条链断掉；写成 ``[1,3]`` 则前后严格自洽，输出恰好
是 Table 2 声明的 256×8×1。因此判为论文排印错误，本实现取 ``[1,3]``。

论文未说明而必须自行决定的两点（如实记录，不假装照做）：
1. **激活函数**：Table 1 只列了 conv/pool，没写激活。旧 MATLAB 脚本通篇没有 ReLU
   （只有 BatchNorm），而纯线性堆叠难以拟合。本实现默认 ReLU，并保留
   ``activation="none"`` 以便量化这一选择的影响。
2. **BatchNorm**：论文未提。旧脚本每个卷积后都有 BN。本实现默认不加（更贴近
   Table 1 的字面内容），``use_bn=True`` 可开。

损失：论文 Eq.(1) 取 λ2=λ3=0（只留重建项）。**这是被迫的** —— 论文全文没有给出
λ1/λ2/λ3 的取值，其 L_late 又要求「一部分潜变量」对齐物理量但未指明是哪一部分、
也没有可用的接口，因此无法忠实复现那两项。报告中必须声明这一局限。
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .. import progress
from ..data import dataset as ds_mod
from ..models.cell_ae import RobustNormalizer

# 论文 Table 1 的层超参（集中在此，便于与原文逐项对照）
ENC_CONV1 = dict(out_channels=16, kernel=(5, 2), stride=(1, 2))
ENC_POOL1 = dict(kernel=(3, 1), stride=(3, 1))
ENC_CONV2 = dict(out_channels=32, kernel=(1, 10), stride=(1, 1))
ENC_POOL2 = dict(kernel=(3, 1), stride=(3, 1))
ENC_BOTTLENECK = dict(out_channels=256, kernel=(28, 1), stride=(1, 1))

DEC_CONV1 = dict(out_channels=32, kernel=(28, 1), stride=(2, 1))
DEC_CONV2 = dict(out_channels=32, kernel=(3, 1), stride=(3, 1))
# ↓ 论文写 [3,1]，按其自身尺寸链应为 [1,3]（见模块 docstring 的订正说明）
DEC_CONV3 = dict(out_channels=16, kernel=(1, 8), stride=(1, 3), paper_stride=(3, 1))
DEC_CONV4 = dict(out_channels=16, kernel=(3, 1), stride=(3, 1))
DEC_CONV5 = dict(out_channels=1, kernel=(5, 1), stride=(1, 1))


def _act(kind: str) -> nn.Module:
    if kind == "none":
        return nn.Identity()
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "gelu":
        return nn.GELU()
    raise ValueError(f"未知激活 {kind}")


class LFAAE(nn.Module):
    """论文 Table 1 的卷积自编码器。

    输入 ``(B, C_in, T)``（库内统一约定：通道优先），内部转成论文的
    ``(B, 1, T, C)``（时间当高、通道当宽），输出重建的 8 路电压 ``(B, n_cells, T)``。
    非对称性由通道维的卷积承担：``[1×10]`` 把宽 10 压成 1，解码端 ``[1×8]`` 再展开成 8。
    """

    # 前向只需要 x（见 batfd/models/inference.py 的分派逻辑）
    requires_current: bool = False

    def __init__(
        self,
        in_channels: int = 20,
        n_cells: int = 8,
        t_len: int = 256,
        *,
        activation: str = "relu",
        use_bn: bool = False,
        norm_floor: float = 1e-3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_cells = n_cells
        self.t_len = t_len

        self.normalizer = RobustNormalizer(in_channels, floor=norm_floor)
        self.cell_norm = RobustNormalizer(n_cells, floor=norm_floor)

        def block(out_ch, kernel, stride):
            layers = [nn.Conv2d(1, out_ch, kernel, stride=stride)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(_act(activation))
            return layers

        enc: list[nn.Module] = []
        enc += block(ENC_CONV1["out_channels"], ENC_CONV1["kernel"], ENC_CONV1["stride"])
        enc.append(nn.MaxPool2d(ENC_POOL1["kernel"], stride=ENC_POOL1["stride"]))
        enc.append(nn.Conv2d(ENC_CONV1["out_channels"], ENC_CONV2["out_channels"],
                             ENC_CONV2["kernel"], stride=ENC_CONV2["stride"]))
        if use_bn:
            enc.append(nn.BatchNorm2d(ENC_CONV2["out_channels"]))
        enc.append(_act(activation))
        enc.append(nn.MaxPool2d(ENC_POOL2["kernel"], stride=ENC_POOL2["stride"]))
        enc.append(nn.Conv2d(ENC_CONV2["out_channels"], ENC_BOTTLENECK["out_channels"],
                             ENC_BOTTLENECK["kernel"], stride=ENC_BOTTLENECK["stride"]))
        if use_bn:
            enc.append(nn.BatchNorm2d(ENC_BOTTLENECK["out_channels"]))
        enc.append(_act(activation))
        self.encoder = nn.Sequential(*enc)

        def tblock(in_ch, out_ch, kernel, stride):
            layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel, stride=stride)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(_act(activation))
            return layers

        self.decoder = nn.Sequential(
            *tblock(256, DEC_CONV1["out_channels"], DEC_CONV1["kernel"], DEC_CONV1["stride"]),
            *tblock(DEC_CONV1["out_channels"], DEC_CONV2["out_channels"],
                    DEC_CONV2["kernel"], DEC_CONV2["stride"]),
            *tblock(DEC_CONV2["out_channels"], DEC_CONV3["out_channels"],
                    DEC_CONV3["kernel"], DEC_CONV3["stride"]),
            *tblock(DEC_CONV3["out_channels"], DEC_CONV4["out_channels"],
                    DEC_CONV4["kernel"], DEC_CONV4["stride"]),
            # 最后一层不加激活：重建目标是连续电压值，输出层不该被截断
            nn.ConvTranspose2d(DEC_CONV4["out_channels"], 1,
                               DEC_CONV5["kernel"], stride=DEC_CONV5["stride"]),
        )

    def forward(self, x: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
        """``x``: (B, C_in, T) -> 重建电压 (B, n_cells, T)，处于标准化空间。

        ``normalize=False`` 跳过输入归一化，供形状核对等不关心数值的场合使用
        （否则未拟合的标准化器会按设计直接报错）。
        """
        xn = self.normalizer(x) if normalize else x  # (B, C, T)
        h = xn.permute(0, 2, 1).unsqueeze(1)         # (B, 1, T, C)
        z = self.encoder(h)                          # (B, 256, 1, 1)
        y = self.decoder(z)                          # (B, 1, T, C_out)
        return y.squeeze(1).permute(0, 2, 1).contiguous()   # (B, C_out, T)

    @torch.no_grad()
    def fit_normalizers(self, x: torch.Tensor, v: torch.Tensor) -> None:
        """只用训练集拟合输入与电压的稳健标准化统计量。"""
        self.normalizer.fit(x)
        self.cell_norm.fit(v)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build(cfg: dict) -> LFAAE:
    from ..data import channels

    return LFAAE(
        in_channels=len(channels.input_cols(cfg)),
        n_cells=len(channels.cell_voltage_cols(cfg)),
        t_len=int(cfg["data"]["expected_rows"]),
        activation=str(cfg.get("baseline", {}).get("activation", "relu")),
        use_bn=bool(cfg.get("baseline", {}).get("use_bn", False)),
        norm_floor=float(cfg["model"]["norm_floor"]),
    )


def train(
    cfg: dict,
    cache: dict,
    *,
    out_dir: Path,
    tag: str = "lfaae",
    epochs: int | None = None,
    verbose: bool = True,
) -> dict:
    """训练基线。损失只含重建项（论文 Eq.1 取 λ2=λ3=0，理由见模块 docstring）。"""
    from ..models.train import pick_device, seed_everything

    tr = cfg["train"]
    seed_everything(int(tr["seed"]))
    device = pick_device(cfg)

    split = ds_mod.chronological_split(
        np.asarray(cache["ids"]), cache.get("time"), float(cfg["split"]["val_ratio"])
    )
    train_ds = ds_mod.WindowDataset(cfg, cache, split.train_idx)
    val_ds = ds_mod.WindowDataset(cfg, cache, split.val_idx)

    model = build(cfg).to(device)
    out_dir = Path(out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[lfaae] 参数量 {model.param_count():,}  设备 {device}")
    print(f"[lfaae] 训练窗口 {len(train_ds)}  验证窗口 {len(val_ds)}")

    fit_loader = DataLoader(train_ds, batch_size=int(tr["batch_size"]), shuffle=False, num_workers=0)
    xs, vs = [], []
    for b in fit_loader:
        xs.append(b["x"])
        vs.append(b["v"])
    model.fit_normalizers(torch.cat(xs), torch.cat(vs))
    del xs, vs
    print("[lfaae] 标准化统计量已拟合（仅训练集）")

    train_loader = DataLoader(train_ds, batch_size=int(tr["batch_size"]), shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(tr["batch_size"]), shuffle=False, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=float(tr["lr"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    epochs = int(epochs if epochs is not None else tr["max_epochs"])

    history: list[dict] = []
    best = float("inf")
    best_ep = -1
    bad = 0
    t0 = time.time()

    def run(loader, train_mode: bool, desc: str = ""):
        model.train(train_mode)
        tot, n = 0.0, 0
        it = progress.bar(loader, desc=desc, leave=False) if desc else loader
        for b in it:
            x = b["x"].to(device, non_blocking=True)
            v = b["v"].to(device, non_blocking=True)
            with torch.set_grad_enabled(train_mode):
                loss = nn.functional.mse_loss(model(x), model.cell_norm(v))
            if train_mode:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr["grad_clip"]))
                opt.step()
            tot += float(loss.detach())
            n += 1
            if desc:
                it.set_postfix({"mse": f"{loss.detach().item():.5f}"})  # type: ignore[attr-defined]
        if desc:
            it.close()  # type: ignore[attr-defined]
        return tot / max(n, 1)

    epoch_bar = progress.bar(range(1, epochs + 1), desc=f"训练 {tag}", unit="ep")
    for ep in epoch_bar:
        ltr = run(train_loader, True, f"  ep{ep} train")
        lva = run(val_loader, False, f"  ep{ep} val  ")
        sched.step(lva)
        history.append({"epoch": ep, "train_mse": ltr, "val_mse": lva,
                        "lr": opt.param_groups[0]["lr"]})
        if lva < best - 1e-7:
            best, best_ep, bad = lva, ep, 0
            torch.save({"model_state": model.state_dict(), "epoch": ep, "val_mse": lva,
                        "cfg_snapshot": cfg, "history": history}, out_dir / "best.pt")
        else:
            bad += 1
        epoch_bar.set_postfix(  # type: ignore[attr-defined]
            {
                "tr": f"{ltr:.5f}",
                "va": f"{lva:.5f}",
                "best": f"{best:.5f}@{best_ep}",
                "bad": f"{bad}/{int(tr['patience']) * 3}",
                "lr": f"{opt.param_groups[0]['lr']:.1e}",
            }
        )
        if verbose and (ep == 1 or ep % 10 == 0 or lva <= best):
            progress.log(
                f"  ep{ep:>4}  train {ltr:.6f}  val {lva:.6f}  "
                f"lr {opt.param_groups[0]['lr']:.2e}{'  *best*' if lva <= best else ''}"
            )
        if bad >= int(tr["patience"]) * 3:   # 基线收敛慢，早停放宽容忍
            epoch_bar.close()  # type: ignore[attr-defined]
            progress.log(f"  验证损失连续 {bad} 个 epoch 未改善，早停于 ep{ep}")
            break
    else:
        epoch_bar.close()  # type: ignore[attr-defined]

    torch.save({"model_state": model.state_dict(), "epoch": len(history), "val_mse": best,
                "cfg_snapshot": cfg, "history": history}, out_dir / "last.pt")
    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    print(f"[lfaae] 完成 {len(history)} epoch，{time.time() - t0:.1f}s，最优 val MSE={best:.6f} @ ep{best_ep}")
    return {"best_val_mse": best, "best_epoch": best_ep, "epochs_run": len(history),
            "out_dir": str(out_dir), "history": history}


def load(path: str | Path, cfg: dict) -> LFAAE:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build(cfg)
    model.load_state_dict(ckpt["model_state"])
    return model


# 论文 Table 1 声明的逐层尺寸，记作 (时间, 宽, 通道)。
# ⚠ 卷积张量是 (B, 通道, 高, 宽)，核对时不能直接取 shape[2:]（会把 256 通道的瓶颈
# 误报成 (1,1)）；逐项吻合是「实现与论文是同一个网络」的前提。
TABLE1_SHAPES = [
    ("输入", (256, 20, 1)),
    ("enc_conv1 16x[5x2] s[1,2]", (252, 10, 16)),
    ("enc_pool1 [3x1] s[3,1]", (84, 10, 16)),
    ("enc_conv2 32x[1x10] s[1,1]", (84, 1, 32)),
    ("enc_pool2 [3x1] s[3,1]", (28, 1, 32)),
    ("enc_bottleneck 256x[28x1]", (1, 1, 256)),   # 潜变量 256 维（§4.1）
    ("dec_conv1 32x[28x1] s[2,1]", (28, 1, 32)),
    ("dec_conv2 32x[3x1] s[3,1]", (84, 1, 32)),
    ("dec_conv3 16x[1x8] s[1,3]", (84, 8, 16)),   # stride 已按尺寸链订正
    ("dec_conv4 16x[3x1] s[3,1]", (252, 8, 16)),
    ("dec_conv5 1x[5x1] s[1,1]", (256, 8, 1)),    # 与 Table 2「输出 256×8×1」一致
]


def verify_shapes(cfg: dict, *, verbose: bool = True) -> bool:
    """跑一次假前向，逐层核对是否与 Table 1 一致。

    这是 P3 的把关项：论文 Table 1 的 stride 有排印错误，光看代码看不出实现
    有没有真的对上原文的其余部分，必须用实际张量形状验证。
    """
    from ..data import channels

    model = build(cfg).eval()
    x = torch.zeros(2, len(channels.input_cols(cfg)), int(cfg["data"]["expected_rows"]))

    # 用前向钩子只捕获与 Table 1 对应的层（卷积 / 池化 / 转置卷积）。
    # 不能简单地逐个模块记账 —— 激活层与 BN 也会产生输出，行数会对不上。
    captured: list[tuple[str, tuple[int, int, int]]] = []
    table_layers = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, (nn.Conv2d, nn.MaxPool2d, nn.ConvTranspose2d))
    ]

    def _record(name):
        def hook(mod, inp, out):
            # (B, C, H, W) -> (H, W, C)，与 TABLE1_SHAPES 的记法一致
            captured.append((name, (int(out.shape[2]), int(out.shape[3]), int(out.shape[1]))))
        return hook

    handles = [m.register_forward_hook(_record(n)) for n, m in table_layers]
    with torch.no_grad():
        model(x, normalize=False)
    for hd in handles:
        hd.remove()

    got: list[tuple[str, tuple[int, int, int]]] = [
        ("输入", (int(x.shape[2]), int(x.shape[1]), 1))
    ] + captured

    ok = True
    for (name_exp, shape_exp), (name_got, shape_got) in zip(TABLE1_SHAPES, got):
        match = shape_exp == shape_got
        ok = ok and match
        if verbose:
            print(f"    {'OK  ' if match else 'FAIL'} {name_exp:32s} 期望 {shape_exp}  实际 {shape_got}")
    return ok
