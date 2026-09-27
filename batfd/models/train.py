"""自定义训练循环。

论文 Eq.(1) 的 L_late 要把**指定的潜变量槽位**对齐到**指定的物理量**，L_reg 是
潜分布对 N(0,I) 的 KL，这需要在前向里拿到中间潜变量并按槽位分别求损失 ——
「输入→输出」式的封装层（含 MATLAB 的 ``trainNetwork`` + ``regressionLayer``）
都表达不了。

纪律
----
* 标准化统计量（输入、电压、物理目标）**一律只在训练集上拟合**；
* 早停看验证集总损失，验证集是各 ID 的时间尾部（见 data/dataset.py 的说明）；
* 随机种子固定，保证可复现；
* 每个 epoch 的三项损失分量分别记入 CSV —— 消融时要能看清增益来自哪一项。
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import progress
from ..data import channels, dataset as ds_mod
from .cell_ae import CellResolvedAE
from .losses import TargetScaler, total_loss


@dataclass
class TrainBundle:
    """训练所需的一切，便于在脚本之间传递与存盘。"""

    cfg: dict
    model: CellResolvedAE
    train_ds: ds_mod.WindowDataset
    val_ds: ds_mod.WindowDataset
    split: ds_mod.SplitIndex
    target_scaler: TargetScaler
    device: torch.device
    out_dir: Path


def pick_device(cfg: dict) -> torch.device:
    want = str(cfg["train"].get("device", "auto")).lower()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if want == "cuda" and not torch.cuda.is_available():
        print("[train] 配置要求 cuda 但 torch.cuda.is_available() 为假，回退 cpu")
        return torch.device("cpu")
    return torch.device(want)


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _iter_chunks(n: int, size: int):
    for s in range(0, n, size):
        yield s, min(s + size, n)


@torch.no_grad()
def _fit_normalizers(model: CellResolvedAE, loader: DataLoader, cfg: dict, device: torch.device) -> None:
    """只用训练集数据拟合输入/电压/电流的稳健标准化统计量（分块前向到 CPU 后一次写入 buffer）。"""
    xs, vs, iis = [], [], []
    for batch in loader:
        xs.append(batch["x"])
        vs.append(batch["v"])
        iis.append(batch["i"])
    x = torch.cat(xs, dim=0)
    v = torch.cat(vs, dim=0)
    i = torch.cat(iis, dim=0)

    model.normalizer.fit(x)
    model.cell_norm.fit(v)

    i_fit = type(model.normalizer)(1, floor=model.normalizer.floor)
    i_fit.fit(i.unsqueeze(1))
    idx = ds_mod.current_channel_index(cfg)
    model.set_current_channel_index(idx)
    model.normalizer.median[idx] = i_fit.median[0]
    model.normalizer.scale[idx] = i_fit.scale[0]

    print(
        f"[train] 标准化统计量已拟合（仅用训练集）："
        f"输入 {tuple(x.shape)}，电压 {tuple(v.shape)}，电流通道 idx={idx}"
    )
    del xs, vs, iis, x, v, i


def build_model(cfg: dict) -> CellResolvedAE:
    """按配置建模型。模型规模与输入通道数由配置推导，避免手写常量错位。"""
    mcfg = cfg["model"]
    n_cells = len(channels.cell_voltage_cols(cfg))
    in_ch = len(channels.input_cols(cfg))
    return CellResolvedAE(
        n_cells=n_cells,
        in_channels=in_ch,
        cell_latent=int(mcfg["cell_latent"]),
        pack_latent=int(mcfg["pack_latent"]),
        t_len=int(cfg["data"]["expected_rows"]),
        cell_conv=list(mcfg["cell_conv"]),
        pack_conv=list(mcfg["pack_conv"]),
        decoder_hidden=int(mcfg["decoder_hidden"]),
        cell_resolved=bool(mcfg["cell_resolved"]),
        variational=bool(mcfg["variational"]),
        use_sibling_context=bool(mcfg["use_sibling_context"]),
        decoder_current_skip=bool(mcfg.get("decoder_current_skip", False)),
        norm_floor=float(mcfg["norm_floor"]),
    )


def fit_target_scaler(cache: dict, train_idx: np.ndarray, cfg: dict) -> TargetScaler:
    """在训练集窗口上拟合物理目标（单体内阻 + SOC）的稳健标准化。

    仅用训练集 —— 用上测试集统计量就等于把测试集信息泄漏进训练。
    """
    hidden = cache.get("hidden")
    n_cells = len(channels.cell_voltage_cols(cfg))
    if hidden is None:
        nan = torch.full((n_cells,), float("nan"))
        return TargetScaler(nan, torch.ones(n_cells), torch.tensor(float("nan")), torch.tensor(1.0))

    hcfg = cfg["hidden"]
    res_cols = [c - 1 for c in hcfg["resistance_channels"]]
    h = np.asarray(hidden)[train_idx]
    res = torch.from_numpy(np.ascontiguousarray(h[:, res_cols], dtype=np.float64)).float()
    soc = torch.from_numpy(np.ascontiguousarray(h[:, hcfg["soc_channel"] - 1], dtype=np.float64)).float()
    scaler = TargetScaler.fit(res, soc, floor=float(cfg["model"]["norm_floor"]))
    print(
        f"[train] 物理目标标准化已拟合（仅训练集）："
        f"内阻 median={scaler.res_median.numpy().round(4).tolist()} "
        f"scale={scaler.res_scale.numpy().round(4).tolist()} "
        f"SOC median={float(scaler.soc_median):.4f} scale={float(scaler.soc_scale):.4f}"
    )
    return scaler


def prepare(
    cfg: dict,
    cache: dict,
    *,
    out_dir: Path,
    tag: str = "run",
    fit_norm: bool = True,
) -> TrainBundle:
    """建模型 / 切分 / 拟合标准化统计量，返回可训练的 bundle。"""
    train_cfg = cfg["train"]
    seed_everything(int(train_cfg["seed"]))
    device = pick_device(cfg)

    split = ds_mod.chronological_split(
        np.asarray(cache["ids"]),
        cache.get("time"),
        float(cfg["split"]["val_ratio"]),
    )
    train_ds = ds_mod.WindowDataset(cfg, cache, split.train_idx)
    val_ds = ds_mod.WindowDataset(cfg, cache, split.val_idx)

    print(f"[train] 切分（按 ID 的时间顺序，val 取时间尾部）：")
    for row in split.summary():
        print(f"    ID={row['id']:>3}  共 {row['n']:>6}  训练 {row['n_train']:>6}  验证 {row['n_val']:>5}")

    model = build_model(cfg).to(device)
    print(f"[train] 模型：{type(model).__name__}，可训练参数 {model.param_count():,}  设备 {device}")

    bundle_dir = Path(out_dir) / tag
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if fit_norm:
        loader = DataLoader(
            train_ds, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=0
        )
        _fit_normalizers(model, loader, cfg, device)

    scaler = fit_target_scaler(cache, split.train_idx, cfg)
    return TrainBundle(cfg, model, train_ds, val_ds, split, scaler, device, bundle_dir)


def _run_epoch(
    model: CellResolvedAE,
    loader: DataLoader,
    scaler: TargetScaler,
    device: torch.device,
    cfg: dict,
    optimizer: torch.optim.Optimizer | None,
    grad_scaler: Any,
    *,
    desc: str = "",
) -> dict[str, float]:
    """跑一个 epoch。``optimizer`` 为 None 时是评估模式。"""
    tr = cfg["train"]
    training = optimizer is not None
    model.train(training)

    agg: dict[str, float] = {}
    n_batch = 0

    it = progress.bar(loader, desc=desc, leave=False) if desc else loader
    for batch in it:
        x = batch["x"].to(device, non_blocking=True)
        v = batch["v"].to(device, non_blocking=True)
        i = batch["i"].to(device, non_blocking=True)
        res = batch["res"].to(device, non_blocking=True)
        soc = batch["soc"].to(device, non_blocking=True)

        res_norm = scaler.norm_res(res)
        soc_norm = scaler.norm_soc(soc)

        with torch.set_grad_enabled(training):
            with torch.autocast("cuda", enabled=bool(tr["amp"]) and device.type == "cuda"):
                _, out = model(x, v, i, return_output=True)
                v_norm = model.cell_norm(v)
                loss, stats = total_loss(
                    v_hat=out.v_hat,
                    v_norm=v_norm,
                    a_pred=out.a_cell,
                    res_target=res_norm,
                    soc_pred=out.soc_pack,
                    soc_target=soc_norm,
                    mu_cell=out.mu_cell,
                    logvar_cell=out.logvar_cell,
                    mu_pack=out.mu_pack,
                    logvar_pack=out.logvar_pack,
                    lambda1=float(tr["lambda1"]),
                    lambda2=float(tr["lambda2"]),
                    lambda3=float(tr["lambda3"]),
                )

        if training:
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr["grad_clip"]))
            grad_scaler.step(optimizer)
            grad_scaler.update()

        for k, val in stats.items():
            agg[k] = agg.get(k, 0.0) + val
        n_batch += 1
        if desc:
            it.set_postfix(
                {  # type: ignore[attr-defined]
                    "loss": f"{stats['loss']:.4f}",
                    "rec": f"{stats['recon']:.4f}",
                }
            )

    if desc:
        it.close()  # type: ignore[attr-defined]
    return {k: v / max(n_batch, 1) for k, v in agg.items()}


def train(bundle: TrainBundle, *, epochs: int | None = None, verbose: bool = True) -> dict:
    """执行训练，返回 history（含最优 epoch 与最优验证损失）。"""
    cfg, model, device = bundle.cfg, bundle.model, bundle.device
    tr = cfg["train"]
    epochs = int(epochs if epochs is not None else tr["max_epochs"])

    pin = device.type == "cuda"
    train_loader = DataLoader(
        bundle.train_ds,
        batch_size=int(tr["batch_size"]),
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        bundle.val_ds,
        batch_size=int(tr["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    grad_scaler = torch.amp.GradScaler("cuda", enabled=bool(tr["amp"]) and device.type == "cuda")

    log_path = bundle.out_dir / "history.csv"
    best_val = float("inf")
    best_epoch = -1
    bad = 0
    history: list[dict] = []
    t0 = time.time()

    epoch_bar = progress.bar(range(1, epochs + 1), desc=f"训练 {bundle.out_dir.name}", unit="ep")
    for ep in epoch_bar:
        tr_stats = _run_epoch(
            model, train_loader, bundle.target_scaler, device, cfg, optimizer, grad_scaler,
            desc=f"  ep{ep} train",
        )
        va_stats = _run_epoch(
            model, val_loader, bundle.target_scaler, device, cfg, None, grad_scaler,
            desc=f"  ep{ep} val  ",
        )
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(va_stats["loss"])

        row = {"epoch": ep, "lr": lr_now}
        row.update({f"train_{k}": v for k, v in tr_stats.items()})
        row.update({f"val_{k}": v for k, v in va_stats.items()})
        history.append(row)

        improved = va_stats["loss"] < best_val - 1e-6
        if improved:
            best_val, best_epoch, bad = va_stats["loss"], ep, 0
            _save_checkpoint(bundle, ep, best_val, history, is_best=True)
        else:
            bad += 1

        epoch_bar.set_postfix(  # type: ignore[attr-defined]
            {
                "tr": f"{tr_stats['loss']:.4f}",
                "va": f"{va_stats['loss']:.4f}",
                "rec": f"{va_stats['recon']:.4f}",
                "best": f"{best_val:.4f}@{best_epoch}",
                "bad": f"{bad}/{int(tr['patience'])}",
                "lr": f"{lr_now:.1e}",
            }
        )
        if verbose and (ep == 1 or ep % 5 == 0 or improved):
            progress.log(
                f"  ep{ep:>4}  train {tr_stats['loss']:.5f} "
                f"(rec {tr_stats['recon']:.5f} late {tr_stats['late']:.5f} reg {tr_stats['reg']:.5f})  "
                f"val {va_stats['loss']:.5f} (rec {va_stats['recon']:.5f})  "
                f"lr {lr_now:.2e}{'  *best*' if improved else ''}"
            )

        if bad >= int(tr["patience"]):
            epoch_bar.close()  # type: ignore[attr-defined]
            progress.log(f"  验证损失连续 {bad} 个 epoch 未改善，早停于 ep{ep}")
            break
    else:
        epoch_bar.close()  # type: ignore[attr-defined]

    elapsed = time.time() - t0
    with log_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    _save_checkpoint(bundle, best_epoch, best_val, history, is_best=False)
    print(f"[train] 完成：{len(history)} 个 epoch，{elapsed:.1f}s，最优 val loss={best_val:.6f} @ ep{best_epoch}")
    print(f"[train] 日志 {log_path}，checkpoint {bundle.out_dir}")
    return {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "elapsed_sec": elapsed,
        "history": history,
    }


def _save_checkpoint(bundle: TrainBundle, epoch: int, val_loss: float, history: list[dict], *, is_best: bool) -> None:
    payload = {
        "model_state": bundle.model.state_dict(),
        "target_scaler": bundle.target_scaler.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "cfg_snapshot": json.loads(json.dumps(bundle.cfg, default=str)),
        "split_per_id": bundle.split.per_id,
        "history": history,
    }
    name = "best.pt" if is_best else "last.pt"
    torch.save(payload, bundle.out_dir / name)


def load_checkpoint(path: str | Path, cfg: dict) -> tuple[CellResolvedAE, TargetScaler, dict]:
    """读回 checkpoint（模型结构优先按 checkpoint 自带的 ``cfg_snapshot`` 重建）。

    ⚠ 两处必须显式处理，都是实测踩过的坑：

    1. **架构取自 ``cfg_snapshot``，不是传入的 ``cfg``。**
       消融覆盖（如 ``cell_resolved=False``）只写在 checkpoint 里；若按
       ``base.yaml`` 建模，``shared_latent`` 这类 tag 会在 ``load_state_dict``
       处形状不匹配。让 checkpoint 自带架构，读取方就不必知道它当初怎么配的。
       ``cfg_snapshot`` 缺失时退回传入的 ``cfg``。

    2. **必须补 ``set_current_channel_index``。**
       ``_cur_idx`` 是普通属性、**不进 ``state_dict``**，训练时由 :func:`prepare`
       设置。漏了它，``forward`` 里的 ``_normalize_current`` 会直接抛
       RuntimeError —— 这条推理通路因此一直没跑通过。
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    run_cfg = ckpt.get("cfg_snapshot") or cfg
    model = build_model(run_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.set_current_channel_index(ds_mod.current_channel_index(run_cfg))
    scaler = TargetScaler.from_state(ckpt["target_scaler"])
    return model, scaler, ckpt


def overfit_single_batch(
    cfg: dict, bundle: TrainBundle, *, steps: int = 300, batch_size: int = 8
) -> dict:
    """单批过拟合自检：固定一个小批次反复训练，重建损失应趋近 0。

    训练通路的「冒烟测试」：连一个小批次都压不下去，说明模型、损失或梯度回传
    有结构性错误，此时跑全量训练只会浪费时间。
    """
    loader = DataLoader(bundle.train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    device = bundle.device
    model = bundle.model
    tr = cfg["train"]

    # 冒烟测试只在单批上拟合标准化统计量 —— 正式训练绝不允许这样做（只用训练集）。
    model.normalizer.fit(batch["x"])
    model.cell_norm.fit(batch["v"])
    model.set_current_channel_index(ds_mod.current_channel_index(cfg))

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    model.train()

    res_norm = bundle.target_scaler.norm_res(batch["res"].to(device))
    soc_norm = bundle.target_scaler.norm_soc(batch["soc"].to(device))
    x = batch["x"].to(device)
    v = batch["v"].to(device)
    i = batch["i"].to(device)

    first = last = None
    for s in range(steps):
        _, out = model(x, v, i, return_output=True)
        loss, stats = total_loss(
            v_hat=out.v_hat,
            v_norm=model.cell_norm(v),
            a_pred=out.a_cell,
            res_target=res_norm,
            soc_pred=out.soc_pack,
            soc_target=soc_norm,
            mu_cell=out.mu_cell,
            logvar_cell=out.logvar_cell,
            mu_pack=out.mu_pack,
            logvar_pack=out.logvar_pack,
            lambda1=float(tr["lambda1"]),
            lambda2=float(tr["lambda2"]),
            lambda3=float(tr["lambda3"]),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr["grad_clip"]))
        opt.step()
        sched.step()
        if s == 0:
            first = dict(stats)
        last = dict(stats)

    ratio = last["recon"] / max(first["recon"], 1e-12)
    print(
        f"[overfit] recon {first['recon']:.6f} -> {last['recon']:.6f}"
        f"（降至 {ratio * 100:.2f}%）  late {first['late']:.4f} -> {last['late']:.4f}"
    )
    return {"first": first, "last": last, "recon_ratio": ratio}
