"""批量重建工具：把整个缓存的窗口过一遍模型，取回实测与重建电压。

基线（论文方法）与我们的模型 forward 签名不同（后者需要额外的电流通道），但下游
的故障指标、检测、出图都要的是同一对 ``(实测电压, 重建电压)``，故用一处 dispatch
收口，避免每个脚本各写一遍分批逻辑。

分批是为了控内存：训练集 26508 个窗口，(N, 8, 256) float32 约 217 MB，一次前向
全放 GPU 会撑爆 8 GB 显存，所以按 batch 走并在 CPU 上累积结果。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import progress
from ..data import channels, dataset as ds_mod


def _reconstruct_batch(model, batch: dict, device: torch.device) -> torch.Tensor:
    """按模型类型分派一次前向，返回标准化空间的重建电压 (B, n_cells, T)。

    分派依据是模型类上的 ``requires_current`` 标记，**不是**属性探测 ——
    LFAAE 同样有 ``cell_norm``，用 ``hasattr`` 猜会走错分支（实测踩过）。
    """
    x = batch["x"].to(device, non_blocking=True)
    if getattr(model, "requires_current", False):
        v = batch["v"].to(device, non_blocking=True)
        i = batch["i"].to(device, non_blocking=True)
        return model(x, v, i)
    return model(x)


@torch.no_grad()
def reconstruct_all(
    model,
    cfg: dict,
    cache: dict,
    *,
    indices: np.ndarray | None = None,
    batch_size: int = 512,
    device: torch.device | str = "cpu",
    denormalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对整个缓存跑重建。

    Returns
    -------
    v_meas, v_rec : (N, n_cells, T) float32
        实测与重建电压。``denormalize=True`` 时已还原回**缓存的原始缩放值**
        （即与 ``signal`` 同一量纲），故障指标可直接用。
    ids : (N,) int16
        每个窗口的包 ID，便于按包切分与排序。
    """
    device = torch.device(device)
    model = model.to(device).eval()

    full = np.arange(int(cache["signal"].shape[0])) if indices is None else np.asarray(indices)
    ds = ds_mod.WindowDataset(cfg, cache, full)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    vcols = channels.cell_voltage_cols(cfg)
    n_cells = len(vcols)

    out_rec = np.empty((len(ds), n_cells, cfg["data"]["expected_rows"]), dtype=np.float32)
    out_meas = np.empty_like(out_rec)
    out_ids = np.empty(len(ds), dtype=np.int16)

    pos = 0
    it = progress.bar(loader, desc=f"  重建 {cache.get('name', '')}", leave=False)
    for batch in it:
        v_hat = _reconstruct_batch(model, batch, device)
        if denormalize:
            v_hat = model.cell_norm.inverse(v_hat)
        v_hat = v_hat.detach().float().cpu().numpy()
        n = v_hat.shape[0]
        out_rec[pos : pos + n] = v_hat
        out_meas[pos : pos + n] = batch["v"].numpy()
        out_ids[pos : pos + n] = batch["id"].numpy()
        pos += n

    it.close()
    if pos != len(ds):
        raise RuntimeError(f"分批写入数量不符：{pos} != {len(ds)}")
    return out_meas, out_rec, out_ids


def checkpoint_path(cfg: dict, tag: str, *, which: str = "best") -> Path:
    return Path(cfg["paths"]["outputs_dir"]) / "runs" / tag / f"{which}.pt"


def reconstruction_dir(cfg: dict, tag: str) -> Path:
    return Path(cfg["paths"]["outputs_dir"]) / "runs" / tag / "recon"


def reconstruct_dataset(
    model,
    cfg: dict,
    cache: dict,
    tag: str,
    dataset_name: str,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 512,
    force: bool = False,
    mmap: bool = True,
):
    """带磁盘缓存的重建。

    同一份数据的重建会被反复用到（故障指标、多种阈值、出图），而前向一遍训练集
    要跑几万个窗口，故缓存；后续步骤也不必持有模型。

    缓存目录：``outputs/runs/<tag>/recon/<dataset_name>/``
    """
    import gc

    d = reconstruction_dir(cfg, tag) / dataset_name
    f_meas, f_rec, f_ids = d / "v_meas.npy", d / "v_rec.npy", d / "ids.npy"

    if not force and f_meas.exists() and f_rec.exists() and f_ids.exists():
        mode = "r" if mmap else None
        print(f"[recon] 命中缓存 {d}")
        return (
            np.load(f_meas, mmap_mode=mode),
            np.load(f_rec, mmap_mode=mode),
            np.load(f_ids, mmap_mode=mode),
        )

    v_meas, v_rec, ids = reconstruct_all(
        model, cfg, cache, batch_size=batch_size, device=device
    )
    d.mkdir(parents=True, exist_ok=True)
    np.save(f_meas, v_meas)
    np.save(f_rec, v_rec)
    np.save(f_ids, ids)
    print(f"[recon] 已缓存 {d}（{v_meas.shape}）")

    del v_meas, v_rec
    gc.collect()
    mode = "r" if mmap else None
    return (
        np.load(f_meas, mmap_mode=mode),
        np.load(f_rec, mmap_mode=mode),
        np.load(f_ids, mmap_mode=mode),
    )
