# Streaming replay demo: fit on a normal reference, score a held-out sequence in
# causal chunks, then animate the result window by window.
#
# This is a replay of frozen inference, not live training, and it is not a
# performance claim. The scored machine comes from the public SMD dataset, which
# carries point labels, so the animation can show hits and misses honestly.
import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chronoguard import AnomalyDetector                                    # noqa: E402
from chronoguard.data import load_smd, make_windows, split_train           # noqa: E402
from chronoguard.evaluation import detection_metrics                       # noqa: E402

C_SIGNAL = "#0072B2"
C_THRESH = "#D55E00"
C_OK = "#009E73"

FFMPEG_HINTS = (
    "D:/Python/miniconda3/envs/manim/Library/bin/ffmpeg.exe",
    "C:/Users/{user}/AppData/Local/ms-playwright/ffmpeg-1011/ffmpeg-win64.exe",
)


def find_ffmpeg(explicit=None):
    '''PATH first, then a couple of known local installs; otherwise None.'''
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    found = shutil.which("ffmpeg")
    if found:
        return Path(found)
    for hint in FFMPEG_HINTS:
        candidate = Path(hint.format(user=os.environ.get("USERNAME", "")))
        if candidate.is_file():
            return candidate
    return None


def predict_in_chunks(model, windows, chunk=512):
    '''Causal streaming inference; state is carried across chunk boundaries.'''
    scores, confirmed, evidence = [], [], []
    state = None
    for start in range(0, len(windows), chunk):
        out = model.predict(windows[start:start + chunk], device_id="demo", state=state)
        scores.append(np.asarray(out.scores, float))
        confirmed.append(np.asarray(out.confirmed, bool))
        evidence.append(np.asarray(out.channel_evidence, float))
        state = out.state
    return np.concatenate(scores), np.concatenate(confirmed), np.concatenate(evidence)


def build(baseline, scores, confirmed, labels, evidence, machine, threshold, args, out):
    n = len(scores)
    highlight = float(np.percentile(evidence, 99.5)) or 1.0

    fig = plt.figure(figsize=(11.0, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.30, 1.05, 0.85])
    ax_heat, ax_score, ax_chan = (fig.add_subplot(gs[i]) for i in range(3))

    im = ax_heat.imshow(evidence.T, aspect="auto", origin="lower", cmap="magma",
                        vmin=0, vmax=highlight, interpolation="nearest")
    ax_heat.set_ylabel("通道")
    ax_heat.set_yticks(np.linspace(0, evidence.shape[1] - 1, 8, dtype=int))
    ax_heat.set_title(
        f"逐通道标准化偏离　｜　{machine}　｜　{args.method.upper()}　｜　"
        f"窗口长度 T={args.window_size}　｜　分块流式推理回放",
        loc="left", fontweight="bold", fontsize=10)
    fig.colorbar(im, ax=ax_heat, pad=0.008, fraction=0.022, label="偏离")

    x = np.arange(n)
    for start, stop in _runs(labels):
        ax_score.axvspan(start, stop, color="#CC6677", alpha=0.28, lw=0)
    ax_score.plot(x, scores, color=C_SIGNAL, lw=0.5, rasterized=True, label="异常分数")
    ax_score.axhline(threshold, color=C_THRESH, lw=1.1, ls="--",
                     label=f"阈值 = {threshold:.3g}")
    ax_score.set_yscale("log")
    ax_score.set_ylabel("异常分数")
    ax_score.legend(loc="upper left", fontsize=7, ncol=2)
    ax_score.text(0.995, 0.94, "红色背景 = 数据集中标注的真实异常区间",
                  transform=ax_score.transAxes, ha="right", va="top",
                  fontsize=7, color="#555555")
    cursor = ax_score.axvline(0, color="#222222", lw=1.0, alpha=0.9)
    ax_score.set_xlim(-n * 0.004, n * 1.004)

    bars = ax_chan.barh(np.arange(8), np.zeros(8), color=C_SIGNAL, height=0.72)
    ax_chan.set_ylim(7.6, -0.6)
    ax_chan.set_yticks(np.arange(8))
    ax_chan.set_xlim(0, highlight)
    ax_chan.set_xlabel("当前窗口的通道证据（前 8 位，越靠上越可疑）")
    ax_chan.set_yticklabels(["" for _ in range(8)], fontsize=7)

    status = ax_score.text(0.012, 0.05, "", transform=ax_score.transAxes,
                           va="bottom", fontsize=9.5, fontweight="bold",
                           bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                                     edgecolor="#CCCCCC", linewidth=0.6, alpha=0.92))

    def update(frame):
        i = min(frame * args.step, n - 1)
        cursor.set_xdata([i, i])
        order = np.argsort(-evidence[i])[:8]
        for bar, channel in zip(bars, order):
            value = float(evidence[i, channel])
            bar.set_width(value)
            bar.set_color(C_THRESH if value >= highlight * 0.6 else C_SIGNAL)
        ax_chan.set_yticklabels([f"ch {int(c)}" for c in order], fontsize=7)
        if scores[i] > threshold:
            state, color = ("已确认报警" if confirmed[i] else "超阈值"), C_THRESH
        else:
            state, color = "正常", C_OK
        if labels[i]:
            state += "　｜　真实异常"
        status.set_text(f"窗口 {i:>6}　　分数 {scores[i]:8.3g}　　{state}")
        status.set_color(color)
        return [cursor, status, *bars]

    frames = (n + args.step - 1) // args.step
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)
    meta = {"title": "streaming anomaly detection replay", "artist": "chronoguard",
            "comment": "replay of frozen causal inference; not a performance claim"}
    if args.format == "mp4":
        writer = FFMpegWriter(fps=args.fps, bitrate=args.bitrate, metadata=meta,
                              extra_args=["-pix_fmt", "yuv420p"])
    else:
        writer = PillowWriter(fps=args.fps, metadata=meta)
    anim.save(out, writer=writer, dpi=args.dpi)
    plt.close(fig)
    return frames, baseline


def _runs(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smd-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "datasets/smd")
    parser.add_argument("--machine", default="machine-1-1")
    parser.add_argument("--method", default="lof", choices=("robust", "lof", "iforest", "pca"))
    parser.add_argument("--window-size", type=int, default=1,
                        help="T; 1 is the point baseline, larger values build time-series features")
    parser.add_argument("--start", type=int, default=19000, help="first test point to replay")
    parser.add_argument("--stop", type=int, default=21300, help="one past the last test point")
    parser.add_argument("--step", type=int, default=5, help="test points advanced per frame")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--bitrate", type=int, default=6000)
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--ffmpeg", default=None, help="path to an ffmpeg binary")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    ffmpeg = find_ffmpeg(args.ffmpeg)
    if args.format == "mp4" and ffmpeg is None:
        parser.error("no ffmpeg found; pass --ffmpeg PATH, or use --format gif")
    if ffmpeg is not None:
        matplotlib.rcParams["animation.ffmpeg_path"] = str(ffmpeg)

    matplotlib.rcParams.update({
        "font.family": ["Microsoft YaHei", "Arial", "sans-serif"],
        "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans", "sans-serif"],
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False,
    })

    output = args.output or Path(f"live_demo_{args.machine}_{args.method}.{args.format}")

    train, test, labels = load_smd(args.smd_dir, args.machine)
    fit, cal, _ = split_train(train)
    T = args.window_size
    print(f"拟合 {len(fit)} 行　校准 {len(cal)} 行　测试 {len(test)} 行　窗口长度 T={T}")

    model = AnomalyDetector(args.method, feature_mode="independent",
                            quantile=.99, persistence=5, random_state=42)
    model.fit(make_windows(np.asarray(fit), T, 1)[0]).calibrate(make_windows(np.asarray(cal), T, 1)[0])

    windows, ends = make_windows(np.asarray(test), T, 1)
    scores, confirmed, evidence = predict_in_chunks(model, windows)
    # Window ends align each output back to a raw test point, so labels can follow.
    labels = labels[ends]
    print(f"阈值 {model.threshold_:.6g}　确认报警 {int(confirmed.sum())} / {len(confirmed)}")
    print(f"预热区 {T - 1} 点无输出；评价覆盖原始点 {ends[0]}–{ends[-1]}")

    lo, hi = args.start, min(args.stop, len(scores))
    block_scores, block_confirmed = scores[lo:hi], confirmed[lo:hi]
    block_labels, block_evidence = labels[lo:hi], evidence[lo:hi]
    metrics = detection_metrics(block_labels, block_confirmed)
    print(f"片段严格逐点：P={metrics['precision']:.3f}　R={metrics['recall']:.3f}　"
          f"F1={metrics['f1']:.3f}　正常点FPR={metrics['normal_false_positive_rate']:.2%}")
    print(f"片段真实异常点 {int(block_labels.sum())} / {len(block_labels)}")

    frames, _ = build(None, block_scores, block_confirmed, block_labels, block_evidence,
                      args.machine, model.threshold_, args, output)
    print(f"已写出 {output}")
    print(f"  {frames} 帧　{frames / args.fps:.1f} 秒　"
          f"{output.stat().st_size / 1024 / 1024:.1f} MB")
    print("注意：这是冻结因果推理的回放，不是性能证据。")


if __name__ == "__main__":
    main()
