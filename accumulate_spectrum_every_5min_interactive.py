#!/usr/bin/env python3
"""每5分钟能谱累加与交互式查看（Python版）

功能对应 MATLAB 逻辑：
1. 选择原始能谱文件夹
2. 筛选 CH1 文件并按文件名时间排序
3. 双峰（氢峰+铁峰）对齐
4. 氢峰后固定偏移开始放大
5. 手动选择基准区间并取中位数基准谱
6. 相对偏差 + 连续异常道检测
7. 交互式滑块查看与异常区段标注
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


@dataclass
class Config:
    file_feature: str = "CH1"
    output_folder_name: str = "每5分钟累加结果"

    align_spectra: bool = True
    smoothing_window: int = 11

    relative_threshold: float = 0.10
    min_contiguous_channels: int = 10

    group_size: int = 5
    amplification_factor: float = 30.0
    amp_start_offset: int = 50

    iron_range_default: Tuple[int, int] = (2450, 2550)
    hydrogen_range_default: Tuple[int, int] = (1450, 1550)

    ignore_before_hydrogen_peak: bool = True
    hydrogen_peak_ignore_offset: int = 0
    ignore_after_iron_peak: bool = True
    iron_peak_cutoff_offset: int = 50

    channel_count: int = 4096


TIME_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})\s+(\d{2}-\d{2}-\d{2})")


def smooth(y: np.ndarray, win: int) -> np.ndarray:
    win = max(3, int(win) | 1)
    if y.size < win:
        return y
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(y, kernel, mode="same")


def extract_time_from_name(fname: str, fallback_idx: int) -> tuple[datetime, str, str]:
    m = TIME_PATTERN.search(fname)
    if not m:
        dt = datetime.fromtimestamp(fallback_idx)
        return dt, "未知日期", "未知时间"
    d, t = m.group(1), m.group(2)
    dt = datetime.strptime(f"{d} {t.replace('-', ':')}", "%Y-%m-%d %H:%M:%S")
    return dt, d, t


def read_spectrum(path: Path, n: int) -> np.ndarray:
    arr = np.loadtxt(path, dtype=float)
    arr = np.ravel(arr)
    if arr.size != n:
        raise ValueError(f"{path.name} 数据长度={arr.size}，预期={n}")
    return arr


def find_peak_smoothed(spectrum: np.ndarray, start_ch: int, end_ch: int, smooth_window: int) -> tuple[int, float, float]:
    start = max(1, start_ch)
    end = min(len(spectrum), end_ch)
    if start >= end:
        val = float(spectrum[start - 1])
        return start, val, 0.0

    seg = spectrum[start - 1:end]
    sm = smooth(seg, smooth_window)
    idx0 = int(np.argmax(sm))
    peak_val = float(sm[idx0])
    peak_pos = start + idx0

    left = max(0, idx0 - 10)
    right = min(sm.size, idx0 + 11)
    bg = np.concatenate([sm[left:max(left, idx0 - 3)], sm[min(right, idx0 + 3):right]])
    noise = np.std(bg) if bg.size > 0 else 0.0
    snr = peak_val / noise if noise > 1e-12 else math.inf
    return peak_pos, peak_val, float(snr)


def apply_shift_only(spectrum: np.ndarray, shift: int) -> np.ndarray:
    out = np.zeros_like(spectrum)
    n = spectrum.size
    if shift > 0:
        out[shift:] = spectrum[: n - shift]
    elif shift < 0:
        out[: n + shift] = spectrum[-shift:]
    else:
        out[:] = spectrum
    return out


def apply_scale_shift(spectrum: np.ndarray, scale: float, shift: int) -> np.ndarray:
    n = spectrum.size
    x_target = np.arange(1, n + 1, dtype=float)
    x_source = (x_target - shift) / scale
    x_source = np.clip(x_source, 1.0, n)
    return np.interp(x_source, np.arange(1, n + 1, dtype=float), spectrum)


def detect_segments(mask: np.ndarray, min_length: int) -> list[np.ndarray]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    segs: list[np.ndarray] = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        seg = np.arange(start, prev + 1)
        if seg.size >= min_length:
            segs.append(seg)
        start = prev = i
    seg = np.arange(start, prev + 1)
    if seg.size >= min_length:
        segs.append(seg)
    return segs


def choose_baseline_range(sorted_files: Sequence[str], dates: Sequence[str], times: Sequence[str]) -> tuple[int, int]:
    root = tk.Tk()
    root.title("选择基准时间区间")
    root.geometry("1000x700")

    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    tk.Label(frame, text="按时间排序文件（可多选）", font=("Arial", 12, "bold")).pack(anchor="w")

    listbox = tk.Listbox(frame, selectmode=tk.EXTENDED, width=140, height=28)
    scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=listbox.yview)
    listbox.configure(yscrollcommand=scrollbar.set)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    for i, (f, d, t) in enumerate(zip(sorted_files, dates, times), start=1):
        listbox.insert(tk.END, f"{i:4d} | {d} {t} | {f}")

    result = {"ok": False, "start": 1, "end": len(sorted_files)}

    def confirm() -> None:
        sel = listbox.curselection()
        if not sel:
            messagebox.showwarning("提示", "请至少选择一个文件")
            return
        result["start"] = min(sel) + 1
        result["end"] = max(sel) + 1
        result["ok"] = True
        root.destroy()

    def cancel() -> None:
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.pack(fill=tk.X, pady=8)
    tk.Button(btn_frame, text="确认", command=confirm, width=12).pack(side=tk.LEFT, padx=6)
    tk.Button(btn_frame, text="取消（使用全部）", command=cancel, width=16).pack(side=tk.LEFT, padx=6)

    root.mainloop()
    return result["start"], result["end"]


def select_peak_ranges(first_spectrum: np.ndarray, cfg: Config) -> tuple[tuple[int, int], tuple[int, int]]:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(np.arange(1, cfg.channel_count + 1), first_spectrum, lw=1)
    ax.set_title("第一个能谱：点击查看坐标；关闭窗口后输入峰位范围")
    ax.set_xlabel("道址")
    ax.set_ylabel("计数")
    ax.grid(True, alpha=0.3)

    ann = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                      bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ann.set_visible(False)

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), float(event.ydata)
        ann.xy = (x, y)
        ann.set_text(f"({x}, {y:.0f})")
        ann.set_visible(True)
        print(f"点击: 道址={x}, 计数={y:.0f}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show(block=True)

    root = tk.Tk()
    root.withdraw()

    h_start = simpledialog.askinteger("氢峰区间", "氢峰起始道址", initialvalue=cfg.hydrogen_range_default[0])
    h_end = simpledialog.askinteger("氢峰区间", "氢峰结束道址", initialvalue=cfg.hydrogen_range_default[1])
    fe_start = simpledialog.askinteger("铁峰区间", "铁峰起始道址", initialvalue=cfg.iron_range_default[0])
    fe_end = simpledialog.askinteger("铁峰区间", "铁峰结束道址", initialvalue=cfg.iron_range_default[1])

    root.destroy()

    if None in (h_start, h_end, fe_start, fe_end) or h_start >= h_end or fe_start >= fe_end:
        return cfg.hydrogen_range_default, cfg.iron_range_default

    return (h_start, h_end), (fe_start, fe_end)


def align_all_spectra(raw_spectra: np.ndarray, hydrogen_range: tuple[int, int], iron_range: tuple[int, int], cfg: Config):
    n_files = raw_spectra.shape[1]
    aligned = np.zeros_like(raw_spectra)
    hydrogen_peaks = np.zeros(n_files, dtype=int)

    ref = raw_spectra[:, 0]
    h_ref, _, _ = find_peak_smoothed(ref, hydrogen_range[0], hydrogen_range[1], cfg.smoothing_window)
    fe_ref, _, _ = find_peak_smoothed(ref, iron_range[0], iron_range[1], cfg.smoothing_window)
    ref_dist = fe_ref - h_ref

    aligned[:, 0] = ref
    hydrogen_peaks[0] = h_ref

    for i in range(1, n_files):
        s = raw_spectra[:, i]
        h, _, h_snr = find_peak_smoothed(s, hydrogen_range[0], hydrogen_range[1], cfg.smoothing_window)
        fe, _, fe_snr = find_peak_smoothed(s, iron_range[0], iron_range[1], cfg.smoothing_window)
        hydrogen_peaks[i] = h

        quality = (h_snr < 2) + 2 * (fe_snr < 2)

        if quality == 0 and (fe - h) != 0:
            scale = ref_dist / (fe - h)
            if not (0.95 <= scale <= 1.05):
                scale = 1.0
            shift = int(round(h_ref - h * scale))
            if abs(shift) > 50:
                shift = 0
            a = apply_scale_shift(s, scale, shift)
        elif quality == 1:
            shift = fe - fe_ref
            a = apply_shift_only(s, shift if abs(shift) <= 50 else 0)
        elif quality == 2:
            shift = h - h_ref
            a = apply_shift_only(s, shift if abs(shift) <= 50 else 0)
        else:
            a = s
        aligned[:, i] = a

    return aligned, hydrogen_peaks, h_ref, fe_ref


def process_groups(aligned: np.ndarray, hydrogen_peaks_file: np.ndarray, cfg: Config):
    n_files = aligned.shape[1]
    n_groups = math.ceil(n_files / cfg.group_size)

    grouped = np.zeros((cfg.channel_count, n_groups), dtype=float)
    group_counts = np.zeros(n_groups, dtype=float)
    group_h_peaks = np.zeros(n_groups, dtype=int)

    for g in range(n_groups):
        st = g * cfg.group_size
        ed = min((g + 1) * cfg.group_size, n_files)
        summ = np.sum(aligned[:, st:ed], axis=1)
        total = float(np.sum(summ))
        group_counts[g] = total
        if total <= 0:
            continue
        norm = summ / total
        h_peak = int(round(np.mean(hydrogen_peaks_file[st:ed]))) if hydrogen_peaks_file.size else int(np.argmax(norm) + 1)
        group_h_peaks[g] = h_peak
        amp_start = min(cfg.channel_count, h_peak + cfg.amp_start_offset)
        norm[amp_start - 1:] *= cfg.amplification_factor
        grouped[:, g] = norm
    return grouped, group_counts, group_h_peaks


def detect_distortions(grouped: np.ndarray, baseline: np.ndarray, group_h: np.ndarray, iron_peak: int, cfg: Config):
    n_groups = grouped.shape[1]
    diffs = np.zeros_like(grouped)
    max_diffs = np.zeros(n_groups)
    mean_diffs = np.zeros(n_groups)
    anomalies: list[list[np.ndarray]] = []

    cutoff = cfg.channel_count
    if cfg.ignore_after_iron_peak and iron_peak > 0:
        cutoff = min(cfg.channel_count, iron_peak + cfg.iron_peak_cutoff_offset)

    baseline_nz = baseline.copy()
    baseline_nz[baseline_nz == 0] = 1e-10

    for g in range(n_groups):
        r = np.abs(grouped[:, g] - baseline) / baseline_nz
        start = 1
        if cfg.ignore_before_hydrogen_peak and group_h[g] > 0:
            start = max(1, group_h[g] - cfg.hydrogen_peak_ignore_offset)
            r[: start - 1] = 0
        if cutoff < cfg.channel_count:
            r[cutoff:] = 0

        diffs[:, g] = r
        valid = r[start - 1:cutoff]
        max_diffs[g] = float(np.max(valid)) if valid.size else 0.0
        mean_diffs[g] = float(np.mean(valid)) if valid.size else 0.0

        mask = np.zeros(cfg.channel_count, dtype=bool)
        mask[start - 1:cutoff] = r[start - 1:cutoff] > cfg.relative_threshold
        segs = detect_segments(mask, cfg.min_contiguous_channels)
        anomalies.append(segs)

    return diffs, max_diffs, mean_diffs, anomalies


def run_interactive_viewer(grouped: np.ndarray, baseline: np.ndarray, anomalies: list[list[np.ndarray]],
                           diffs: np.ndarray, group_times: list[tuple[str, str]], cfg: Config):
    n_groups = grouped.shape[1]
    positive = grouped[grouped > 0]
    y_min = float(np.min(positive)) * 0.5 if positive.size else 1e-8
    y_max = float(np.max(grouped)) * 3 if np.max(grouped) > 0 else 1

    fig = plt.figure(figsize=(14, 8.5))
    ax = fig.add_axes([0.08, 0.30, 0.88, 0.60])
    ax.set_yscale("log")
    ax.set_xlim(1, cfg.channel_count)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("道址(Channel)")
    ax.set_ylabel("归一化计数(Counts)")

    x = np.arange(1, cfg.channel_count + 1)
    l_base, = ax.plot(x, baseline, color="gray", lw=2.5, label="基准谱")
    l_cur, = ax.plot(x, grouped[:, 0], color="#1f77b4", lw=2, label="当前谱")
    ax.legend(loc="upper right")

    status = fig.text(0.08, 0.22, "状态: 正常", color="green", fontsize=11, weight="bold")
    detail = fig.text(0.08, 0.19, "异常信息: 无", fontsize=10)
    time_txt = fig.text(0.08, 0.16, "时间: -", fontsize=10)

    s_ax = fig.add_axes([0.12, 0.10, 0.70, 0.03])
    slider = Slider(s_ax, "组号", 1, n_groups, valinit=1, valstep=1)

    bprev = Button(fig.add_axes([0.84, 0.09, 0.05, 0.04]), "上一组")
    bnext = Button(fig.add_axes([0.90, 0.09, 0.05, 0.04]), "下一组")

    patches: list = []

    def clear_patches():
        for p in patches:
            p.remove()
        patches.clear()

    def update(idx1: int):
        i = idx1 - 1
        l_cur.set_ydata(grouped[:, i])
        clear_patches()

        segs = anomalies[i]
        if segs:
            for seg in segs:
                x0, x1 = int(seg[0] + 1), int(seg[-1] + 1)
                p = ax.axvspan(x0, x1, color="red", alpha=0.2, linestyle="--")
                patches.append(p)
            status.set_text(f"状态: ⚠️ 报警（{len(segs)}个连续异常区段）")
            status.set_color("red")
            detail.set_text(
                "异常信息: " + " | ".join(
                    [f"{int(s[0]+1)}-{int(s[-1]+1)}道(最大偏差{np.max(diffs[s, i])*100:.1f}%)" for s in segs[:3]]
                )
            )
        else:
            status.set_text("状态: 正常")
            status.set_color("green")
            detail.set_text("异常信息: 无连续异常区段")

        st, ed = group_times[i]
        time_txt.set_text(f"时间: {st} 至 {ed}")
        ax.set_title(f"第{idx1}/{n_groups}组能谱对比（阈值 {cfg.relative_threshold*100:.1f}%）")
        fig.canvas.draw_idle()

    slider.on_changed(lambda v: update(int(v)))
    bprev.on_clicked(lambda event: slider.set_val(max(1, int(slider.val) - 1)))
    bnext.on_clicked(lambda event: slider.set_val(min(n_groups, int(slider.val) + 1)))

    def on_key(event):
        if event.key == "left":
            slider.set_val(max(1, int(slider.val) - 1))
        elif event.key == "right":
            slider.set_val(min(n_groups, int(slider.val) + 1))

    fig.canvas.mpl_connect("key_press_event", on_key)
    update(1)
    plt.show()


def save_outputs(output: Path, grouped: np.ndarray, baseline: np.ndarray, diffs: np.ndarray,
                 max_diffs: np.ndarray, mean_diffs: np.ndarray, anomalies: list[list[np.ndarray]],
                 sorted_files: Sequence[str], group_times: list[tuple[str, str]], cfg: Config) -> None:
    output.mkdir(parents=True, exist_ok=True)
    np.savetxt(output / "时间窗口基准.txt", baseline, fmt="%.10f")

    for g in range(grouped.shape[1]):
        np.savetxt(output / f"group_{g+1:02d}.txt", grouped[:, g], fmt="%.10f")

    with open(output / "相对偏差统计.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["组号", "最大相对偏差%", "平均相对偏差%", "连续异常区段数", "时间起", "时间止"])
        for i in range(grouped.shape[1]):
            w.writerow([
                i + 1,
                max_diffs[i] * 100,
                mean_diffs[i] * 100,
                len(anomalies[i]),
                group_times[i][0],
                group_times[i][1],
            ])

    with open(output / "处理文件顺序.txt", "w", encoding="utf-8") as f:
        for i, name in enumerate(sorted_files, start=1):
            f.write(f"{i:04d}\t{name}\n")


def main() -> None:
    cfg = Config()

    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="选择原始能谱文件夹")
    root.destroy()
    if not folder:
        print("未选择文件夹，程序退出")
        return

    folder_path = Path(folder)
    files = [p.name for p in folder_path.glob("*.txt") if cfg.file_feature in p.name]
    if not files:
        print(f"未找到包含 {cfg.file_feature} 的 txt 文件")
        return

    rec = []
    for i, f in enumerate(files, start=1):
        dt, d, t = extract_time_from_name(f, i)
        rec.append((dt, f, d, t))
    rec.sort(key=lambda x: x[0])

    sorted_files = [r[1] for r in rec]
    file_dates = [r[2] for r in rec]
    file_times = [r[3] for r in rec]

    print(f"找到 {len(sorted_files)} 个文件")
    print(f"最早: {file_dates[0]} {file_times[0]}")
    print(f"最晚: {file_dates[-1]} {file_times[-1]}")

    raw = np.zeros((cfg.channel_count, len(sorted_files)), dtype=float)
    for i, fn in enumerate(sorted_files):
        raw[:, i] = read_spectrum(folder_path / fn, cfg.channel_count)

    st_idx, ed_idx = choose_baseline_range(sorted_files, file_dates, file_times)
    print(f"基准区间: {st_idx} - {ed_idx}")

    h_range, fe_range = cfg.hydrogen_range_default, cfg.iron_range_default
    if cfg.align_spectra:
        h_range, fe_range = select_peak_ranges(raw[:, 0], cfg)

    if cfg.align_spectra:
        aligned, hydrogen_peaks_file, h_ref, fe_ref = align_all_spectra(raw, h_range, fe_range, cfg)
    else:
        aligned = raw
        hydrogen_peaks_file = np.array([], dtype=int)
        h_ref = int(np.argmax(np.median(aligned[:, st_idx - 1:ed_idx], axis=1)) + 1)
        fe_ref, _, _ = find_peak_smoothed(np.median(aligned[:, st_idx - 1:ed_idx], axis=1), *cfg.iron_range_default, cfg.smoothing_window)

    baseline_raw = np.median(aligned[:, st_idx - 1:ed_idx], axis=1)
    baseline = baseline_raw / np.sum(baseline_raw)
    baseline_h_peak = int(round(np.median(hydrogen_peaks_file[st_idx - 1:ed_idx]))) if hydrogen_peaks_file.size else int(np.argmax(baseline) + 1)
    amp_start = min(cfg.channel_count, baseline_h_peak + cfg.amp_start_offset)
    baseline[amp_start - 1:] *= cfg.amplification_factor

    grouped, group_counts, group_h = process_groups(aligned, hydrogen_peaks_file, cfg)

    diffs, max_diffs, mean_diffs, anomalies = detect_distortions(grouped, baseline, group_h, fe_ref, cfg)

    n_groups = grouped.shape[1]
    group_times: list[tuple[str, str]] = []
    for g in range(n_groups):
        s = g * cfg.group_size
        e = min((g + 1) * cfg.group_size, len(sorted_files)) - 1
        group_times.append((f"{file_dates[s]} {file_times[s]}", f"{file_dates[e]} {file_times[e]}"))

    output = folder_path / cfg.output_folder_name
    save_outputs(output, grouped, baseline, diffs, max_diffs, mean_diffs, anomalies, sorted_files, group_times, cfg)
    print(f"输出完成: {output}")

    run_interactive_viewer(grouped, baseline, anomalies, diffs, group_times, cfg)


if __name__ == "__main__":
    main()
