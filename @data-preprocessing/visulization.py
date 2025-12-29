import os
import json
from typing import Iterable, List, Tuple

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GENUS_MAP_PATH = os.path.join(PROJECT_ROOT, "@data", "auxiliary", "genus_map.json")
IMAESTRO_BASE_DIR = os.path.join(PROJECT_ROOT, "@data", "imaestro")
TRAINING_DATA_DIR = os.path.join(IMAESTRO_BASE_DIR, "training-data")


def _sci_formatter(x: float, pos: int | None = None) -> str:
    """将数字格式化为形如 1.23e2 的科学计数法（保留两位小数）。"""
    if x == 0:
        return "0"
    s = f"{x:.2e}"  # 例如 '1.23e+02'
    mantissa, exp = s.split("e")
    exp = exp.lstrip("+0") or "0"  # 去掉前导 + 和 0
    return f"{mantissa}e{exp}"


def plot_genus_distribution_for_sites(
    sites: Iterable[str] = ("bauges", "milicz", "sneznik"),
    save_path: str | None = None,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """可视化每个 site 的 dom_genus_smooth 类别分布（柱状图）。

    - 输入数据：@data/imaestro/{site}/output_tiff/{site}_dom_genus_smooth.tif
    - 横轴：genus_map.json 定义的属名
    - 一张 figure 中纵向 3 行子图（每个 site 一张）
    """

    with open(GENUS_MAP_PATH, "r", encoding="utf-8") as f:
        genus_to_code = json.load(f)

    # 反向映射：数值编码 -> 属名
    code_to_genus = {int(v): k for k, v in genus_to_code.items()}
    codes_sorted = sorted(code_to_genus.keys())
    genus_names = [code_to_genus[c] for c in codes_sorted]

    sites = list(sites)
    n_sites = len(sites)

    # 根据类别数量自适应宽度
    fig_width = 10
    fig_height = 10
    # 每个子图有自己的 x 轴类别
    fig, axes = plt.subplots(n_sites, 1, figsize=(fig_width, fig_height), sharex=False)
    if n_sites == 1:
        axes = [axes]

    # 为每个类别分配一个固定颜色，在三个子图中保持一致
    cmap = plt.get_cmap("tab20")
    num_colors = cmap.N
    colors = [cmap(i % num_colors) for i in range(len(genus_names))]

    for ax, site in zip(axes, sites):
        tif_path = os.path.join(
            IMAESTRO_BASE_DIR,
            site,
            "output_tiff",
            f"{site}_dom_genus_smooth.tif",
        )

        if not os.path.exists(tif_path):
            ax.text(
                0.5,
                0.5,
                f"File not found:\n{os.path.relpath(tif_path, PROJECT_ROOT)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        with rasterio.open(tif_path) as src:
            data = src.read(1).astype(float)
            nodata = src.nodata

        mask = np.ones_like(data, dtype=bool)
        if nodata is not None:
            mask &= data != nodata
        mask &= ~np.isnan(data)

        values = data[mask].astype(int)

        counts = np.zeros(len(genus_names), dtype=int)
        if values.size > 0:
            unique, freq = np.unique(values, return_counts=True)
            code_to_index = {code: idx for idx, code in enumerate(codes_sorted)}
            for code, f in zip(unique, freq):
                idx = code_to_index.get(int(code))
                if idx is not None:
                    counts[idx] = f

        total = counts.sum()
        if total == 0:
            ax.text(
                0.5,
                0.5,
                "No valid genus data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        # 只显示该 site 实际出现过的 genus
        idx_nonzero = np.where(counts > 0)[0]
        site_genus = [genus_names[i] for i in idx_nonzero]
        site_counts = counts[idx_nonzero]
        site_colors = [colors[i] for i in idx_nonzero]

        max_count = float(site_counts.max())

        bar_positions = np.arange(len(site_genus))
        bars = ax.bar(bar_positions, site_counts, color=site_colors, width=0.8)
        ax.set_ylabel("Count")
        ax.set_title(site)

        # 在每个 bar 上标注百分比和个数（科学计数法，保留两位小数）
        percents = site_counts.astype(float) / float(total) * 100.0
        for rect, p, c in zip(bars, percents, site_counts):
            height = rect.get_height()
            y_text = height + max_count * 0.03
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                y_text,
                f"{p:.2f}%\n{_sci_formatter(float(c))}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        ax.set_ylim(0.0, max_count * 1.25)

        ax.set_xticks(bar_positions)
        ax.set_xticklabels(site_genus, rotation=45, ha="right")

        # y 轴使用科学计数法刻度
        ax.yaxis.set_major_formatter(FuncFormatter(_sci_formatter))

    fig.subplots_adjust(right=0.8, bottom=0.25)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes

def visualize_sample_patches(
    train_index: int = 0,
    val_index: int = 0,
    test_index: int = 0,
    save_path: str | None = None,
):
    """从 npy 训练数据中各挑选一个样本，可视化为 3 行 5 列的图。

    行：train, val, test
    列：VV, VH, Genus, Height, Biomass

    - VV, VH 使用 "cividis" 连续色标
    - Genus 使用离散色标（"tab20"）
    - Height, Biomass 使用 "viridis" 连续色标
    """

    # 加载 npy
    patches_train = np.load(os.path.join(TRAINING_DATA_DIR, "patches_train.npy"))
    patches_val = np.load(os.path.join(TRAINING_DATA_DIR, "patches_val.npy"))
    patches_test = np.load(os.path.join(TRAINING_DATA_DIR, "patches_test.npy"))

    n_train, _, H, W = patches_train.shape
    n_val = patches_val.shape[0]
    n_test = patches_test.shape[0]

    if not (0 <= train_index < n_train):
        raise IndexError(f"train_index {train_index} out of range [0, {n_train})")
    if not (0 <= val_index < n_val):
        raise IndexError(f"val_index {val_index} out of range [0, {n_val})")
    if not (0 <= test_index < n_test):
        raise IndexError(f"test_index {test_index} out of range [0, {n_test})")

    # 选取样本 (C, H, W)
    sample_train = patches_train[train_index]
    sample_val = patches_val[val_index]
    sample_test = patches_test[test_index]

    # 通道顺序：0: VH, 1: VV, 2: biomass, 3: genus, 4: height
    def _extract_views(sample: np.ndarray) -> list[np.ndarray]:
        vh_raw = sample[0].astype(float)
        vv_raw = sample[1].astype(float)
        biomass = sample[2]
        genus = sample[3]
        height = sample[4]

        s1_raw = np.stack([vh_raw, vv_raw], axis=0)
        max_abs = float(np.nanmax(np.abs(s1_raw)))
        if max_abs > 100.0:
            scale = 0.01
            vh = vh_raw * scale
            vv = vv_raw * scale
        else:
            vh = vh_raw
            vv = vv_raw

        return [vv, vh, genus, height, biomass]

    data_rows = [
        _extract_views(sample_train),
        _extract_views(sample_val),
        _extract_views(sample_test),
    ]

    row_labels = ["train", "val", "test"]
    col_labels = ["vv", "vh", "genus", "height", "biomass"]

    fig, axes = plt.subplots(3, 5, figsize=(12, 8))

    # 在第一行上方写列标题
    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontsize=10)

    # 行首竖着写 train/val/test
    for i, row_label in enumerate(row_labels):
        axes[i, 0].set_ylabel(row_label, rotation=90, fontsize=10)

    for i in range(3):
        for j in range(5):
            ax = axes[i, j]
            img = data_rows[i][j]

            if j == 2:
                # Genus：离散色标
                img_int = img.astype(int)
                im = ax.imshow(img_int, cmap="tab20", interpolation="nearest")
            elif j in (0, 1):
                # VV / VH：cividis
                im = ax.imshow(img, cmap="cividis")
            else:
                # Height / Biomass：viridis
                im = ax.imshow(img, cmap="viridis")

            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


# plot_genus_distribution_for_sites(save_path="../@plots/data-insight/genus_distribution.png")
visualize_sample_patches(
    train_index=10,
    val_index=5,
    test_index=3,
    save_path="../@plots/data-insight/sample_patches_10_5_3.png",
)