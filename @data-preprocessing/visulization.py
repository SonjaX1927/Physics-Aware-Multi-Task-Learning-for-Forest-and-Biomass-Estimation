import os
import json
from typing import Iterable, List, Tuple

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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

    # 根据类别数量自适应宽度（整体稍小一些）
    fig_width = 10
    fig_height = 6

    # 特殊布局：3 个 site，bauges 在第一行整行，milicz 和 sneznik 在第二行左右各一幅
    if set(sites) == {"bauges", "milicz", "sneznik"} and n_sites == 3:
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1.0], width_ratios=[1.5, 0.7])

        axes_dict: dict[str, plt.Axes] = {}
        axes_dict["bauges"] = fig.add_subplot(gs[0, :])
        axes_dict["milicz"] = fig.add_subplot(gs[1, 0])
        axes_dict["sneznik"] = fig.add_subplot(gs[1, 1])

        axes = [axes_dict[s] for s in sites]
    else:
        # 回退到常规的按行堆叠布局
        fig, axes = plt.subplots(n_sites, 1, figsize=(fig_width, fig_height), sharex=False)
        if n_sites == 1:
            axes = [axes]

    # 为每个 site 使用渐变色，并给柱子添加描边
    base_cmap = plt.get_cmap("viridis")

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

        # 只显示该 site 实际出现过的 genus，并按样本量从大到小排序
        idx_nonzero = np.where(counts > 0)[0]
        if idx_nonzero.size == 0:
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

        site_genus = [genus_names[i] for i in idx_nonzero]
        site_counts = counts[idx_nonzero]

        # 按计数从大到小重新排序
        sort_idx = np.argsort(-site_counts.astype(float))
        site_counts = site_counts[sort_idx]
        site_genus = [site_genus[i] for i in sort_idx]

        # 渐变色：同一个 site 内从浅到深
        n_bars = len(site_genus)
        color_positions = np.linspace(0.3, 0.9, n_bars)
        site_colors = [base_cmap(p) for p in color_positions]

        max_count = float(site_counts.max())

        bar_positions = np.arange(len(site_genus))
        bars = ax.bar(
            bar_positions,
            site_counts,
            color=site_colors,
            width=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_ylabel("Count")
        if site == "bauges":
            ax.set_title("Bauges")
        elif site == "milicz":
            ax.set_title("Milicz")
        else:
            ax.set_title("Sneznik")

        # 在每个 bar 上标注百分比和个数（科学计数法，保留两位小数）
        percents = site_counts.astype(float) / float(total) * 100.0
        for rect, p, c in zip(bars, percents, site_counts):
            height = rect.get_height()
            y_text = height + max_count * 0.03
            # 修改 _sci_formatter 为保留 1 位小数
            c_str = f"{float(c):.1e}"
            mantissa, exp = c_str.split("e")
            exp = exp.lstrip("+0") or "0"
            c_formatted = f"{mantissa}e{exp}"
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                y_text,
                f"{p:.1f}%\n{c_formatted}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_ylim(0.0, max_count * 1.25)

        ax.set_xticks(bar_positions)
        ax.set_xticklabels(site_genus, rotation=45, ha="right")

        # y 轴使用科学计数法刻度，不显示加号
        def sci_formatter(x, pos):
            if x == 0:
                return '0'
            s = f'{x:.1e}'
            mantissa, exp = s.split('e')
            exp = exp.lstrip('+0') or '0'
            return f'{mantissa}e{exp}'
        
        ax.yaxis.set_major_formatter(FuncFormatter(sci_formatter))

    fig.subplots_adjust(right=0.8, bottom=0.25)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def visualize_site_rasters(
    site: str,
    save_path: str | None = None,
    ):
    """可视化单个 site 的 5 个连续/离散栅格为一行 5 列子图。

    列顺序: VV, VH, Genus, Height, Biomass

    数据来源与 build_training_patches._stack_site_arrays 保持一致:
    - VV, VH:  @data/imaestro/{site}/output_backscatter/{site}_vv.tif / {site}_vh.tif
    - biomass: @data/imaestro/{site}/output_tiff/{site}_biomass_t_ha_smooth.tif
    - genus:   @data/imaestro/{site}/output_tiff/{site}_dom_genus_smooth.tif
    - height:  @data/imaestro/{site}/output_tiff/{site}_height95_smooth.tif

    可视化风格与 visualize_sample_patches 保持一致:
    - VV, VH 使用 "cividis" 连续色标
    - Genus 使用离散色标（"tab20"，掩蔽 NaN）
    - Height, Biomass 使用 "viridis" 连续色标
    - 所有子图隐藏坐标轴刻度，仅显示 colorbar，连续变量 colorbar 5 等分刻度
    """

    # 构造各通道的路径（与 build_training_patches 中保持一致）
    vv_path = os.path.join(
        IMAESTRO_BASE_DIR,
        site,
        "output_backscatter",
        f"{site}_vv.tif",
    )
    vh_path = os.path.join(
        IMAESTRO_BASE_DIR,
        site,
        "output_backscatter",
        f"{site}_vh.tif",
    )
    biomass_path = os.path.join(
        IMAESTRO_BASE_DIR,
        site,
        "output_tiff",
        f"{site}_biomass_t_ha_smooth.tif",
    )
    genus_path = os.path.join(
        IMAESTRO_BASE_DIR,
        site,
        "output_tiff",
        f"{site}_dom_genus_smooth.tif",
    )
    height_path = os.path.join(
        IMAESTRO_BASE_DIR,
        site,
        "output_tiff",
        f"{site}_height95_smooth.tif",
    )

    paths = [vv_path, vh_path, genus_path, height_path, biomass_path]
    col_labels = ["vv", "vh", "genus", "height", "biomass"]

    arrays: list[np.ndarray] = []
    nodatas: list[float | None] = []

    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing raster for site {site}: {p}")

        with rasterio.open(p) as src:
            arr = src.read(1).astype(float)
            nodata = src.nodata

        arrays.append(arr)
        nodatas.append(nodata)

    # 单行 5 列
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))

    # 设置列标题（只显示波段名，不包含 site 名）
    for j, label in enumerate(col_labels):
        axes[j].set_title(label, fontsize=8)

    # 在最左侧子图上竖直写 site 名
    axes[0].set_ylabel(site, rotation=90, fontsize=8)

    for j, (ax, img, nodata) in enumerate(zip(axes, arrays, nodatas)):
        if j == 2:
            # Genus：离散色标，与 visualize_sample_patches 中 genus 处理方式保持一致
            mask_invalid = ~np.isfinite(img)
            if nodata is not None:
                mask_invalid |= img == nodata

            img_int = np.zeros_like(img, dtype=int)
            if (~mask_invalid).any():
                img_int[~mask_invalid] = img[~mask_invalid].astype(int)

            img_masked = np.ma.array(img_int, mask=mask_invalid)

            # 使用 genus_map.json 中定义的完整编码范围，确保不同 site 颜色一致
            with open(GENUS_MAP_PATH, "r", encoding="utf-8") as f:
                genus_to_code = json.load(f)

            all_codes = sorted(int(v) for v in genus_to_code.values())
            if all_codes:
                vmin, vmax = min(all_codes), max(all_codes)
            else:
                vmin, vmax = 0, 1

            # 仅在 colorbar 上标注主类的简称
            main_genus_order = [
                ("fagus", "Fag."),
                ("abies", "Abi."),
                ("pinus", "Pin."),
                ("fraxinus", "Fra."),
                ("quercus", "Que."),
                ("picea", "Pic."),
                ("carpinus", "Car."),
            ]

            # 构建不区分大小写的 name->code 映射
            name_to_code_lower: dict[str, int] = {}
            for name, code in genus_to_code.items():
                try:
                    name_to_code_lower[name.lower()] = int(code)
                except Exception:
                    continue

            major_codes: list[int] = []
            major_labels: list[str] = []
            for name, short in main_genus_order:
                code = name_to_code_lower.get(name)
                if code is not None:
                    major_codes.append(code)
                    major_labels.append(short)

            cmap = plt.get_cmap("tab20").copy()
            cmap.set_bad(color="white")

            im = ax.imshow(
                img_masked,
                cmap=cmap,
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )

            cax = inset_axes(
                ax,
                width="3%",
                height="100%",
                loc="center right",
                borderpad=0.0,
            )
            cbar = plt.colorbar(im, cax=cax)

            # 只在 colorbar 上标出主类的编码位置，并使用缩写标签
            if major_codes:
                cbar.set_ticks(major_codes)
                cbar.set_ticklabels(major_labels)
            cbar.ax.tick_params(labelsize=6, pad=1)
        else:
            # 连续变量：VV / VH / Height / Biomass
            if j in (0, 1):
                cmap = "cividis"
            else:
                cmap = "viridis"

            valid_mask = np.isfinite(img)
            if nodata is not None:
                valid_mask &= img != nodata

            if valid_mask.any():
                vmin = float(np.nanmin(img[valid_mask]))
                vmax = float(np.nanmax(img[valid_mask]))
                if not np.isfinite(vmin) or not np.isfinite(vmax):
                    vmin, vmax = 0.0, 1.0
            else:
                vmin, vmax = 0.0, 1.0

            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

            cax = inset_axes(
                ax,
                width="3%",
                height="100%",
                loc="center right",
                borderpad=0.0,
            )
            cbar = plt.colorbar(im, cax=cax)
            ticks = np.linspace(vmin, vmax, 5)
            cbar.set_ticks(ticks)
            cbar.formatter = FormatStrFormatter("%.1f")  # one decimal
            cbar.update_ticks()
            cbar.ax.tick_params(labelsize=6, pad=0.5)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.2)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def visualize_all_sites_rasters(
    sites: Iterable[str] = ("bauges", "milicz", "sneznik"),
    save_path: str | None = None,
    ):
    """可视化所有 sites 的 5 个连续/离散栅格为 3 行 5 列子图，共享波段名。

    行：bauges, milicz, sneznik
    列：VV, VH, Genus, Height, Biomass
    """

    col_labels = ["VV", "VH", "Genus", "Height", "Biomass"]
    sites_list = list(sites)
    n_sites = len(sites_list)

    # 3 行 5 列，使用 constrained_layout 以避免 tight_layout 警告
    fig, axes = plt.subplots(n_sites, 5, figsize=(15, 3 * n_sites), 
                             constrained_layout=False, gridspec_kw={'wspace': 0.2, 'hspace': 0.25})
    if n_sites == 1:
        axes = axes.reshape(1, -1)

    # 设置列标题（只在第一行上方显示波段名）
    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontsize=10)

    # 加载 genus_map 用于统一 genus 色标范围
    with open(GENUS_MAP_PATH, "r", encoding="utf-8") as f:
        genus_to_code = json.load(f)

    all_codes = sorted(int(v) for v in genus_to_code.values())
    if all_codes:
        genus_vmin, genus_vmax = min(all_codes), max(all_codes)
    else:
        genus_vmin, genus_vmax = 0, 1

    # 主类编码和标签
    main_genus_order = [
        ("fagus", "Fag."),
        ("abies", "Abi."),
        ("pinus", "Pin."),
        ("fraxinus", "Fra."),
        ("quercus", "Que."),
        ("picea", "Pic."),
        ("carpinus", "Car."),
    ]

    name_to_code_lower: dict[str, int] = {}
    for name, code in genus_to_code.items():
        try:
            name_to_code_lower[name.lower()] = int(code)
        except Exception:
            continue

    major_codes: list[int] = []
    major_labels: list[str] = []
    for name, short in main_genus_order:
        code = name_to_code_lower.get(name)
        if code is not None:
            major_codes.append(code)
            major_labels.append(short)

    # 遍历每个 site
    for i, site in enumerate(sites_list):
        # 构造各通道的路径
        vv_path = os.path.join(IMAESTRO_BASE_DIR, site, "output_backscatter", f"{site}_vv.tif")
        vh_path = os.path.join(IMAESTRO_BASE_DIR, site, "output_backscatter", f"{site}_vh.tif")
        biomass_path = os.path.join(IMAESTRO_BASE_DIR, site, "output_tiff", f"{site}_biomass_t_ha_smooth.tif")
        genus_path = os.path.join(IMAESTRO_BASE_DIR, site, "output_tiff", f"{site}_dom_genus_smooth.tif")
        height_path = os.path.join(IMAESTRO_BASE_DIR, site, "output_tiff", f"{site}_height95_smooth.tif")

        paths = [vv_path, vh_path, genus_path, height_path, biomass_path]

        arrays: list[np.ndarray] = []
        nodatas: list[float | None] = []

        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing raster for site {site}: {p}")

            with rasterio.open(p) as src:
                arr = src.read(1).astype(float)
                nodata = src.nodata

            arrays.append(arr)
            nodatas.append(nodata)

        # 在最左侧子图上竖直写 site 名
        axes[i, 0].set_ylabel(site.capitalize(), rotation=90, fontsize=10)

        for j, (ax, img, nodata) in enumerate(zip(axes[i], arrays, nodatas)):
            if j == 2:
                # Genus：离散色标
                mask_invalid = ~np.isfinite(img)
                if nodata is not None:
                    mask_invalid |= img == nodata

                img_int = np.zeros_like(img, dtype=int)
                if (~mask_invalid).any():
                    img_int[~mask_invalid] = img[~mask_invalid].astype(int)

                img_masked = np.ma.array(img_int, mask=mask_invalid)

                cmap = plt.get_cmap("tab20").copy()
                cmap.set_bad(color="white")

                im = ax.imshow(
                    img_masked,
                    cmap=cmap,
                    interpolation="nearest",
                    vmin=genus_vmin,
                    vmax=genus_vmax,
                    aspect='auto',
                )

                cax = inset_axes(
                    ax,
                    width="3%",
                    height="100%",
                    loc="center right",
                    borderpad=0.0,
                )
                cbar = plt.colorbar(im, cax=cax)

                if major_codes:
                    cbar.set_ticks(major_codes)
                    cbar.set_ticklabels(major_labels)
                cbar.ax.tick_params(labelsize=8, pad=1)
            else:
                # 连续变量：VV / VH / Height / Biomass
                if j in (0, 1):
                    cmap = "cividis"
                else:
                    cmap = "viridis"

                valid_mask = np.isfinite(img)
                if nodata is not None:
                    valid_mask &= img != nodata

                if valid_mask.any():
                    vmin = float(np.nanmin(img[valid_mask]))
                    vmax = float(np.nanmax(img[valid_mask]))
                    if not np.isfinite(vmin) or not np.isfinite(vmax):
                        vmin, vmax = 0.0, 1.0
                else:
                    vmin, vmax = 0.0, 1.0

                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

                cax = inset_axes(
                    ax,
                    width="3%",
                    height="100%",
                    loc="center right",
                    borderpad=0.0,
                )
                cbar = plt.colorbar(im, cax=cax)
                ticks = np.linspace(vmin, vmax, 5)
                cbar.set_ticks(ticks)
                cbar.formatter = FormatStrFormatter("%.1f")
                cbar.update_ticks()
                cbar.ax.tick_params(labelsize=8, pad=0.5)

            ax.set_xticks([])
            ax.set_yticks([])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_genus_value_histograms_across_sites(
    value_type: str = "biomass",
    sites: Iterable[str] = ("bauges", "milicz", "sneznik"),
    bins: int = 20,
    save_path: str | None = None,
    ):
    """按属（genus）分别绘制 height/biomass 在所有 site 上的像元直方图。

    - 输入数据（以 biomass 为例）：
      - dom_genus：@data/imaestro/{site}/output_tiff/{site}_dom_genus_smooth.tif
      - biomass： @data/imaestro/{site}/output_tiff/{site}_biomass_t_ha_smooth.tif
      - height：  @data/imaestro/{site}/output_tiff/{site}_height_m_smooth.tif
    - 首先在所有 site 上按像元汇总每个 genus 的 value（height/biomass），
      然后为每个 genus 画一个直方图，所有子图排布在一张大图中（多行多列）。
    """

    value_type = value_type.lower()
    if value_type not in {"biomass", "height"}:
        raise ValueError(f"Unsupported value_type: {value_type}. Use 'biomass' or 'height'.")

    if value_type == "biomass":
        value_suffix = "_biomass_t_ha_smooth.tif"
        value_label = "Biomass (t/ha)"
        title = ""
    else:
        value_suffix = "_height95_smooth.tif"
        value_label = "Height (m)"
        title = ""

    with open(GENUS_MAP_PATH, "r", encoding="utf-8") as f:
        genus_to_code = json.load(f)
    code_to_genus = {int(v): k for k, v in genus_to_code.items()}

    # 聚合所有 site 上的数值：code -> list[np.ndarray]
    genus_values: dict[int, list[np.ndarray]] = {}

    sites = list(sites)
    for site in sites:
        genus_tif = os.path.join(
            IMAESTRO_BASE_DIR,
            site,
            "output_tiff",
            f"{site}_dom_genus_smooth.tif",
        )
        value_tif = os.path.join(
            IMAESTRO_BASE_DIR,
            site,
            "output_tiff",
            f"{site}{value_suffix}",
        )

        if not os.path.exists(genus_tif) or not os.path.exists(value_tif):
            print("Skip site due to missing files:", site)
            print("  genus:", os.path.relpath(genus_tif, PROJECT_ROOT))
            print("  value:", os.path.relpath(value_tif, PROJECT_ROOT))
            continue

        with rasterio.open(genus_tif) as src_genus, rasterio.open(value_tif) as src_val:
            genus_data = src_genus.read(1).astype(float)
            value_data = src_val.read(1).astype(float)
            nodata_genus = src_genus.nodata
            nodata_value = src_val.nodata

        # 有效像元掩膜：两张图都非 nodata 且非 NaN
        mask = np.ones_like(genus_data, dtype=bool)
        if nodata_genus is not None:
            mask &= genus_data != nodata_genus
        if nodata_value is not None:
            mask &= value_data != nodata_value
        mask &= np.isfinite(genus_data) & np.isfinite(value_data)

        if not mask.any():
            continue

        genus_vals = genus_data[mask].astype(int)
        value_vals = value_data[mask].astype(float)

        unique_codes = np.unique(genus_vals)
        for code in unique_codes:
            idx = genus_vals == code
            if not np.any(idx):
                continue
            vals = value_vals[idx]
            if vals.size == 0:
                continue
            genus_values.setdefault(int(code), []).append(vals)

    # 将 list[np.ndarray] 合并为一个大数组，并只保留在 genus_map 中出现的 code
    aggregated: dict[int, np.ndarray] = {}
    for code, chunks in genus_values.items():
        if code not in code_to_genus:
            continue
        concatenated = np.concatenate(chunks)
        if concatenated.size > 0:
            aggregated[code] = concatenated

    if not aggregated:
        print("No valid data found for any genus.")
        return None, None

    # 按样本量排序：每个 genus 的像元数量从大到小
    codes_sorted = sorted(aggregated.keys())
    sizes = np.array([aggregated[c].size for c in codes_sorted], dtype=float)
    sort_idx = np.argsort(-sizes)
    codes_sorted = [codes_sorted[i] for i in sort_idx]
    genus_names = [code_to_genus[c] for c in codes_sorted]
    values_list = [aggregated[c] for c in codes_sorted]

    # 全局取值范围，便于不同 genus 之间对比
    all_vals = np.concatenate(values_list)
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))

    n_genus = len(codes_sorted)
    # 合理设置子图布局：最多 4 列，整体子图更紧凑一些
    n_cols = min(6, n_genus)
    n_rows = int(np.ceil(n_genus / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2 * n_cols, 2.6 * n_rows),
        sharex=True,
        sharey=True,
    )
    # 统一为 1D 列表，便于遍历
    if isinstance(axes, plt.Axes):
        axes_list = [axes]
    else:
        axes_list = np.array(axes).ravel().tolist()

    for idx, (genus_name, vals) in enumerate(zip(genus_names, values_list)):
        ax = axes_list[idx]
        ax.hist(
            vals,
            bins=bins,
            range=(vmin, vmax),
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_title(genus_name, fontsize=20)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.1e}"))
        ax.tick_params(axis="both", labelsize=13)

    # 对多余子图（如果有）隐藏坐标轴
    for j in range(n_genus, len(axes_list)):
        axes_list[j].set_axis_off()

    # 只在最底行的子图上标注 x 轴标签
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            if idx >= len(axes_list):
                continue
            ax = axes_list[idx]
            if r == n_rows - 1 and c==3 and ax.get_visible():
                ax.set_xlabel(value_label, fontsize=17)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes_list


def plot_genus_distribution_by_split(
    save_path: str | None = None,
    ):
    """按像元统计类别在 train/val/test 三个 split 中的分布，对比柱状图。

    - 统计单位：像元（pixels），而非 patch 数量。
    - 数据来源：patches_{split}.npy 的 genus 通道（index=3）。
    - x 轴为 genus 名称，每个类别下面有三根柱（train, val, test）。
    - 每个柱上方用两行文字标识：百分比和像元个数（均保留两位小数）。
    """

    # 读取三个 split 的 patches，并提取 genus 通道的像元值
    def _extract_genus_pixels(split_name: str) -> np.ndarray:
        patches = np.load(os.path.join(TRAINING_DATA_DIR, f"patches_{split_name}.npy"))
        # 通道顺序：0 VH, 1 VV, 2 biomass, 3 genus, 4 height
        genus = patches[:, 3, :, :].astype(float)
        vals = genus[np.isfinite(genus)]
        return vals.astype(int)

    vals_train = _extract_genus_pixels("train")
    vals_val = _extract_genus_pixels("val")
    vals_test = _extract_genus_pixels("test")

    if vals_train.size + vals_val.size + vals_test.size == 0:
        print("No valid genus pixels in any split.")
        return None, None

    all_codes = np.unique(
        np.concatenate([vals_train.ravel(), vals_val.ravel(), vals_test.ravel()])
    )
    all_codes = np.sort(all_codes.astype(int))

    # 将数值编码映射为属名，用于 x 轴标签
    try:
        with open(GENUS_MAP_PATH, "r", encoding="utf-8") as f:
            genus_to_code = json.load(f)
        code_to_genus = {int(v): k for k, v in genus_to_code.items()}
        genus_labels = [code_to_genus.get(int(c), str(int(c))) for c in all_codes]
    except Exception:
        # 如果读取或解析失败，则退回到使用数值编码
        genus_labels = [str(int(c)) for c in all_codes]

    def _count(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
        return np.array([np.sum(values == c) for c in codes], dtype=int)

    counts_train = _count(vals_train, all_codes)
    counts_val = _count(vals_val, all_codes)
    counts_test = _count(vals_test, all_codes)

    # 按照 train 像元数量从大到小对类别进行排序，便于比较主导类别
    sort_idx = np.argsort(-counts_train.astype(float))
    all_codes = all_codes[sort_idx]
    genus_labels = [genus_labels[i] for i in sort_idx]
    counts_train = counts_train[sort_idx]
    counts_val = counts_val[sort_idx]
    counts_test = counts_test[sort_idx]

    x = np.arange(len(all_codes))
    # 调整每个类别内部三根柱子的宽度，减小相邻类别之间的间隔
    width = 0.3
    fig, ax = plt.subplots(figsize=(18, 4))

    bars_train = ax.bar(x - width, counts_train, width, label="train")
    bars_val = ax.bar(x, counts_val, width, label="val")
    bars_test = ax.bar(x + width, counts_test, width, label="test")

    ax.set_xticks(x)
    ax.tick_params(axis="x", labelsize=8) 
    ax.tick_params(axis="y", labelsize=8) 
    ax.set_xticklabels(genus_labels, rotation=45, ha="right")
    ax.set_ylabel("Pixel count", fontsize=8)
    ax.set_xlabel("Genus", fontsize=8)
    ax.set_xlim(-0.5, len(all_codes) - 0.5)
    ax.legend(fontsize=8)

    # 在每个柱上标注百分比和像元个数（两行，均保留两位小数）
    totals = [counts_train.sum(), counts_val.sum(), counts_test.sum()]
    max_count = float(max(counts_train.max(), counts_val.max(), counts_test.max()))

    for bars, counts, total in zip(
        [bars_train, bars_val, bars_test],
        [counts_train, counts_val, counts_test],
        totals,
    ):
        for rect, c in zip(bars, counts):
            height = rect.get_height()
            if total > 0 and height > 0:
                p = float(c) / float(total) * 100.0
                # 相对于柱高增加一个小偏移量，将文字放在柱顶稍上方
                y_text = height + max_count * 0.01
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    y_text,
                    # f"{p:.2f}%\n{float(c):.2f}",
                    f"{p:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=5,
                )

    # y 轴使用科学计数法，便于不同数量级比较
    ax.yaxis.set_major_formatter(FuncFormatter(_sci_formatter))

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_genus_distribution_by_split_patch_level(
    save_path: str | None = None,
    ):
    """按 patch 级别统计类别在 train/val/test 三个 split 中的分布对比柱状图。

    - 统计单位：patch（dominant_genus 标签）。
    - 数据来源：labels_dominant_genus_{split}.npy。
    - x 轴为 genus 名称，每个类别下面有三根柱（train, val, test）。
    - 每个柱上方用两行文字标识：百分比和 patch 个数（均保留两位小数）。
    """

    labels_train = np.load(
        os.path.join(TRAINING_DATA_DIR, "labels_dominant_genus_train.npy")
    ).astype(float)
    labels_val = np.load(
        os.path.join(TRAINING_DATA_DIR, "labels_dominant_genus_val.npy")
    ).astype(float)
    labels_test = np.load(
        os.path.join(TRAINING_DATA_DIR, "labels_dominant_genus_test.npy")
    ).astype(float)

    vals_train = labels_train[np.isfinite(labels_train)].astype(int)
    vals_val = labels_val[np.isfinite(labels_val)].astype(int)
    vals_test = labels_test[np.isfinite(labels_test)].astype(int)

    if vals_train.size + vals_val.size + vals_test.size == 0:
        print("No valid patch labels in any split.")
        return None, None

    all_codes = np.unique(
        np.concatenate([vals_train.ravel(), vals_val.ravel(), vals_test.ravel()])
    )
    all_codes = np.sort(all_codes.astype(int))

    # 将数值编码映射为属名，用于 x 轴标签
    try:
        with open(GENUS_MAP_PATH, "r", encoding="utf-8") as f:
            genus_to_code = json.load(f)
        code_to_genus = {int(v): k for k, v in genus_to_code.items()}
        genus_labels = [code_to_genus.get(int(c), str(int(c))) for c in all_codes]
    except Exception:
        genus_labels = [str(int(c)) for c in all_codes]

    def _count(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
        return np.array([np.sum(values == c) for c in codes], dtype=int)

    counts_train = _count(vals_train, all_codes)
    counts_val = _count(vals_val, all_codes)
    counts_test = _count(vals_test, all_codes)
    
    sort_idx = np.argsort(counts_train + counts_val + counts_test)[::-1]
    all_codes = all_codes[sort_idx]
    counts_train = counts_train[sort_idx]
    counts_val = counts_val[sort_idx]
    counts_test = counts_test[sort_idx]
    genus_labels = [genus_labels[i] for i in sort_idx]

    x = np.arange(len(all_codes))
    # 与像元级别的图保持一致，使用更大的柱宽以减小类别之间的间隔
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 4))

    bars_train = ax.bar(
        x - width,
        counts_train,
        width,
        label="train",
        edgecolor="black",
        linewidth=0.6,
    )
    bars_val = ax.bar(
        x,
        counts_val,
        width,
        label="val",
        edgecolor="black",
        linewidth=0.6,
    )
    bars_test = ax.bar(
        x + width,
        counts_test,
        width,
        label="test",
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_xticks(x)
    ax.tick_params(axis="y", labelsize=8) 
    ax.set_xticklabels(genus_labels, fontsize=10, ha="right")
    ax.set_ylabel("Patch count", fontsize=10)
    ax.set_xlabel("Genus", fontsize=10)
    ax.legend(fontsize=10)

    # 在每个柱上标注百分比和 patch 个数（两行，均保留两位小数）
    totals = [counts_train.sum(), counts_val.sum(), counts_test.sum()]
    max_count = float(max(counts_train.max(), counts_val.max(), counts_test.max()))

    # 给顶部留出更多空间，避免文字贴在坐标轴边框上
    ax.set_ylim(0.0, max_count * 1.30)

    for bars, counts, total in zip(
        [bars_train, bars_val, bars_test],
        [counts_train, counts_val, counts_test],
        totals,
    ):
        for rect, c in zip(bars, counts):
            height = rect.get_height()
            if total > 0 and height > 0:
                p = float(c) / float(total) * 100.0
                # 在柱顶上方留出足够空隙，避免与图框重叠
                y_text = height + max_count * 0.01
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    y_text,
                    f"{p:.1f}%\n{float(c):.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def _plot_scalar_histograms_by_split(
    channel_idx: int,
    title: str,
    xlabel: str,
    bins: int = 40,
    max_value: float | None = None,
    save_path: str | None = None,
    ):
    """通用函数：对比 train/val/test 在给定通道上的像元直方图。"""

    patches_train = np.load(os.path.join(TRAINING_DATA_DIR, "patches_train.npy"))
    patches_val = np.load(os.path.join(TRAINING_DATA_DIR, "patches_val.npy"))
    patches_test = np.load(os.path.join(TRAINING_DATA_DIR, "patches_test.npy"))

    def _extract_valid_channel(patches: np.ndarray) -> np.ndarray:
        vals = patches[:, channel_idx, :, :].astype(float)
        vals = vals[np.isfinite(vals)]
        if max_value is not None:
            vals = vals[vals <= max_value]
        return vals

    vals_train = _extract_valid_channel(patches_train)
    vals_val = _extract_valid_channel(patches_val)
    vals_test = _extract_valid_channel(patches_test)

    all_vals = np.concatenate([
        vals_train.ravel(),
        vals_val.ravel(),
        vals_test.ravel(),
    ])

    if all_vals.size == 0:
        print("No valid data for histograms.")
        return None, None

    vmin = float(all_vals.min())
    vmax = float(all_vals.max())
    if max_value is not None:
        vmax = min(vmax, float(max_value))

    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    split_names = ["train", "val", "test"]
    split_vals = [vals_train, vals_val, vals_test]

    for ax, name, vals in zip(axes, split_names, split_vals):
        if vals.size == 0:
            ax.text(0.5, 0.5, f"No data in {name}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        ax.hist(
            vals,
            bins=bins,
            range=(vmin, vmax),
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_ylabel("Count")
        ax.set_title(name)
        ax.yaxis.set_major_formatter(FuncFormatter(_sci_formatter))

    axes[-1].set_xlabel(xlabel)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_height_biomass_histograms_by_split(
    bins: int = 40,
    height_max_value: float | None = None,
    biomass_max_value: float | None = None,
    save_path: str | None = None,
    ):
    """绘制 3 行 2 列直方图，左列为 height，右列为 biomass，对比 train/val/test 的分布。"""

    patches_train = np.load(os.path.join(TRAINING_DATA_DIR, "patches_train.npy"))
    patches_val = np.load(os.path.join(TRAINING_DATA_DIR, "patches_val.npy"))
    patches_test = np.load(os.path.join(TRAINING_DATA_DIR, "patches_test.npy"))

    def _extract_valid_channel(patches: np.ndarray, channel_idx: int, max_value: float | None = None) -> np.ndarray:
        vals = patches[:, channel_idx, :, :].astype(float)
        vals = vals[np.isfinite(vals)]
        if max_value is not None:
            vals = vals[vals <= max_value]
        return vals

    height_train = _extract_valid_channel(patches_train, 4, height_max_value)
    height_val = _extract_valid_channel(patches_val, 4, height_max_value)
    height_test = _extract_valid_channel(patches_test, 4, height_max_value)

    biomass_train = _extract_valid_channel(patches_train, 2, biomass_max_value)
    biomass_val = _extract_valid_channel(patches_val, 2, biomass_max_value)
    biomass_test = _extract_valid_channel(patches_test, 2, biomass_max_value)

    all_height = np.concatenate([height_train.ravel(), height_val.ravel(), height_test.ravel()])
    all_biomass = np.concatenate([biomass_train.ravel(), biomass_val.ravel(), biomass_test.ravel()])

    if all_height.size == 0 or all_biomass.size == 0:
        print("No valid data for histograms.")
        return None, None

    height_vmin, height_vmax = float(all_height.min()), float(all_height.max())
    biomass_vmin, biomass_vmax = float(all_biomass.min()), float(all_biomass.max())
    
    if height_max_value is not None:
        height_vmax = min(height_vmax, float(height_max_value))
    if biomass_max_value is not None:
        biomass_vmax = min(biomass_vmax, float(biomass_max_value))

    fig, axes = plt.subplots(3, 2, figsize=(8, 6), sharex='col')
    split_names = ["Train", "Val", "Test"]
    height_vals = [height_train, height_val, height_test]
    biomass_vals = [biomass_train, biomass_val, biomass_test]

    for i, (name, h_vals, b_vals) in enumerate(zip(split_names, height_vals, biomass_vals)):
        ax_height = axes[i, 0]
        ax_biomass = axes[i, 1]
        
        if h_vals.size > 0:
            ax_height.hist(
                h_vals,
                bins=bins,
                range=(height_vmin, height_vmax),
                color="#1f77b4",
                edgecolor="black",
                linewidth=0.5,
            )
        else:
            ax_height.text(0.5, 0.5, f"No data in {name}", ha="center", va="center", transform=ax_height.transAxes, fontsize=12)
            ax_height.set_axis_off()
        
        if b_vals.size > 0:
            ax_biomass.hist(
                b_vals,
                bins=bins,
                range=(biomass_vmin, biomass_vmax),
                color="#1f77b4",
                edgecolor="black",
                linewidth=0.5,
            )
        else:
            ax_biomass.text(0.5, 0.5, f"No data in {name}", ha="center", va="center", transform=ax_biomass.transAxes)
            ax_biomass.set_axis_off()
        
        ax_height.set_ylabel(name)
        ax_height.yaxis.set_major_formatter(FuncFormatter(_sci_formatter))
        ax_biomass.yaxis.set_major_formatter(FuncFormatter(_sci_formatter))

    axes[2, 0].set_xlabel("Height (m)", fontsize=12)
    axes[2, 1].set_xlabel("Biomass (Mg/ha)", fontsize=12)
    
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

    row_labels = ["Train", "Validation", "Test"]
    col_labels = ["VV", "VH", "Genus", "Height", "Biomass"]

    fig, axes = plt.subplots(3, 5, figsize=(12, 6))

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
                # Genus：离散色标，掩蔽 NaN 区域，只在 colorbar 上标出实际存在的整数编码
                mask_invalid = ~np.isfinite(img)

                # 只对有效像元做整数化，避免对 NaN 直接 astype(int) 产生 RuntimeWarning
                img_int = np.zeros_like(img, dtype=int)
                if (~mask_invalid).any():
                    img_int[~mask_invalid] = img[~mask_invalid].astype(int)

                img_masked = np.ma.array(img_int, mask=mask_invalid)

                # 仅基于有效像元的取值确定 vmin/vmax 和刻度
                valid_vals = img_int[~mask_invalid]
                if valid_vals.size == 0:
                    vmin, vmax = 0, 1
                    ticks = []
                else:
                    unique_vals = np.unique(valid_vals)
                    vmin, vmax = int(unique_vals.min()), int(unique_vals.max())
                    ticks = unique_vals.tolist()

                cmap = plt.get_cmap("tab20").copy()
                cmap.set_bad(color="white")

                im = ax.imshow(
                    img_masked,
                    cmap=cmap,
                    interpolation="nearest",
                    vmin=vmin,
                    vmax=vmax,
                )

                cbar = plt.colorbar(
                    im,
                    ax=ax,
                    fraction=0.03,  # thinner
                    pad=0.02,
                    aspect=30,      # keep bar tall relative to width
                )
                if ticks:
                    cbar.set_ticks(ticks)
                cbar.ax.tick_params(labelsize=8)

            else:
                # 连续变量：VV / VH / Height / Biomass
                if j in (0, 1):
                    cmap = "cividis"
                else:
                    cmap = "viridis"

                valid_mask = np.isfinite(img)
                if valid_mask.any():
                    vmin = float(np.nanmin(img[valid_mask]))
                    vmax = float(np.nanmax(img[valid_mask]))
                    if not np.isfinite(vmin) or not np.isfinite(vmax):
                        vmin, vmax = 0.0, 1.0
                else:
                    vmin, vmax = 0.0, 1.0

                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

                cbar = plt.colorbar(
                    im,
                    ax=ax,
                    fraction=0.03,  # thinner bar
                    pad=0.02,
                    aspect=30,      # match image height while thin
                )
                ticks = np.linspace(vmin, vmax, 5)
                cbar.set_ticks(ticks)
                cbar.formatter = FormatStrFormatter("%.1f")  # one decimal
                cbar.update_ticks()
                cbar.ax.tick_params(labelsize=8)

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.25, wspace=0.2)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes

def main():
    # plot_genus_distribution_for_sites(save_path="../@plots/data-insight/genus_distribution.png")
    visualize_sample_patches(
        train_index=72, # 9 
        val_index=80, # 15
        test_index=93, # 88
        save_path="../@plots/data-insight/sample_patches_10_5_3.png",
    )

    # 单独可视化三个 site 的 VV/VH/Genus/Height/Biomass（每次一行 5 列）
    # visualize_site_rasters(
    #     site="bauges",
    #     save_path="../@plots/data-insight/site_rasters_bauges.png",
    # )
    # visualize_site_rasters(
    #     site="milicz",
    #     save_path="../@plots/data-insight/site_rasters_milicz.png",
    # )
    # visualize_site_rasters(
    #     site="sneznik",
    #     save_path="../@plots/data-insight/site_rasters_sneznik.png",
    # )
    visualize_all_sites_rasters(
        sites=("bauges", "milicz", "sneznik"),
        save_path="../@plots/data-insight/all_sites_rasters.png",
    )
    # plot_genus_distribution_by_split(
    #     save_path="../@plots/data-insight/genus_split_distribution.png"
    # )
    # plot_height_histograms_by_split(
    #     save_path="../@plots/data-insight/height_hist_split.png"
    # )
    # plot_biomass_histograms_by_split(
    #     save_path="../@plots/data-insight/biomass_hist_split.png"
    # )
    # plot_height_biomass_histograms_by_split(
    #     bins=40,
    #     height_max_value=None,
    #     biomass_max_value=None,
    #     save_path="../@plots/data-insight/height_biomass_hist_split.png",
    # )
    # plot_genus_distribution_by_split_patch_level(
    #     save_path="../@plots/data-insight/genus_split_distribution_patch_level.png"
    # )
    # plot_genus_value_histograms_across_sites(
    #     value_type="biomass",
    #     save_path="../@plots/data-insight/biomass_hist_per_genus_across_sites.png",
    # )
    # plot_genus_value_histograms_across_sites(
    #     value_type="height",
    #     save_path="../@plots/data-insight/height_hist_per_genus_across_sites.png",
    # )

if __name__ == "__main__":
    main()
