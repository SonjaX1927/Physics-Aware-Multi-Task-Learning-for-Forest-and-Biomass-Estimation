import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import rasterio
from matplotlib import pyplot as plt


SITES: Tuple[str, ...] = ("bauges", "milicz", "sneznik")
PATCH_SIZE: int = 64
STRIDE: int = 32  # patch 之间的滑动步长
MAX_NODATA_FRACTION: float = 0.8  # 删除 NoData 比例 > 80% 的 patch
BLOCK_SIZE_PATCHES: Tuple[int, int] = (4, 4)  # (行方向 patch 数, 列方向 patch 数)
RANDOM_SEED: int = 42


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_BASE_DIR = PROJECT_ROOT / "@data" / "imaestro"
OUT_DIR = DATA_BASE_DIR / "training-data"


def _stack_site_arrays(site: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取并堆叠一个 site 的 5 个栅格，返回 stack 和全局有效掩膜。

    通道顺序:
    0: VH backscatter (dB)
    1: VV backscatter (dB)
    2: biomass_t_ha_smooth
    3: dom_genus_smooth (数值编码)
    4: height95_smooth
    """

    root = DATA_BASE_DIR / site
    paths = [
        root / "output_backscatter" / f"{site}_vh.tif",
        root / "output_backscatter" / f"{site}_vv.tif",
        root / "output_tiff" / f"{site}_biomass_t_ha_smooth.tif",
        root / "output_tiff" / f"{site}_dom_genus_smooth.tif",
        root / "output_tiff" / f"{site}_height95_smooth.tif",
    ]

    arrays: List[np.ndarray] = []
    valids: List[np.ndarray] = []
    width = height = None

    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing raster for site {site}: {p}")

        with rasterio.open(p) as src:
            arr = src.read(1).astype("float32")
            nodata = src.nodata

        if width is None:
            height, width = arr.shape
        else:
            if arr.shape != (height, width):
                raise ValueError(f"Shape mismatch for {p}: expected {(height, width)}, got {arr.shape}")

        valid = np.ones_like(arr, dtype=bool)
        if nodata is not None:
            valid &= arr != nodata
        valid &= ~np.isnan(arr)

        arrays.append(arr)
        valids.append(valid)

    stack = np.stack(arrays, axis=0)  # (C, H, W)
    valid_all = np.logical_and.reduce(valids)  # (H, W)
    return stack, valid_all


def _extract_patches_for_site(
    site: str,
    stack: np.ndarray,
    valid_mask: np.ndarray,
    patch_size: int,
    stride: int,
    max_nodata_fraction: float,
    block_size_patches: Tuple[int, int],
) -> Tuple[List[np.ndarray], List[Dict]]:
    """从单个 site 的 stack 中提取 patch 及其元信息。"""

    _, height, width = stack.shape

    rows, cols = np.where(valid_mask)
    if rows.size == 0:
        return [], []

    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    patches: List[np.ndarray] = []
    metas: List[Dict] = []

    block_h, block_w = block_size_patches

    # 仅在包含有效像元的范围内滑动窗口，步长由 stride 控制
    patch_row_idx = 0
    row_end = max(row_max - patch_size + 1, row_min)
    for row in range(row_min, row_end + 1, stride):
        if row + patch_size > height:
            continue

        patch_col_idx = 0
        col_end = max(col_max - patch_size + 1, col_min)
        for col in range(col_min, col_end + 1, stride):
            if col + patch_size > width:
                continue

            patch_valid = valid_mask[row : row + patch_size, col : col + patch_size]
            valid_fraction = float(patch_valid.mean())
            if valid_fraction < (1.0 - max_nodata_fraction):
                patch_col_idx += 1
                continue

            patch = stack[:, row : row + patch_size, col : col + patch_size]

            dom_genus = patch[3]  # 通道 3 为 dom_genus_smooth
            dom_valid = ~np.isnan(dom_genus)
            if not dom_valid.any():
                patch_col_idx += 1
                continue

            codes, counts = np.unique(dom_genus[dom_valid].astype("int32"), return_counts=True)
            dominant_code = int(codes[np.argmax(counts)])

            block_row = patch_row_idx // block_h
            block_col = patch_col_idx // block_w
            block_id = f"{site}_br{block_row}_bc{block_col}"

            metas.append(
                {
                    "site": site,
                    "row": int(row),
                    "col": int(col),
                    "patch_row_idx": int(patch_row_idx),
                    "patch_col_idx": int(patch_col_idx),
                    "block_row": int(block_row),
                    "block_col": int(block_col),
                    "block_id": block_id,
                    "valid_fraction": valid_fraction,
                    "dominant_genus": dominant_code,
                    "genus_codes": codes.tolist(),
                    "genus_counts": counts.astype(int).tolist(),
                }
            )
            patches.append(patch)

            patch_col_idx += 1

        patch_row_idx += 1

    return patches, metas


def _split_blocks(
    metas: List[Dict],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = RANDOM_SEED,
) -> Dict[str, List[int]]:
    """按 block 级别划分 train/val/test，返回每个 split 对应的 patch 索引。"""

    from collections import defaultdict

    block_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, m in enumerate(metas):
        block_to_indices[m["block_id"]].append(idx)

    blocks = list(block_to_indices.items())
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)

    total_patches = sum(len(v) for _, v in blocks)
    r_train, r_val, r_test = ratios
    target = {
        "train": int(total_patches * r_train),
        "val": int(total_patches * r_val),
        "test": int(total_patches * r_test),
    }

    splits: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    order = ["train", "val", "test"]

    for block_id, idxs in blocks:
        # 选择当前最“缺”的 split
        candidates = [k for k in order if counts[k] < target[k]]
        if not candidates:
            candidates = order

        def _score(split: str) -> float:
            t = max(target[split], 1)
            return counts[split] / t

        best_split = min(candidates, key=_score)

        splits[best_split].extend(idxs)
        counts[best_split] += len(idxs)

    return splits


def build_training_patches() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_patches: List[np.ndarray] = []
    all_metas: List[Dict] = []

    for site in SITES:
        print(f"Processing site: {site}")
        stack, valid_mask = _stack_site_arrays(site)
        patches, metas = _extract_patches_for_site(
            site,
            stack,
            valid_mask,
            PATCH_SIZE,
            STRIDE,
            MAX_NODATA_FRACTION,
            BLOCK_SIZE_PATCHES,
        )
        print(f"  generated {len(patches)} patches for {site}")
        all_patches.extend(patches)
        all_metas.extend(metas)

    if not all_patches:
        print("No patches generated. Please check input rasters.")
        return

    patches_arr = np.stack(all_patches, axis=0).astype("float32")  # (N, C, H, W)

    splits = _split_blocks(all_metas)

    for split_name, indices in splits.items():
        indices_arr = np.array(indices, dtype="int64")
        split_patches = patches_arr[indices_arr]

        labels = np.array(
            [all_metas[i]["dominant_genus"] for i in indices], dtype="int32"
        )
        sites = np.array([all_metas[i]["site"] for i in indices], dtype=object)

        np.save(OUT_DIR / f"patches_{split_name}.npy", split_patches)
        np.save(OUT_DIR / f"labels_dominant_genus_{split_name}.npy", labels)
        np.save(OUT_DIR / f"sites_{split_name}.npy", sites)

        # 打印每个 split 的概览
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(
            f"Split {split_name}: {len(indices)} patches, "
            f"label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}"
        )


if __name__ == "__main__":
    build_training_patches()
