import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import json
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from rasterio.transform import xy


SITES: Tuple[str, ...] = ("bauges", "milicz", "sneznik")
PATCH_SIZE: int = 64
STRIDE: int = 32  # patch 之间的滑动步长
MAX_NODATA_FRACTION: float = 0.8  # 删除 NoData 比例 > 80% 的 patch
BLOCK_SIZE_PATCHES: Tuple[int, int] = (4, 4)  # (行方向 patch 数, 列方向 patch 数)
RANDOM_SEED: int =  123 # 1, 7, 17, 23, 42, 57, 100, 123, 250, 500


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
    """从单个 site 的 stack 中提取 patch 及其元信息（方案 B：块之间不重叠）。

    方案 B：
    - 在像元坐标上定义不重叠的 block 网格，每个 block 是一个矩形区域；
    - 在每个 block 内部用给定的 stride 提取 patch，且 patch 的范围不会跨出该 block；
    - 不同 block 的空间范围互不重叠，因此后续按 block 划分的 train/val/test 在空间上严格独立。
    """

    _, height, width = stack.shape

    rows, cols = np.where(valid_mask)
    if rows.size == 0:
        return [], []

    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    patches: List[np.ndarray] = []
    metas: List[Dict] = []

    block_h, block_w = block_size_patches

    # 每个 block 在像元坐标中的高度/宽度（确保内部可以放下 block_h × block_w 个 patch，且 patch 不跨 block 边界）
    block_height = (block_h - 1) * stride + patch_size
    block_width = (block_w - 1) * stride + patch_size

    # 在包含有效像元的 ROI 内，所有合法的 patch 左上角行/列的最大值
    global_row_end = max(row_max - patch_size + 1, row_min)
    global_col_end = max(col_max - patch_size + 1, col_min)

    block_row_idx = 0
    block_row_start = row_min
    while block_row_start <= global_row_end:
        # 该 block 允许的最大 patch 起始行（既不超过全局 row_end，也不超过 block 内部范围）
        block_row_limit = block_row_start + (block_h - 1) * stride
        row_limit = min(global_row_end, block_row_limit)

        block_col_idx = 0
        block_col_start = col_min
        while block_col_start <= global_col_end:
            block_col_limit = block_col_start + (block_w - 1) * stride
            col_limit = min(global_col_end, block_col_limit)

            block_id = f"{site}_br{block_row_idx}_bc{block_col_idx}"

            # 在该 block 内部用 stride 提取 patch，且 patch 完全落在 block 范围内
            row = block_row_start
            while row <= row_limit:
                if row + patch_size > height:
                    break

                col = block_col_start
                while col <= col_limit:
                    if col + patch_size > width:
                        break

                    patch_valid = valid_mask[row : row + patch_size, col : col + patch_size]
                    valid_fraction = float(patch_valid.mean())
                    if valid_fraction < (1.0 - max_nodata_fraction):
                        col += stride
                        continue

                    patch = stack[:, row : row + patch_size, col : col + patch_size]

                    dom_genus = patch[3]  # 通道 3 为 dom_genus_smooth
                    dom_valid = ~np.isnan(dom_genus)
                    if not dom_valid.any():
                        col += stride
                        continue

                    codes, counts = np.unique(
                        dom_genus[dom_valid].astype("int32"), return_counts=True
                    )
                    dominant_code = int(codes[np.argmax(counts)])

                    # 为了可读性，仍然记录 patch 在整个 ROI 内的索引（基于 row_min/col_min 和 stride）
                    patch_row_idx = int((row - row_min) // stride)
                    patch_col_idx = int((col - col_min) // stride)

                    metas.append(
                        {
                            "site": site,
                            "row": int(row),
                            "col": int(col),
                            "patch_row_idx": patch_row_idx,
                            "patch_col_idx": patch_col_idx,
                            "block_row": int(block_row_idx),
                            "block_col": int(block_col_idx),
                            "block_id": block_id,
                            "valid_fraction": valid_fraction,
                            "dominant_genus": dominant_code,
                            "genus_codes": codes.tolist(),
                            "genus_counts": counts.astype(int).tolist(),
                        }
                    )
                    patches.append(patch)

                    col += stride

                row += stride

            block_col_idx += 1
            block_col_start += block_width

        block_row_idx += 1
        block_row_start += block_height

    return patches, metas


def _split_blocks(
    metas: List[Dict],
    ratios: Tuple[float, float, float] = (0.75, 0.15, 0.1),
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


def export_split_blocks_geojson(
    ratios: Tuple[float, float, float] = (0.75, 0.15, 0.1),
    seed: int = RANDOM_SEED,
) -> None:
    all_metas: List[Dict] = []

    for site in SITES:
        stack, valid_mask = _stack_site_arrays(site)
        _, metas = _extract_patches_for_site(
            site,
            stack,
            valid_mask,
            PATCH_SIZE,
            STRIDE,
            MAX_NODATA_FRACTION,
            BLOCK_SIZE_PATCHES,
        )
        all_metas.extend(metas)

    if not all_metas:
        print("No metadata generated. Please check input rasters.")
        return

    splits = _split_blocks(all_metas, ratios=ratios, seed=seed)

    block_info: Dict[str, Dict] = {}
    for m in all_metas:
        block_id = m["block_id"]
        site = m["site"]
        row = int(m["row"])
        col = int(m["col"])
        r0 = row
        r1 = row + PATCH_SIZE
        c0 = col
        c1 = col + PATCH_SIZE

        if block_id not in block_info:
            block_info[block_id] = {
                "site": site,
                "row_min": r0,
                "row_max": r1,
                "col_min": c0,
                "col_max": c1,
                "block_row": int(m["block_row"]),
                "block_col": int(m["block_col"]),
            }
        else:
            info = block_info[block_id]
            if site != info["site"]:
                raise ValueError(f"Block {block_id} spans multiple sites.")
            info["row_min"] = min(info["row_min"], r0)
            info["row_max"] = max(info["row_max"], r1)
            info["col_min"] = min(info["col_min"], c0)
            info["col_max"] = max(info["col_max"], c1)

    split_block_ids: Dict[str, Dict[str, int]] = {}
    for split_name, indices in splits.items():
        counts: Dict[str, int] = {}
        for idx in indices:
            m = all_metas[idx]
            bid = m["block_id"]
            counts[bid] = counts.get(bid, 0) + 1
        split_block_ids[split_name] = counts

    site_to_meta: Dict[str, Tuple] = {}
    for site in SITES:
        root = DATA_BASE_DIR / site
        ref_path = root / "output_backscatter" / f"{site}_vh.tif"
        if not ref_path.exists():
            raise FileNotFoundError(f"Missing reference raster for site {site}: {ref_path}")
        with rasterio.open(ref_path) as src:
            site_to_meta[site] = (src.transform, src.crs)

    out_dir = PROJECT_ROOT / "@data" / "auxiliary" / "split-blocks"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, block_counts in split_block_ids.items():
        # Group features by site so that each GeoJSON has a single, correct CRS
        features_by_site: Dict[str, List[Dict]] = {}

        for block_id, num_patches in block_counts.items():
            info = block_info[block_id]
            site = info["site"]
            transform, crs = site_to_meta[site]

            row_min = info["row_min"]
            row_max = info["row_max"]
            col_min = info["col_min"]
            col_max = info["col_max"]

            rows = [row_min, row_min, row_max, row_max]
            cols = [col_min, col_max, col_max, col_min]
            xs, ys = xy(transform, rows, cols, offset="ul")
            coords = list(zip(xs, ys))
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            feature = {
                "type": "Feature",
                "properties": {
                    "block_id": block_id,
                    "site": site,
                    "split": split_name,
                    "num_patches": int(num_patches),
                    "block_row": int(info["block_row"]),
                    "block_col": int(info["block_col"]),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
            }

            if site not in features_by_site:
                features_by_site[site] = []
            features_by_site[site].append(feature)

        # Write one GeoJSON per (split, site) so each layer has a consistent CRS
        for site, features in features_by_site.items():
            _, crs = site_to_meta[site]
            crs_name = str(crs) if crs is not None else None

            geojson_obj: Dict[str, object] = {
                "type": "FeatureCollection",
                "features": features,
            }
            if crs_name is not None:
                geojson_obj["crs"] = {
                    "type": "name",
                    "properties": {"name": crs_name},
                }

            out_path = out_dir / f"{split_name}_{site}_blocks.geojson"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(geojson_obj, f, indent=2)

            print(f"Wrote {len(features)} blocks to {out_path} (CRS={crs_name})")


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
    export_split_blocks_geojson()
